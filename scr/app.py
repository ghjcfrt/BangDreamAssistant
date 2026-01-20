from __future__ import annotations

import faulthandler
import os
import queue
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from adb_client import AdbClient
from vision import TemplateMatcher
from workers import RealtimeWorker, SharedContext, TaskWorker, Thresholds


@dataclass(frozen=True)
class TaskSpec:
    key: str
    name: str
    description: str


class AppWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BBAssistant - ADB 触发器")
        self.resize(980, 560)

        self.log_queue: "queue.Queue[str]" = queue.Queue()

        # 启动自检：便于定位是否误用到 uv 的 Python 3.13.0 等解释器
        try:
            self.log(f"Python: {sys.version.splitlines()[0]} | exe: {sys.executable}")
            if sys.version_info[:3] == (3, 13, 0):
                self.log("提示：检测到 Python 3.13.0；若退出时报 threading shutdown SystemError，建议升级到 3.13.1+ 或使用 3.12")
        except Exception:
            pass

        base_dir = Path(__file__).resolve().parent.parent
        img_dir = base_dir / "img"

        self.adb = AdbClient(adb_path="adb")
        self.matcher = TemplateMatcher(img_dir)
        self.matcher.load_defaults()

        self._build_ui()

        self.ctx = SharedContext(
            adb=self.adb,
            serial_getter=self._get_serial,
            matcher=self.matcher,
            thresholds_getter=self._get_thresholds,
            log_queue=self.log_queue,
        )

        self.realtime_worker = RealtimeWorker(self.ctx)
        self.task_worker = TaskWorker(self.ctx)

        self._shot_thread: Optional["_ScreenshotThread"] = None
        self._shot_dialog: Optional["_ScreenshotPreviewDialog"] = None

        self._log_timer = QtCore.QTimer(self)
        self._log_timer.setInterval(120)
        self._log_timer.timeout.connect(self._drain_logs)
        self._log_timer.start()

        self._state_timer = QtCore.QTimer(self)
        self._state_timer.setInterval(250)
        self._state_timer.timeout.connect(self._sync_states)
        self._state_timer.start()

        self.refresh_devices()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        left = QtWidgets.QVBoxLayout()
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)
        root.addLayout(left, 3)
        root.addLayout(right, 1)

        # ADB/设备
        gb_device = QtWidgets.QGroupBox("ADB/设备")
        left.addWidget(gb_device)
        form = QtWidgets.QGridLayout(gb_device)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.adb_path_edit = QtWidgets.QLineEdit("adb")
        self.adb_apply_btn = QtWidgets.QPushButton("应用")
        self.adb_apply_btn.clicked.connect(self.apply_adb_path)

        form.addWidget(QtWidgets.QLabel("ADB 路径："), 0, 0)
        form.addWidget(self.adb_path_edit, 0, 1)
        form.addWidget(self.adb_apply_btn, 0, 2)

        self.addr_edit = QtWidgets.QLineEdit("127.0.0.1:16384")
        self.connect_btn = QtWidgets.QPushButton("connect")
        self.connect_btn.clicked.connect(self.connect_mumu)
        form.addWidget(QtWidgets.QLabel("MuMu 地址："), 1, 0)
        form.addWidget(self.addr_edit, 1, 1)
        form.addWidget(self.connect_btn, 1, 2)

        self.device_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_devices)
        form.addWidget(QtWidgets.QLabel("设备："), 2, 0)
        form.addWidget(self.device_combo, 2, 1)
        form.addWidget(self.refresh_btn, 2, 2)

        # 触发模式
        gb_mode = QtWidgets.QGroupBox("触发模式")
        left.addWidget(gb_mode)
        v_mode = QtWidgets.QVBoxLayout(gb_mode)

        self.realtime_check = QtWidgets.QCheckBox("实时触发（后台持续检测并点击 menu → skip1 → skip2）")
        self.realtime_check.toggled.connect(self.on_toggle_realtime)
        v_mode.addWidget(self.realtime_check)

        # 任务列表（可扩展）
        gb_tasks = QtWidgets.QGroupBox("任务触发")
        v_mode.addWidget(gb_tasks)
        v_tasks = QtWidgets.QVBoxLayout(gb_tasks)
        v_tasks.setSpacing(6)

        self._task_specs = [
            TaskSpec(
                key="task_read_no_voice_lock",
                name="读剧情/对话",
                description=""
            )
        ]

        self._task_buttons: dict[str, QtWidgets.QPushButton] = {}

        for spec in self._task_specs:
            row = QtWidgets.QHBoxLayout()
            name_label = QtWidgets.QLabel(spec.name)
            name_label.setToolTip(spec.description)
            row.addWidget(name_label)
            row.addStretch(1)
            btn = QtWidgets.QPushButton("开始")
            btn.setFixedWidth(90)
            btn.clicked.connect(lambda _=False, key=spec.key: self._toggle_task(key))
            row.addWidget(btn)
            v_tasks.addLayout(row)

            desc = QtWidgets.QLabel(spec.description)
            desc.setStyleSheet("color: #666;")
            desc.setWordWrap(True)
            v_tasks.addWidget(desc)

            self._task_buttons[spec.key] = btn

        # 阈值
        gb_th = QtWidgets.QGroupBox("相似度阈值（0~1，越小越容易触发）")
        left.addWidget(gb_th)
        grid = QtWidgets.QGridLayout(gb_th)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        def th_row(row: int, left_name: str, right_name: str):
            l1 = QtWidgets.QLabel(left_name)
            e1 = QtWidgets.QLineEdit()
            e1.setFixedWidth(90)
            l2 = QtWidgets.QLabel(right_name)
            e2 = QtWidgets.QLineEdit()
            e2.setFixedWidth(90)
            grid.addWidget(l1, row, 0)
            grid.addWidget(e1, row, 1)
            grid.addWidget(l2, row, 2)
            grid.addWidget(e2, row, 3)
            return e1, e2

        self.th_menu, self.th_read = th_row(0, "menu", "read")
        self.th_skip1, self.th_no_voice = th_row(1, "skip1", "no_voice")
        self.th_skip2, self.th_lock = th_row(2, "skip2", "lock")

        self.reset_th_btn = QtWidgets.QPushButton("阈值恢复默认")
        self.reset_th_btn.clicked.connect(self.reset_thresholds)
        grid.addWidget(self.reset_th_btn, 3, 0, 1, 4, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        self.reset_thresholds()

        # 日志
        gb_log = QtWidgets.QGroupBox("日志")
        left.addWidget(gb_log, 1)
        v_log = QtWidgets.QVBoxLayout(gb_log)
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QtGui.QFont("Consolas", 10))
        v_log.addWidget(self.log_text)

        # 右侧提示
        gb_debug = QtWidgets.QGroupBox("调试")
        right.addWidget(gb_debug)
        v_debug = QtWidgets.QVBoxLayout(gb_debug)
        v_debug.setSpacing(8)

        # OCR 加速选择（CPU / DirectML / CUDA）
        accel_row = QtWidgets.QHBoxLayout()
        accel_row.addWidget(QtWidgets.QLabel("OCR 加速："))
        self.ocr_accel_combo = QtWidgets.QComboBox()
        self.ocr_accel_combo.addItems(["CPU", "DirectML", "CUDA"])
        self.ocr_accel_combo.setToolTip(
            "选择 OCR 推理加速方式。\n"
            "- DirectML：默认（Windows 推荐，免装 CUDA/cuDNN）\n"
            "- CPU：兼容性最好\n"
            "- CUDA：需要安装 CUDA 12.x + cuDNN 9.x"
        )
        self.ocr_accel_combo.currentIndexChanged.connect(self.on_change_ocr_accel)
        accel_row.addWidget(self.ocr_accel_combo, 1)
        v_debug.addLayout(accel_row)

        self.ocr_accel_hint = QtWidgets.QLabel("")
        self.ocr_accel_hint.setStyleSheet("color: #666;")
        self.ocr_accel_hint.setWordWrap(True)
        v_debug.addWidget(self.ocr_accel_hint)

        self.screenshot_btn = QtWidgets.QPushButton("截屏测试")
        self.screenshot_btn.clicked.connect(self.on_screenshot_test)
        v_debug.addWidget(self.screenshot_btn)

        self.screenshot_hint = QtWidgets.QLabel("点击后会抓取一张当前屏幕截图，并在新窗口预览")
        self.screenshot_hint.setStyleSheet("color: #666;")
        self.screenshot_hint.setWordWrap(True)
        v_debug.addWidget(self.screenshot_hint)

        gb_tip = QtWidgets.QGroupBox("提示")
        right.addWidget(gb_tip)
        v_tip = QtWidgets.QVBoxLayout(gb_tip)
        tip = (
            "1) 先确保电脑能直接运行 adb（或填入 adb.exe 路径）\n"
            "2) MuMu 端口以模拟器设置为准；常见：127.0.0.1:16384\n"
            "3) 实时触发与任务触发可同时运行（互不阻塞）\n"
            "4) 如误触发，调高阈值；如识别不到，调低阈值\n"
        )
        tip_label = QtWidgets.QLabel(tip)
        tip_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        v_tip.addWidget(tip_label)
        v_tip.addStretch(1)

        # UI 初始化结束后同步一次 OCR 加速状态
        self._sync_ocr_accel_ui()

    def _sync_ocr_accel_ui(self) -> None:
        raw = (os.environ.get("BB_OCR_ACCEL") or "dml").strip().lower()
        mapping = {"cpu": 0, "dml": 1, "cuda": 2}
        mode = raw if raw in mapping else "dml"

        # 关键：把 UI 选项对应的模式写回环境变量，确保首次 OCR 初始化就按该模式创建。
        # 否则 RapidOCR 在第一次被调用时可能会按其默认值走 CPU，导致需要“切换一次再切回”才生效。
        os.environ["BB_OCR_ACCEL"] = mode

        idx = mapping.get(mode, 0)
        self.ocr_accel_combo.blockSignals(True)
        self.ocr_accel_combo.setCurrentIndex(idx)
        self.ocr_accel_combo.blockSignals(False)

        # 尽量展示当前可用 provider（不强依赖 onnxruntime）
        providers_text = ""
        try:
            import onnxruntime as ort  # type: ignore

            providers_text = ", ".join(ort.get_available_providers())
        except Exception:
            providers_text = "(未安装/无法导入 onnxruntime)"

        self.ocr_accel_hint.setText(f"当前：{mode.upper()} | providers: {providers_text}")

    def on_change_ocr_accel(self) -> None:
        text = (self.ocr_accel_combo.currentText() or "CPU").strip().lower()
        mode = "dml"
        if "cpu" in text:
            mode = "cpu"
        elif "cuda" in text:
            mode = "cuda"

        os.environ["BB_OCR_ACCEL"] = mode

        # 让新配置立刻生效：清理 OCR 引擎缓存，下次 OCR 会按新 mode 初始化
        try:
            self.matcher._ocr_engine = None  # type: ignore[attr-defined]
        except Exception:
            pass

        self.log(f"OCR 加速已切换为：{mode}（下次 OCR 调用生效）")
        self._sync_ocr_accel_ui()

    def _get_serial(self) -> str:
        return self.device_combo.currentText().strip()

    def _parse_threshold(self, text: str, default: float) -> float:
        try:
            return float(text.strip())
        except Exception:
            return default

    def _get_thresholds(self) -> Thresholds:
        return Thresholds(
            values={
                # 置信度已统一为 0~1（详见 vision.py），默认阈值相应上调。
                # menu/skip* 默认走 OCR：置信度通常比模板匹配略低，阈值相应下调
                "menu": self._parse_threshold(self.th_menu.text(), 0.60),
                "skip1": self._parse_threshold(self.th_skip1.text(), 0.60),
                "skip2": self._parse_threshold(self.th_skip2.text(), 0.60),
                "read": self._parse_threshold(self.th_read.text(), 0.60),
                "no_voice": self._parse_threshold(self.th_no_voice.text(), 0.60),
                # lock 明确走灰度模板匹配（灰度优先），分数通常比边缘匹配更保守
                "lock": self._parse_threshold(self.th_lock.text(), 0.90),
            }
        )

    def log(self, msg: str) -> None:
        try:
            self.log_queue.put_nowait(msg)
        except Exception:
            pass

    def _drain_logs(self) -> None:
        changed = False
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.append(msg)
            changed = True
        if changed:
            c = self.log_text.textCursor()
            c.movePosition(QtGui.QTextCursor.MoveOperation.End)
            self.log_text.setTextCursor(c)

    def apply_adb_path(self) -> None:
        path = self.adb_path_edit.text().strip() or "adb"
        self.adb = AdbClient(adb_path=path)
        self.ctx.adb = self.adb
        self.log(f"已应用 ADB 路径：{path}")
        self.refresh_devices()

    def connect_mumu(self) -> None:
        addr = self.addr_edit.text().strip()
        if not addr:
            self.log("MuMu 地址为空")
            return
        try:
            out = self.adb.connect(addr)
            self.log(f"adb connect 输出：{out}")
        except Exception as e:
            self.log(f"connect 失败：{e}")
        self.refresh_devices()

    def refresh_devices(self) -> None:
        try:
            devices = self.adb.list_devices()
            serials = [d.serial for d in devices]
            items = [f"{d.serial} ({d.status})" for d in devices]

            current = self._get_serial()
            self.device_combo.blockSignals(True)
            self.device_combo.clear()
            self.device_combo.addItems(serials)
            if current and current in serials:
                self.device_combo.setCurrentText(current)
            self.device_combo.blockSignals(False)

            self.log(f"已发现设备：{', '.join(items) if items else '无'}")
        except Exception as e:
            self.log(f"刷新设备失败：{e}")

    def reset_thresholds(self) -> None:
        self.th_menu.setText("0.60")
        self.th_skip1.setText("0.60")
        self.th_skip2.setText("0.60")
        self.th_read.setText("0.60")
        self.th_no_voice.setText("0.60")
        self.th_lock.setText("0.90")
        self.log("阈值已恢复默认")

    def _ensure_screenshot_dialog(self) -> "_ScreenshotPreviewDialog":
        if self._shot_dialog is None:
            self._shot_dialog = _ScreenshotPreviewDialog(parent=self)
        return self._shot_dialog

    @QtCore.Slot()
    def on_screenshot_test(self) -> None:
        try:
            _ = self._get_serial()
        except Exception as e:
            self.log(f"截屏测试失败：{e}")
            return

        dlg = self._ensure_screenshot_dialog()
        dlg.set_status_text("正在截屏...")
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

        if self._shot_thread is not None:
            try:
                if self._shot_thread.isRunning():
                    self.log("截屏测试：正在进行中")
                    return
            except RuntimeError:
                # QThread 已被 Qt deleteLater/父对象回收，但 Python 仍持有 wrapper
                self._shot_thread = None

        self.screenshot_btn.setEnabled(False)
        self.log("截屏测试：开始")

        t = _ScreenshotThread(self.ctx, parent=self)
        t.success.connect(self._on_screenshot_ready)
        t.failure.connect(lambda msg: self.log(f"截屏测试失败：{msg}"))
        t.finished.connect(self._on_screenshot_thread_finished)
        t.finished.connect(t.deleteLater)
        self._shot_thread = t
        t.start()

    @QtCore.Slot()
    def _on_screenshot_thread_finished(self) -> None:
        # 线程结束时：恢复按钮 + 释放对已 deleteLater 对象的引用，避免二次点击崩溃
        self.screenshot_btn.setEnabled(True)
        try:
            sender = self.sender()
        except Exception:
            sender = None
        if sender is None or self._shot_thread is sender:
            self._shot_thread = None

    @QtCore.Slot(object)
    def _on_screenshot_ready(self, screen_bgr: np.ndarray) -> None:
        try:
            # screen_bgr: numpy ndarray (BGR)
            h = int(getattr(screen_bgr, "shape")[0])
            w = int(getattr(screen_bgr, "shape")[1])

            rgb = screen_bgr[..., ::-1].copy()
            bytes_per_line = int(rgb.shape[1] * rgb.shape[2])
            qimg = QtGui.QImage(
                rgb.data,
                int(rgb.shape[1]),
                int(rgb.shape[0]),
                bytes_per_line,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
            pix = QtGui.QPixmap.fromImage(qimg)

            dlg = self._ensure_screenshot_dialog()
            dlg.setWindowTitle(f"截图预览（{w}x{h}）")
            dlg.set_pixmap(pix)
            self.log(f"截屏测试：完成（{w}x{h}）")
        except Exception as e:
            dlg = self._ensure_screenshot_dialog()
            dlg.set_status_text("(截图显示失败)")
            self.log(f"截屏测试：显示失败：{e}")

    @QtCore.Slot(bool)
    def on_toggle_realtime(self, enabled: bool) -> None:
        if enabled:
            self.realtime_worker.start()
        else:
            self.realtime_worker.stop()
            self.log("实时触发：已暂停")

    def start_task(self) -> None:
        self.task_worker.start()

    def stop_task(self) -> None:
        self.task_worker.stop()

    def _toggle_task(self, key: str) -> None:
        # 目前只有一个 TaskWorker；后续新增任务时可以在这里做分发
        if key != "task_read_no_voice_lock":
            self.log(f"未知任务：{key}")
            return

        if self.task_worker.is_running():
            self.task_worker.stop()
        else:
            self.task_worker.start()
        self._sync_states()

    def _sync_states(self) -> None:
        # 同步 UI 状态：任务可能会自行停止（lock 未检测到时）
        btn = self._task_buttons.get("task_read_no_voice_lock")
        if btn is not None:
            btn.setText("停止" if self.task_worker.is_running() else "开始")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.realtime_check.setChecked(False)
            # 关键：关闭窗口时要彻底停掉后台线程并 join，避免解释器退出阶段残留线程
            self.realtime_worker.shutdown(timeout_s=2.0)
            self.task_worker.shutdown(timeout_s=2.0)

            # 截屏测试线程如果仍在跑，尽量等待其结束（它本身是一次性任务）
            t = self._shot_thread
            if t is not None:
                try:
                    if t.isRunning():
                        t.wait(800)
                except Exception:
                    pass
        finally:
            super().closeEvent(event)


class _ScreenshotThread(QtCore.QThread):
    success = QtCore.Signal(object)
    failure = QtCore.Signal(str)

    def __init__(self, ctx: SharedContext, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._ctx = ctx

    def run(self) -> None:
        try:
            img = self._ctx.screenshot()
            self.success.emit(img)
        except Exception as e:
            self.failure.emit(str(e))


class _ScreenshotPreviewDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("截图预览")
        self.resize(520, 900)

        self._last_pix: Optional[QtGui.QPixmap] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        btn_row = QtWidgets.QHBoxLayout()
        self._save_btn = QtWidgets.QPushButton("保存截图")
        self._save_btn.clicked.connect(self._save_screenshot)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch(1)
        self._save_hint = QtWidgets.QLabel("（保存到 screenshots/ 目录）")
        self._save_hint.setStyleSheet("color: #666;")
        btn_row.addWidget(self._save_hint)
        layout.addLayout(btn_row)

        self._view = _ScaledImageView()
        layout.addWidget(self._view, 1)

    def set_status_text(self, text: str) -> None:
        self._last_pix = None
        self._view.set_status_text(text)

    def set_pixmap(self, pix: QtGui.QPixmap) -> None:
        self._last_pix = pix
        self._view.set_pixmap(pix)

    def _save_screenshot(self) -> None:
        if self._last_pix is None or self._last_pix.isNull():
            self.set_status_text("(暂无可保存的截图)")
            return

        base_dir = Path(__file__).resolve().parent.parent
        out_dir = base_dir / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        w = int(self._last_pix.width())
        h = int(self._last_pix.height())
        out_path = out_dir / f"shot_{ts}_{w}x{h}.png"

        ok = self._last_pix.save(str(out_path), "PNG")
        if ok:
            self._save_hint.setText(f"已保存：{out_path.name}")
        else:
            self._save_hint.setText("保存失败")


class _ScaledImageView(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pix: Optional[QtGui.QPixmap] = None
        self._status_text = "(暂无截图)"
        self.setMinimumSize(280, 180)
        self.setAutoFillBackground(False)

    def set_status_text(self, text: str) -> None:
        self._pix = None
        self._status_text = text
        self.update()

    def set_pixmap(self, pix: QtGui.QPixmap) -> None:
        self._pix = pix
        self._status_text = ""
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor("#111"))
        pen = QtGui.QPen(QtGui.QColor("#333"))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        if self._pix is None or self._pix.isNull():
            painter.setPen(QtGui.QColor("#ddd"))
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, self._status_text)
            return

        scaled = self._pix.scaled(
            rect.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        x = int((rect.width() - scaled.width()) / 2)
        y = int((rect.height() - scaled.height()) / 2)
        painter.drawPixmap(x, y, scaled)


def main() -> None:
    # 如果出现“闪退”（例如 C 扩展崩溃），尽量打印出 Python 栈，方便进一步定位。
    # 也可通过环境变量 PYTHONFAULTHANDLER=1 开启。
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    app = QtWidgets.QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
