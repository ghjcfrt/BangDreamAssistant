from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from adb_client import AdbClient
from bb_operator import Operator
from vision import TemplateMatcher, decode_png, decode_screencap_raw


@dataclass(frozen=True)
class Thresholds:
    values: Dict[str, float]

    # 获取指定模板的匹配阈值，若不存在则返回默认值
    def get(self, name: str, default: float = 0.9) -> float:
        return float(self.values.get(name, default))


class SharedContext:
    def __init__(
        self,
        adb: AdbClient,
        serial_getter: Callable[[], str],
        matcher: TemplateMatcher,
        thresholds_getter: Callable[[], Thresholds],
        recognition_mode_getter: Callable[[], str],
        log_queue: "queue.Queue[str]",
    ) -> None:
        self.adb = adb
        self.serial_getter = serial_getter
        self.matcher = matcher
        self.thresholds_getter = thresholds_getter
        self.recognition_mode_getter = recognition_mode_getter
        self.log_queue = log_queue
        self.adb_lock = threading.Lock()
        # 重要：TemplateMatcher 内部包含 OCR 引擎与缓存，且底层依赖 OpenCV/onnxruntime。
        # 多线程并发调用在部分环境下可能触发 C 扩展崩溃（表现为“闪退”）。
        # 这里做串行化，保证 OCR/匹配同一时间只跑一份。
        self.vision_lock = threading.Lock()
        self._prefer_raw_screencap = True

    def log(self, msg: str) -> None:
        try:
            self.log_queue.put_nowait(msg)
        except Exception:
            pass

    def _serial(self) -> str:
        serial = self.serial_getter().strip()
        if not serial:
            raise RuntimeError("未选择设备")
        return serial

    def screenshot(self):
        serial = self._serial()
        with self.adb_lock:
            if self._prefer_raw_screencap:
                try:
                    raw = self.adb.screencap_raw(serial)
                    return decode_screencap_raw(raw)
                except Exception as e:
                    # 只降级一次：避免每次都重试 raw 导致额外开销
                    self._prefer_raw_screencap = False
                    self.log(f"raw screencap 不可用，回退 PNG：{e}")

            png = self.adb.screencap_png(serial)
        return decode_png(png)

    def tap(self, x: int, y: int) -> None:
        serial = self._serial()
        with self.adb_lock:
            self.adb.tap(serial, x, y)

    def recognition_mode(self) -> str:
        raw = (self.recognition_mode_getter() or "template").strip().lower()
        return raw if raw in {"template", "ocr"} else "template"

    def find(self, screen_bgr, name: str):
        with self.vision_lock:
            mode = self.recognition_mode()
            allow_ocr = (mode == "ocr")
            return self.matcher.find_best(screen_bgr, name, allow_ocr=allow_ocr)

    def find2(self, screen_bgr, name: str, *, allow_ocr: bool = True):
        with self.vision_lock:
            return self.matcher.find_best(screen_bgr, name, allow_ocr=allow_ocr)

    def find_template_only(self, screen_bgr, name: str):
        with self.vision_lock:
            return self.matcher.find_template_only(screen_bgr, name)

    def find_ocr_only(self, screen_bgr, name: str):
        with self.vision_lock:
            return self.matcher.find_ocr_only(screen_bgr, name)


class RealtimeWorker:
    def __init__(self, ctx: SharedContext) -> None:
        self.ctx = ctx
        self.operator = Operator(ctx)
        self._enabled = threading.Event()
        self._terminate = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_best_log: Dict[str, float] = {}
        # 上一次实际点击到的按钮：用于下一轮调整优先级（menu -> skip1 -> skip2 -> menu）
        self._last_clicked: Optional[str] = None
        self._last_click_ts: Dict[str, float] = {}
        # menu 按钮在多数界面里“常驻”，如果 skip1/skip2 未识别到会导致反复点 menu。
        # 加一个轻量冷却，避免短时间内连续重复点击。
        self._click_cooldown_s: Dict[str, float] = {"menu": 1.2}
        # 限制实时识别频率：减少截图 + OCR 负载
        self._min_cycle_s: float = 0.25
        # OCR 作为保底：每个按钮最多 1s 触发一次 OCR
        self._ocr_min_interval_s: float = 1.0
        self._last_ocr_try: Dict[str, float] = {}

    def _priority_names(self) -> tuple[str, ...]:
        base: tuple[str, str, str] = ("menu", "skip1", "skip2")
        if self._last_clicked not in base:
            return base
        idx = base.index(self._last_clicked)
        # 轮转：上次点了谁，下次就优先检查它的下一个
        return base[idx + 1 :] + base[: idx + 1]

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            self._enabled.set()
            return
        # 允许重复 start：如果上次线程已结束，重置 terminate 标志
        self._terminate.clear()
        self._enabled.set()
        self._thread = threading.Thread(target=self._run, name="realtime-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._enabled.clear()

    def shutdown(self, timeout_s: float = 2.0) -> None:
        """彻底停止线程（用于程序退出）。"""
        self._enabled.clear()
        self._terminate.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=max(0.0, float(timeout_s)))

    def is_enabled(self) -> bool:
        return self._enabled.is_set()

    def _run(self) -> None:
        self.ctx.log("实时触发：已启动")
        while not self._terminate.is_set():
            if not self._enabled.is_set():
                # 等待启用或退出
                self._terminate.wait(0.15)
                continue
            try:
                cycle_start = time.perf_counter()
                # 关键性能点：原实现每轮会对 menu/skip1/skip2 各截图一次（timeout=0 也会截图），
                # ADB screencap + PNG 解码会明显拖慢模拟器渲染。
                # 这里改为“一轮截图一次，在同一张图上匹配多个模板”，并保持原优先级。

                screen = self.ctx.screenshot()
                thresholds = self.ctx.thresholds_getter()
                mode = self.ctx.recognition_mode()

                for name in self._priority_names():
                    if self._terminate.is_set() or (not self._enabled.is_set()):
                        break
                    now = time.time()
                    last = float(self._last_best_log.get(name, 0.0))
                    log_best = (now - last) >= 1.0
                    if log_best:
                        self._last_best_log[name] = now

                    th = thresholds.get(name, 0.90)

                    # 点击冷却：避免同一个按钮在短时间内被反复点（尤其是 menu 常驻场景）
                    cd = float(self._click_cooldown_s.get(name, 0.0))
                    if cd > 0:
                        last_click = float(self._last_click_ts.get(name, 0.0))
                        if (now - last_click) < cd:
                            continue

                    m = None
                    if mode == "template":
                        # 模板识别：每轮都跑（轻）
                        m = self.ctx.find_template_only(screen, name)
                    else:
                        # OCR 识别：很重，降频触发（每个按钮最多 1s 一次）
                        ocr_last = float(self._last_ocr_try.get(name, 0.0))
                        if (now - ocr_last) >= self._ocr_min_interval_s:
                            self._last_ocr_try[name] = now
                            m = self.ctx.find_ocr_only(screen, name)

                    if m is not None and m.score >= th:
                        self.operator.click_box(m, after_sleep=0.1)
                        self._last_clicked = name
                        self._last_click_ts[name] = time.time()
                        break

                    # 失败也输出最高置信度(best)的日志（限流到每秒最多一次）
                    if log_best:
                        if m is None:
                            self.ctx.log(f"wait_img 失败：{name} best=N/A th={th:.3f} timeout=0.0s")
                            # 输出 ROI/过程图，便于排查为什么一直 N/A
                            self.ctx.matcher.debug_dump_if_enabled(screen, name, reason="best_na", mode=mode)
                        else:
                            x, y = m.center
                            self.ctx.log(
                                f"wait_img 失败：{name} best={m.score:.3f} th={th:.3f} at=({x},{y}) timeout=0.0s"
                            )

                elapsed = time.perf_counter() - cycle_start
                sleep_s = self._min_cycle_s - float(elapsed)
                if sleep_s > 0:
                    # 允许在 sleep 期间快速退出
                    self._terminate.wait(sleep_s)
            except Exception as e:
                self.ctx.log(f"实时触发异常：{e}")
                self._terminate.wait(0.5)


class TaskWorker:
    def __init__(self, ctx: SharedContext) -> None:
        self.ctx = ctx
        self.operator = Operator(ctx)
        self._running = threading.Event()
        self._terminate = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            self._running.set()
            return
        self._terminate.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="task-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        self.ctx.log("任务触发：停止信号已发送")

    def shutdown(self, timeout_s: float = 2.0) -> None:
        """彻底停止线程（用于程序退出）。"""
        self._running.clear()
        self._terminate.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=max(0.0, float(timeout_s)))

    def is_running(self) -> bool:
        return self._running.is_set()

    def _run_once(self) -> None:
        # 轻量优化：同一轮里如果没有点击 read，就复用截图检测 no_voice，减少一次 ADB screencap。
        screen = self.ctx.screenshot()
        thresholds = self.ctx.thresholds_getter()

        read_box = self.ctx.find(screen, "read")
        read_th = thresholds.get("read", 0.90)
        clicked_read = False
        if read_box is not None and read_box.score >= read_th and self._running.is_set():
            self.operator.click_box(read_box, after_sleep=0.5)
            clicked_read = True

        if clicked_read:
            screen = self.ctx.screenshot()

        no_voice_box = self.ctx.find(screen, "no_voice")
        no_voice_th = thresholds.get("no_voice", 0.90)
        if no_voice_box is not None and no_voice_box.score >= no_voice_th and self._running.is_set():
            self.operator.click_box(no_voice_box, after_sleep=0.1)

        # 等待实时触发跳过（可在等待期间快速退出）
        self._terminate.wait(7.0)

        # 部分剧情存在确认按钮：这里必须刷新截图。
        # 否则会在“等待前”的旧画面上找 enter，导致永远找不到。
        enter_th = thresholds.get("enter", 0.9)
        max_enter_clicks = 8
        for _ in range(max_enter_clicks):
            if self._terminate.is_set() or (not self._running.is_set()):
                break

            screen = self.ctx.screenshot()
            enter_box = self.ctx.find(screen, "enter")
            if enter_box is None:
                break
            if enter_box.score < enter_th:
                break

            self.operator.click_box(enter_box, after_sleep=0.1)
            # 给 UI 一点时间刷新（同时可快速退出）
            self._terminate.wait(0.35)
        
        # 允许退出时快速中断等待
        self._terminate.wait(2.0)

    def _run(self) -> None:
        self.ctx.log("任务触发：已启动")
        while (not self._terminate.is_set()) and self._running.is_set():
            try:
                self._run_once()
                locked = self.operator.exists_img("lock")
                if locked:
                    # lock 存在：重复该任务
                    continue

                # lock 不存在：再执行一轮，然后停止
                self.ctx.log("lock 未检测到：额外再执行一轮后停止")
                self._run_once()
                self.ctx.log("任务触发：已停止（lock 未检测到）")
                self._running.clear()
            except Exception as e:
                self.ctx.log(f"任务触发异常：{e}")
                self._terminate.wait(0.5)
                self.ctx.log(f"任务触发异常：{e}")
                self._terminate.wait(0.5)
