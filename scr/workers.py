from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from adb_client import AdbClient
from bb_operator import Operator
from vision import Match, TemplateMatcher, decode_png, decode_screencap_raw


@dataclass(frozen=True)
class Thresholds:
    values: Dict[str, float]

    # 获取指定模板的匹配阈值，若不存在则返回默认值
    def get(self, name: str, default: float = 0.9) -> float:
        return float(self.values.get(name, default))


class SharedContext:
    _ocr_only_names = {"menu", "read"}

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
        self._task_exclusive = threading.Event()
        self._prefer_raw_screencap = True
        self._screencap_mode = (
            os.environ.get("BB_SCREENCAP_MODE") or "auto"
        ).strip().lower()
        if self._screencap_mode not in {"auto", "raw", "png"}:
            self._screencap_mode = "auto"

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

    def _should_try_raw_screencap(self, serial: str) -> bool:
        if self._screencap_mode == "png":
            return False
        if self._screencap_mode == "raw":
            return self._prefer_raw_screencap
        # TCP ADB 模拟器上 raw screencap 更容易阻塞；auto 下直接用 PNG。
        if ":" in serial:
            return False
        return self._prefer_raw_screencap

    def screenshot(self):
        serial = self._serial()
        with self.adb_lock:
            if self._should_try_raw_screencap(serial):
                try:
                    raw = self.adb.screencap_raw(serial)
                    return decode_screencap_raw(raw)
                except Exception as e:
                    # 只降级一次：避免每次都重试 raw 导致额外开销
                    self._prefer_raw_screencap = False
                    self.log(f"raw screencap 已切换为 PNG：{e}")

            png = self.adb.screencap_png(serial)
        return decode_png(png)

    def tap(self, x: int, y: int) -> None:
        serial = self._serial()
        with self.adb_lock:
            self.adb.tap(serial, x, y)

    def recognition_mode(self) -> str:
        raw = (self.recognition_mode_getter() or "template").strip().lower()
        return raw if raw in {"template", "ocr"} else "template"

    def begin_task_exclusive(self) -> None:
        self._task_exclusive.set()

    def end_task_exclusive(self) -> None:
        self._task_exclusive.clear()

    def is_task_exclusive_active(self) -> bool:
        return self._task_exclusive.is_set()

    def is_ocr_only(self, name: str) -> bool:
        return str(name) in self._ocr_only_names

    def find(self, screen_bgr, name: str):
        with self.vision_lock:
            if self.is_ocr_only(name):
                return self.matcher.find_ocr_only(screen_bgr, name)
            mode = self.recognition_mode()
            allow_ocr = mode == "ocr"
            return self.matcher.find_best(screen_bgr, name, allow_ocr=allow_ocr)

    def find2(self, screen_bgr, name: str, *, allow_ocr: bool = True):
        with self.vision_lock:
            if self.is_ocr_only(name):
                return self.matcher.find_ocr_only(screen_bgr, name)
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
        # 限制实时循环频率：减少截图负载
        self._min_cycle_s: float = 0.25

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
        self._thread = threading.Thread(
            target=self._run, name="realtime-worker", daemon=True
        )
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
            if self.ctx.is_task_exclusive_active():
                # “读剧情/对话”任务运行期间独占识别/点击流程，确保先从 lock 开始。
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
                    attempted = False
                    detect_mode = "ocr" if self.ctx.is_ocr_only(name) else mode
                    if detect_mode == "template":
                        # 模板识别：每轮都跑（轻）
                        m = self.ctx.find_template_only(screen, name)
                        attempted = True
                    else:
                        m = self.ctx.find_ocr_only(screen, name)
                        attempted = True

                    if attempted:
                        self.operator.log_detection_result(
                            name, m, th, mode=detect_mode
                        )

                    if m is not None and m.score >= th:
                        if self.ctx.is_task_exclusive_active():
                            break
                        self.operator.click_box(m, after_sleep=0.1)
                        self._last_clicked = name
                        self._last_click_ts[name] = time.time()
                        break

                    # 调试落盘仍按节流控制，只在真正执行了识别且没有命中时保存过程图。
                    if attempted and log_best and m is None:
                        self.ctx.matcher.debug_dump_if_enabled(
                            screen, name, reason="best_na", mode=detect_mode
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
        self._state_lock = threading.Lock()
        self._run_stop: Optional[threading.Event] = None

    def start(self) -> None:
        with self._state_lock:
            if self._running.is_set():
                return

            old_stop = self._run_stop
            if old_stop is not None:
                old_stop.set()

            self._terminate.clear()
            self._running.set()
            self.ctx.begin_task_exclusive()
            run_stop = threading.Event()
            self._run_stop = run_stop
            self._thread = threading.Thread(
                target=self._run, args=(run_stop,), name="task-worker", daemon=True
            )
        self._thread.start()

    def stop(self) -> None:
        with self._state_lock:
            self._running.clear()
            run_stop = self._run_stop
            if run_stop is not None:
                run_stop.set()
            self.ctx.end_task_exclusive()
        self.ctx.log("任务触发：停止信号已发送")

    def shutdown(self, timeout_s: float = 2.0) -> None:
        """彻底停止线程（用于程序退出）。"""
        with self._state_lock:
            self._running.clear()
            self._terminate.set()
            run_stop = self._run_stop
            if run_stop is not None:
                run_stop.set()
            self.ctx.end_task_exclusive()
            t = self._thread
        if t and t.is_alive():
            t.join(timeout=max(0.0, float(timeout_s)))

    def is_running(self) -> bool:
        return self._running.is_set()

    def _is_active(self, run_stop: threading.Event) -> bool:
        return (
            (not self._terminate.is_set())
            and (not run_stop.is_set())
            and self._running.is_set()
            and self._run_stop is run_stop
        )

    def _target_mode(self, name: str) -> str:
        if self.ctx.is_ocr_only(name):
            return "ocr"

        mode = self.ctx.recognition_mode()
        if mode == "ocr" and name in getattr(self.ctx.matcher, "_ocr_patterns", {}):
            return "ocr"
        return "template"

    def _find_once(
        self, screen, name: str, thresholds: Thresholds
    ) -> tuple[Optional[Match], float, str]:
        th = thresholds.get(name, 0.90)
        mode = self._target_mode(name)
        if mode == "ocr":
            match = self.ctx.find_ocr_only(screen, name)
        else:
            match = self.ctx.find_template_only(screen, name)
        self.operator.log_detection_result(name, match, th, mode=mode)
        return match, th, mode

    def _wait_and_click(
        self,
        run_stop: threading.Event,
        name: str,
        thresholds: Thresholds,
        *,
        timeout_s: Optional[float],
        interval_s: float = 0.5,
        after_sleep: float = 0.1,
    ) -> bool:
        deadline = None if timeout_s is None else time.time() + max(0.0, timeout_s)
        while self._is_active(run_stop):
            screen = self.ctx.screenshot()
            match, th, _ = self._find_once(screen, name, thresholds)
            if match is not None and match.score >= th and self._is_active(run_stop):
                self.operator.click_box(match, after_sleep=after_sleep)
                return True

            if deadline is not None and time.time() >= deadline:
                return False
            run_stop.wait(max(0.0, interval_s))

        return False

    def _wait_skip_chain_done(
        self, run_stop: threading.Event, thresholds: Thresholds
    ) -> bool:
        chain = ("menu", "skip1", "skip2")
        step_idx = 0
        step_started = time.time()
        self.ctx.log("跳过链：等待 menu → skip1 → skip2")

        while self._is_active(run_stop):
            name = chain[step_idx]
            screen = self.ctx.screenshot()
            match, th, _ = self._find_once(screen, name, thresholds)
            if match is not None and match.score >= th and self._is_active(run_stop):
                self.operator.click_box(match, after_sleep=0.2)
                step_idx += 1
                if step_idx >= len(chain):
                    self.ctx.log("跳过链完成：menu → skip1 → skip2")
                    return True
                step_started = time.time()
                continue

            if step_idx > 0 and (time.time() - step_started) >= 5.0:
                self.ctx.log(
                    f"跳过链未完成：等待 {chain[step_idx]} 超时，重新识别 menu → skip1 → skip2"
                )
                step_idx = 0
                step_started = time.time()

            run_stop.wait(0.5)

        return False

    def _run_dialog_cycle(
        self,
        run_stop: threading.Event,
        thresholds: Thresholds,
        *,
        final_cycle: bool = False,
    ) -> bool:
        clicked_read = self._wait_and_click(
            run_stop,
            "read",
            thresholds,
            timeout_s=5.0,
            interval_s=0.5,
            after_sleep=0.5,
        )
        if not clicked_read:
            if self._is_active(run_stop):
                if final_cycle:
                    self.ctx.log("最后一轮 read 未识别到：任务触发已停止")
                else:
                    self.ctx.log("read 未识别到：重新进入 lock 识别")
            return False

        clicked_no_voice = self._wait_and_click(
            run_stop,
            "no_voice",
            thresholds,
            timeout_s=2.0,
            interval_s=0.4,
            after_sleep=0.1,
        )
        if (not clicked_no_voice) and self._is_active(run_stop):
            self.ctx.log("无语音未识别到：继续等待 menu → skip1 → skip2")

        chain_done = self._wait_skip_chain_done(run_stop, thresholds)
        if not chain_done:
            return False

        # 部分剧情存在确认按钮：这里必须刷新截图。
        # 否则会在“等待前”的旧画面上找 enter，导致永远找不到。
        enter_th = thresholds.get("enter", 0.8)
        max_enter_clicks = 8
        for _ in range(max_enter_clicks):
            if not self._is_active(run_stop):
                break

            screen = self.ctx.screenshot()
            enter_box, _, _ = self._find_once(screen, "enter", thresholds)
            if enter_box is None:
                break
            if enter_box.score < enter_th:
                break

            if not self._is_active(run_stop):
                break
            self.operator.click_box(enter_box, after_sleep=0.1)
            # 给 UI 一点时间刷新（同时可快速退出）
            run_stop.wait(0.35)

        return True

    def _finish_run(self, run_stop: threading.Event) -> None:
        with self._state_lock:
            if self._run_stop is run_stop:
                self._running.clear()
                self.ctx.end_task_exclusive()

    def _run(self, run_stop: threading.Event) -> None:
        self.ctx.log("任务触发：已启动")
        while self._is_active(run_stop):
            try:
                thresholds = self.ctx.thresholds_getter()
                screen = self.ctx.screenshot()
                lock_box, lock_th, _ = self._find_once(screen, "lock", thresholds)
                if lock_box is None or lock_box.score < lock_th:
                    self.ctx.log("lock 未检测到：执行最后一轮读剧情/对话")
                    self._run_dialog_cycle(
                        run_stop, thresholds, final_cycle=True
                    )
                    if self._is_active(run_stop):
                        self.ctx.log("任务触发：已停止（最后一轮结束）")
                    self._finish_run(run_stop)
                    break

                if self._run_dialog_cycle(run_stop, thresholds):
                    # 本轮完整完成：回到下一轮 lock 识别
                    continue

                if self._is_active(run_stop):
                    self.ctx.log("本轮未完成：重新进入 lock 识别")
            except Exception as e:
                self.ctx.log(f"任务触发异常：{e}")
                run_stop.wait(0.5)

        self._finish_run(run_stop)
