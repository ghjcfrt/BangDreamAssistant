from __future__ import annotations

import time
from typing import Optional

from vision import Match


class Operator:
    """把“截图→识别→点击”封装成可复用的操作接口。

    目标用法：
        start_box = self.operator.wait_img("read", timeout=30, interval=0.5)
        if start_box:
            self.operator.click_box(start_box, after_sleep=1.5)

    注：这里的 box 实际是 vision.Match（包含 top_left/size/center/score）。
    """

    def __init__(self, ctx) -> None:
        self.ctx = ctx

    def log_detection_result(
        self,
        name: str,
        match: Optional[Match],
        threshold: float,
        mode: Optional[str] = None,
    ) -> None:
        th = float(threshold)
        detect_mode = self._detection_mode_for_log(name, mode)
        if detect_mode == "ocr":
            label = self._ocr_label(name)
            if match is None:
                self.ctx.log(f"OCR 未识别到{label} th={th:.3f}")
                return

            x, y = match.center
            if match.score >= th:
                self.ctx.log(
                    f"OCR 识别到{label} score={match.score:.3f} at=({x},{y})"
                )
            else:
                self.ctx.log(
                    f"OCR 识别到{label}但未达阈值 score={match.score:.3f} th={th:.3f} at=({x},{y})"
                )
            return

        if match is None:
            self.ctx.log(f"检测 {name}：best=N/A th={th:.3f}")
            return

        x, y = match.center
        if match.score >= th:
            self.ctx.log(f"检测到 {name} score={match.score:.3f} at=({x},{y})")
        else:
            self.ctx.log(
                f"检测 {name}：未达阈值 best={match.score:.3f} th={th:.3f} at=({x},{y})"
            )

    def _detection_mode_for_log(self, name: str, mode: Optional[str]) -> str:
        raw = (mode or "").strip().lower()
        if raw in {"template", "ocr"}:
            return raw

        is_ocr_only = getattr(self.ctx, "is_ocr_only", None)
        if callable(is_ocr_only):
            try:
                if is_ocr_only(name):
                    return "ocr"
            except Exception:
                pass

        recognition_mode = getattr(self.ctx, "recognition_mode", None)
        if callable(recognition_mode):
            try:
                raw = str(recognition_mode()).strip().lower()
                if raw == "ocr":
                    matcher = getattr(self.ctx, "matcher", None)
                    patterns = getattr(matcher, "_ocr_patterns", {})
                    return "ocr" if name in patterns else "template"
                if raw == "template":
                    return "template"
            except Exception:
                pass

        return "template"

    @staticmethod
    def _ocr_label(name: str) -> str:
        return {
            "menu": "菜单",
            "skip1": "跳过1",
            "skip2": "跳过2",
            "read": "阅读",
            "no_voice": "无语音",
        }.get(name, name)

    def wait_img(
        self,
        name: str,
        timeout: float = 30.0,
        interval: float = 0.5,
        threshold: Optional[float] = None,
        log_on_timeout: bool = True,
    ) -> Optional[Match]:
        """在超时时间内轮询截图查找模板。

        约定：即使 timeout<=0 也会至少尝试识别一次（方便做“单次探测”）。
        """

        th = (
            self.ctx.thresholds_getter().get(name, 0.90)
            if threshold is None
            else float(threshold)
        )
        deadline = time.time() + float(timeout)

        best: Optional[Match] = None

        while True:
            screen = self.ctx.screenshot()
            m = self.ctx.find(screen, name)
            self.log_detection_result(name, m, th)
            if m is not None:
                if best is None or m.score > best.score:
                    best = m
                if m.score >= th:
                    return m

            if time.time() >= deadline:
                break
            time.sleep(float(interval))

        if log_on_timeout:
            if best is None:
                self.ctx.log(
                    f"wait_img 失败：{name} best=N/A th={th:.3f} timeout={timeout}s"
                )
            else:
                x, y = best.center
                self.ctx.log(
                    f"wait_img 失败：{name} best={best.score:.3f} th={th:.3f} at=({x},{y}) timeout={timeout}s"
                )
        return None

    def exists_img(self, name: str, threshold: Optional[float] = None) -> bool:
        th = (
            self.ctx.thresholds_getter().get(name, 0.90)
            if threshold is None
            else float(threshold)
        )
        screen = self.ctx.screenshot()
        m = self.ctx.find(screen, name)
        self.log_detection_result(name, m, th)
        return m is not None and m.score >= th

    def click_box(self, box: Match, after_sleep: float = 0.0) -> None:
        x, y = box.center
        self.ctx.tap(x, y)
        self.ctx.log(f"点击 {box.name} score={box.score:.3f} at=({x},{y})")
        if after_sleep and after_sleep > 0:
            time.sleep(float(after_sleep))
