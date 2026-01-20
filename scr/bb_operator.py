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

        th = self.ctx.thresholds_getter().get(name, 0.90) if threshold is None else float(threshold)
        deadline = time.time() + float(timeout)

        best: Optional[Match] = None

        while True:
            screen = self.ctx.screenshot()
            m = self.ctx.find(screen, name)
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
                self.ctx.log(f"wait_img 失败：{name} best=N/A th={th:.3f} timeout={timeout}s")
            else:
                x, y = best.center
                self.ctx.log(
                    f"wait_img 失败：{name} best={best.score:.3f} th={th:.3f} at=({x},{y}) timeout={timeout}s"
                )
        return None

    def exists_img(self, name: str, threshold: Optional[float] = None) -> bool:
        th = self.ctx.thresholds_getter().get(name, 0.90) if threshold is None else float(threshold)
        screen = self.ctx.screenshot()
        m = self.ctx.find(screen, name)
        if m is None:
            self.ctx.log(f"检测 {name}：best=N/A th={th:.3f}")
            return False

        if m.score >= th:
            self.ctx.log(f"检测到 {name} score={m.score:.3f}")
            return True

        self.ctx.log(f"检测 {name}：未达阈值 best={m.score:.3f} th={th:.3f}")
        return False

    def click_box(self, box: Match, after_sleep: float = 0.0) -> None:
        x, y = box.center
        self.ctx.tap(x, y)
        self.ctx.log(f"点击 {box.name} score={box.score:.3f} at=({x},{y})")
        if after_sleep and after_sleep > 0:
            time.sleep(float(after_sleep))
