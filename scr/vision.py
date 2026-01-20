from __future__ import annotations

import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from scr.img import IMG

try:
    # rapidocr-onnxruntime
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:  # pragma: no cover
    RapidOCR = None


@dataclass(frozen=True)
class Match:
    name: str
    score: float
    top_left: Tuple[int, int]
    size: Tuple[int, int]

    @property
    def center(self) -> Tuple[int, int]:
        x, y = self.top_left
        w, h = self.size
        return (x + w // 2, y + h // 2)


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _prep_gray(gray: np.ndarray) -> np.ndarray:
    # 保留为通用轻量预处理；真正用于匹配的预处理在 _prep_gray_match 中完成
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _prep_gray_match(gray: np.ndarray) -> np.ndarray:
    """用于模板匹配的灰度预处理。

    参考 SRACore 这类项目的实践：优先保证“置信度数值可用、可对齐阈值”。
    - 不做过强模糊（会把峰值压平）
    - 做直方图均衡，减少亮度/对比度变化影响
    """

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    # equalizeHist 需要单通道 8-bit
    return cv2.equalizeHist(gray)


def _prep_edge(gray: np.ndarray) -> np.ndarray:
    gray = _prep_gray_match(gray)
    edge = cv2.Canny(gray, 50, 150)
    # UI 边缘细且可能有轻微缩放/锯齿，适度膨胀更稳
    return cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)


def _prep_edge_soft(gray: np.ndarray) -> np.ndarray:
    """更敏感的边缘提取（适合半透明/低对比按钮）。"""

    gray = _prep_gray_match(gray)
    edge = cv2.Canny(gray, 20, 80)
    return cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)


class TemplateMatcher:
    def __init__(self, img_dir: Path) -> None:
        self.img_dir = img_dir
        self._templates_gray: Dict[str, np.ndarray] = {}
        self._templates_edge: Dict[str, np.ndarray] = {}
        self._templates_edge_soft: Dict[str, np.ndarray] = {}
        self._sizes: Dict[str, Tuple[int, int]] = {}

        # OCR
        # - 只对“易受分辨率/缩放影响的文字按钮”默认启用 OCR。
        # - ROI 坐标不变（沿用 _roi_rules）。
        self._ocr_enabled_for = {"menu", "skip1", "skip2", "read", "no_voice"}
        self._ocr_engine = None
        self._ocr_patterns: Dict[str, re.Pattern[str]] = {
            # 常见按钮文本：菜单 / 跳过（中英混合也支持）
            "menu": re.compile(r"(菜\s*单|menu)", re.IGNORECASE),
            # 部分情况下只识别到一个字（例如只出“跳”），在 ROI 足够精准时也可认为命中
            "skip1": re.compile(r"(跳\s*过|跳|skip)", re.IGNORECASE),
            "skip2": re.compile(r"(跳\s*过|跳|skip)", re.IGNORECASE),
            "no_voice": re.compile(r"(无\s*语\s*音|no\s*voice)", re.IGNORECASE),
            "read": re.compile(r"(阅\s*读|read)", re.IGNORECASE),
        }

        # (name, w, h) -> (tpl_gray, tpl_edge, tpl_edge_soft)
        self._scaled_cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # ROI 规则：既支持相对(0~1)也支持绝对像素(>=1)。
        # 你提供的 ROI（1920*1080，像素坐标）：
        # - menu:  (1634, 16)  - (1907, 171)
        # - skip1: (775, 34)   - (960, 146)
        # - skip2: (970, 614)  - (1345, 734)
        # - read: (1417, 910)  - (1796, 1033)
        # - no_voice: (808, 814) - (1105, 941)
        # - lock: (326, 134)   - (414, 245)
        # - enter: (763, 722)  - (1154, 880)
        self._roi_rules: Dict[str, Tuple[float, float, float, float]] = {
            "menu": (1634, 16, 1907, 171),
            "skip1": (775, 34, 960, 146),
            "skip2": (970, 614, 1345, 734),
            "read": (1417, 910, 1796, 1033),
            "no_voice": (808, 814, 1105, 941),
            "lock": (326, 134, 414, 245),
            "enter": (763, 722, 1154, 880),
        }

        # 上述 ROI 像素坐标基于 1920x1080；若实际截图分辨率不同，则按比例缩放。
        self._roi_base_size: Tuple[int, int] = (1920, 1080)

        # 经验尺度：不同模拟器/DPI/分辨率下模板可能有轻微缩放
        self._multi_scales = {
            # "skip1": (0.90, 0.95, 1.00, 1.05, 1.10),
        }

        # debug: 保存 lock 识别过程中的 ROI / 预处理结果 / 命中 patch。
        # 开启方式（二选一）：
        # 1) 创建标记文件：./debug/lock/enable （无需设置环境变量）
        # 2) 环境变量：BB_DEBUG_LOCK=1
        # 输出目录：
        # - 默认 ./debug/lock
        # - 或通过 BB_DEBUG_DIR 指定父目录（不含 /lock）
        debug_dir = (os.environ.get("BB_DEBUG_DIR") or "").strip()
        base_debug_dir = Path(debug_dir) if debug_dir else (Path.cwd() / "debug")
        self._debug_lock_dir = base_debug_dir / "lock"
        env_on = (os.environ.get("BB_DEBUG_LOCK") or "").strip().lower() in {"1", "true", "yes", "on"}
        file_on = (self._debug_lock_dir / "enable").exists()
        self._debug_lock = bool(env_on or file_on)

    @staticmethod
    def _imwrite(path: Path, img: np.ndarray) -> None:
        """兼容 Windows 路径（含中文/空格）的写图。"""

        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower() or ".png"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise RuntimeError(f"imencode failed: {path}")
        buf.tofile(str(path))

    def _debug_dump_lock(
        self,
        *,
        sw: int,
        sh: int,
        roi_rect: Tuple[int, int, int, int],
        roi_bgr: np.ndarray,
        roi_gray_raw: np.ndarray,
        roi_gray: np.ndarray,
        roi_edge: np.ndarray,
        roi_edge_soft: np.ndarray,
        best_method: str,
        best_score: float,
        edge_score: float,
        gray_ccorr_score: float,
        gray_sqdiff_score: float,
        edge_upscale: float,
        patch_pad: int,
        best_loc: Tuple[int, int],
        best_size: Tuple[int, int],
        tpl_gray: np.ndarray,
        tpl_edge: np.ndarray,
        tpl_edge_soft: np.ndarray,
        res_map: Optional[np.ndarray],
    ) -> None:
        if not self._debug_lock:
            return

        x0, y0, x1, y1 = roi_rect
        w, h = best_size
        bx, by = best_loc

        stamp = time.strftime("%Y%m%d_%H%M%S")
        uniq = str(time.time_ns())[-6:]
        out_dir = self._debug_lock_dir / f"{stamp}_{uniq}_sw{sw}_sh{sh}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 原始 ROI 与预处理
        meta = (
            f"roi=({x0},{y0})-({x1},{y1}) "
            f"method={best_method} score={best_score:.4f} "
            f"edge={edge_score:.4f} gray_ccorr={gray_ccorr_score:.4f} gray_sqdiff={gray_sqdiff_score:.4f} "
            f"edge_upscale={edge_upscale:.2f} "
            f"patch_pad={int(patch_pad)} "
            f"loc=({bx},{by}) size=({w},{h})"
        )
        (out_dir / "meta.txt").write_text(meta, encoding="utf-8")

        self._imwrite(out_dir / "roi_bgr.png", roi_bgr)
        self._imwrite(out_dir / "roi_gray_raw.png", roi_gray_raw)
        self._imwrite(out_dir / "roi_gray_eq.png", roi_gray)
        self._imwrite(out_dir / "roi_edge.png", roi_edge)
        self._imwrite(out_dir / "roi_edge_soft.png", roi_edge_soft)

        # 模板（当前 scale）
        self._imwrite(out_dir / "tpl_gray.png", tpl_gray)
        self._imwrite(out_dir / "tpl_edge.png", tpl_edge)
        self._imwrite(out_dir / "tpl_edge_soft.png", tpl_edge_soft)

        # best patch
        # - semantic patch: 与 best_loc/best_size 严格一致
        # - padded patch: 为了覆盖 UI 图标抗锯齿/留白（更利于人工排查）
        roi_h, roi_w = roi_bgr.shape[:2]
        sx0, sy0, sx1, sy1 = bx, by, bx + w, by + h
        sx0 = int(max(0, min(roi_w, sx0)))
        sy0 = int(max(0, min(roi_h, sy0)))
        sx1 = int(max(0, min(roi_w, sx1)))
        sy1 = int(max(0, min(roi_h, sy1)))
        patch_sem = roi_bgr[sy0:sy1, sx0:sx1]
        if patch_sem.size:
            self._imwrite(out_dir / "best_patch_semantic_bgr.png", patch_sem)

        pad = int(max(0, patch_pad))
        px0 = int(max(0, sx0 - pad))
        py0 = int(max(0, sy0 - pad))
        px1 = int(min(roi_w, sx1 + pad))
        py1 = int(min(roi_h, sy1 + pad))
        patch_bgr = roi_bgr[py0:py1, px0:px1]
        if patch_bgr.size:
            self._imwrite(out_dir / "best_patch_bgr.png", patch_bgr)
            patch_gray_raw = _to_gray(patch_bgr)
            patch_gray = _prep_gray_match(patch_gray_raw)
            patch_edge = _prep_edge(patch_gray_raw)
            patch_edge_soft = _prep_edge_soft(patch_gray_raw)
            self._imwrite(out_dir / "best_patch_gray_raw.png", patch_gray_raw)
            self._imwrite(out_dir / "best_patch_gray_eq.png", patch_gray)
            self._imwrite(out_dir / "best_patch_edge.png", patch_edge)
            self._imwrite(out_dir / "best_patch_edge_soft.png", patch_edge_soft)

        # 响应图（可选）
        if res_map is not None and isinstance(res_map, np.ndarray) and res_map.size:
            r = res_map.astype(np.float32)
            r = r - float(np.min(r))
            mx = float(np.max(r))
            if mx > 1e-8:
                r = r / mx
            r8 = np.clip(r * 255.0, 0, 255).astype(np.uint8)
            heat = cv2.applyColorMap(r8, cv2.COLORMAP_JET)
            self._imwrite(out_dir / "response_heat.png", heat)

    def load(self, name: str, filename: str) -> None:
        rel = Path(filename)
        # 兼容两种写法：
        # - "enter.png"：相对于 img_dir
        # - "img/enter.png"：相对于项目根目录（img_dir 的父目录）
        if rel.is_absolute():
            path = rel
        elif rel.parts and rel.parts[0] == self.img_dir.name:
            path = self.img_dir.parent / rel
        else:
            path = self.img_dir / rel
        tpl = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if tpl is None:
            raise FileNotFoundError(f"无法读取模板图: {path}")
        gray_raw = _to_gray(tpl)
        gray = _prep_gray_match(gray_raw)
        edge = _prep_edge(gray_raw)
        edge_soft = _prep_edge_soft(gray_raw)
        h, w = gray.shape[:2]
        self._templates_gray[name] = gray
        self._templates_edge[name] = edge
        self._templates_edge_soft[name] = edge_soft
        self._sizes[name] = (w, h)

        # 模板变更时清缓存
        self._scaled_cache = {}

    def enable_ocr_for(self, names: Sequence[str]) -> None:
        self._ocr_enabled_for.update(str(n) for n in names)

    def disable_ocr_for(self, names: Sequence[str]) -> None:
        for n in names:
            self._ocr_enabled_for.discard(str(n))

    def _ensure_ocr(self):
        if self._ocr_engine is not None:
            return self._ocr_engine
        if RapidOCR is None:
            return None
        # 默认参数：更偏“实时 UI 文本”场景，后续可根据截图效果再调。
        # 支持通过环境变量选择加速：
        # - BB_OCR_ACCEL=cpu|cuda|dml
        #   - cuda: 需要安装 onnxruntime-gpu + NVIDIA CUDA 环境
        #   - dml:  需要安装 onnxruntime-directml（AMD/Intel/NVIDIA 都可用）
        # 与 UI 默认一致：未显式配置时默认使用 DirectML（Windows 推荐）。
        accel = (os.environ.get("BB_OCR_ACCEL") or "dml").strip().lower()
        kwargs = {}
        if accel == "cuda":
            kwargs.update({
                "det_use_cuda": True,
                "cls_use_cuda": True,
                "rec_use_cuda": True,
            })
        elif accel == "dml":
            kwargs.update({
                "det_use_dml": True,
                "cls_use_dml": True,
                "rec_use_dml": True,
            })

        # 如果用户配置了加速但环境不满足，RapidOCR 可能会抛异常；这里自动回退到 CPU。
        try:
            self._ocr_engine = RapidOCR(**kwargs)
        except Exception:
            self._ocr_engine = RapidOCR()
        return self._ocr_engine

    @staticmethod
    def _ocr_extract(engine, img_bgr: np.ndarray):
        """尽量兼容不同 RapidOCR 调用签名，统一输出 list[(box4, text, score)].

        box4: [[x,y], [x,y], [x,y], [x,y]]
        """

        # RapidOCR 通常接受 BGR 或 RGB；这里用 BGR 直接尝试
        out = None
        try:
            out = engine(img_bgr)
        except TypeError:
            # 兼容 engine.ocr(img)
            if hasattr(engine, "ocr"):
                out = engine.ocr(img_bgr)
        except Exception:
            out = None

        # 常见返回：(result, elapse)
        if isinstance(out, tuple) and len(out) >= 1:
            result = out[0]
        else:
            result = out

        if not result:
            return []

        items = []
        for it in result:
            # it 可能是 [box, text, score] 或 (box, text, score)
            if not isinstance(it, (list, tuple)) or len(it) < 2:
                continue
            box = it[0]
            text = str(it[1])
            score = float(it[2]) if len(it) >= 3 else 0.0
            items.append((box, text, score))
        return items

    @staticmethod
    def _box_to_rect(box) -> Optional[Tuple[int, int, int, int]]:
        try:
            pts = box
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            x0 = int(max(0, min(xs)))
            y0 = int(max(0, min(ys)))
            x1 = int(max(xs))
            y1 = int(max(ys))
            if x1 <= x0 or y1 <= y0:
                return None
            return (x0, y0, x1, y1)
        except Exception:
            return None

    @staticmethod
    def _ocr_build_variants(
        roi_bgr: np.ndarray, *, aggressive: bool = False
    ) -> Tuple[Sequence[np.ndarray], float]:
        """为 UI 小按钮文本构造多个 OCR 输入变体。

        重点覆盖“粉色底 + 白色字”这类在纯灰度阈值下容易识别失败的情况。
        """

        if roi_bgr.size == 0:
            return ([], 1.0)

        h, w = roi_bgr.shape[:2]

        # RapidOCR 对小字更依赖清晰的边缘与足够分辨率：先放大
        m = max(h, w)
        if m < 160:
            scale = 4.0
        elif m < 420:
            scale = 2.0
        else:
            scale = 1.5
        up = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        variants: list[np.ndarray] = [up]

        # 灰度 + CLAHE：增强局部对比度
        gray = _to_gray(up)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g2 = clahe.apply(gray)
        variants.append(cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR))

        if aggressive:
            # 白字掩膜：低饱和 + 高亮度（对粉底白字很有效）
            hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 185], dtype=np.uint8)
            upper_white = np.array([180, 90, 255], dtype=np.uint8)
            white_mask = cv2.inRange(hsv, lower_white, upper_white)
            white_mask = cv2.medianBlur(white_mask, 3)
            white_mask = cv2.dilate(white_mask, np.ones((2, 2), np.uint8), iterations=1)
            variants.append(cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR))
            variants.append(cv2.cvtColor(255 - white_mask, cv2.COLOR_GRAY2BGR))

            # 自适应阈值 + 反相：作为通用回退
            gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            bin_img = cv2.adaptiveThreshold(
                gray_blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                6,
            )
            variants.append(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR))
            variants.append(cv2.cvtColor(255 - bin_img, cv2.COLOR_GRAY2BGR))

        return (variants, float(scale))

    def _find_by_ocr(self, screen_bgr: np.ndarray, name: str) -> Optional[Match]:
        engine = self._ensure_ocr()
        if engine is None:
            return None

        sh, sw = screen_bgr.shape[:2]
        x0, y0, x1, y1 = self._roi_for(name, sw, sh)

        # OCR 场景下给 ROI 一点 padding，容错不同分辨率/布局的轻微偏移
        pad_x = max(4, int(round((x1 - x0) * 0.04)))
        pad_y = max(4, int(round((y1 - y0) * 0.04)))
        x0p = max(0, x0 - pad_x)
        y0p = max(0, y0 - pad_y)
        x1p = min(sw, x1 + pad_x)
        y1p = min(sh, y1 + pad_y)
        roi = screen_bgr[y0p:y1p, x0p:x1p]
        if roi.size == 0:
            return None

        variants, scale = self._ocr_build_variants(roi, aggressive=(name in {"skip1", "skip2"}))
        if not scale or scale <= 0:
            scale = 1.0

        pattern = self._ocr_patterns.get(name)
        if pattern is None:
            return None

        best: Optional[Match] = None
        for ocr_img in variants:
            for box, text, conf in self._ocr_extract(engine, ocr_img):
                if not text:
                    continue
                if not pattern.search(text):
                    continue

                rect = self._box_to_rect(box)
                if rect is None:
                    continue
                rx0, ry0, rx1, ry1 = rect

                # 注意：OCR 输入对 ROI 做了放大（scale>1）。
                # OCR 返回的 box 坐标在“放大后的图”坐标系里，必须回映射到原 ROI。
                if scale != 1.0:
                    rx0 = int(round(rx0 / scale))
                    ry0 = int(round(ry0 / scale))
                    rx1 = int(round(rx1 / scale))
                    ry1 = int(round(ry1 / scale))

                w = int(rx1 - rx0)
                h = int(ry1 - ry0)
                m = Match(
                    name=name,
                    score=float(max(0.0, min(1.0, conf))),
                    top_left=(int(rx0 + x0p), int(ry0 + y0p)),
                    size=(w, h),
                )
                if best is None or m.score > best.score:
                    best = m

        # 点击策略：统一点 ROI 的几何中心。
        # 这样只要 ROI 配置准确，就不会受 OCR 文本框偏移/裁剪影响。
        if best is not None:
            cx = int(max(0, min(sw - 1, (x0 + x1) // 2)))
            cy = int(max(0, min(sh - 1, (y0 + y1) // 2)))
            return Match(
                name=name,
                score=float(best.score),
                top_left=(cx, cy),
                size=(1, 1),
            )

        return None

    def _roi_for(self, name: str, sw: int, sh: int) -> Tuple[int, int, int, int]:
        rule = getattr(self, "_roi_rules", {}).get(name)
        if not rule:
            return (0, 0, sw, sh)

        x0r, y0r, x1r, y1r = rule

        # 如果四个值都在 0~1，按相对比例解释；否则按绝对像素解释。
        if 0.0 <= float(x0r) <= 1.0 and 0.0 <= float(y0r) <= 1.0 and 0.0 <= float(x1r) <= 1.0 and 0.0 <= float(y1r) <= 1.0:
            x0 = int(max(0, min(sw, round(sw * float(x0r)))))
            y0 = int(max(0, min(sh, round(sh * float(y0r)))))
            x1 = int(max(0, min(sw, round(sw * float(x1r)))))
            y1 = int(max(0, min(sh, round(sh * float(y1r)))))
        else:
            # 绝对像素 ROI 默认按 1920x1080 基准配置；不同分辨率下按比例映射。
            bw, bh = self._roi_base_size
            sx = (float(sw) / float(bw)) if bw > 0 else 1.0
            sy = (float(sh) / float(bh)) if bh > 0 else 1.0
            x0 = int(max(0, min(sw, round(float(x0r) * sx))))
            y0 = int(max(0, min(sh, round(float(y0r) * sy))))
            x1 = int(max(0, min(sw, round(float(x1r) * sx))))
            y1 = int(max(0, min(sh, round(float(y1r) * sy))))

        # 保底，避免异常配置导致空 ROI
        if x1 <= x0:
            x0, x1 = 0, sw
        if y1 <= y0:
            y0, y1 = 0, sh
        return (x0, y0, x1, y1)

    def _get_scaled_tpl(self, name: str, w: int, h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (name, int(w), int(h))
        hit = self._scaled_cache.get(key)
        if hit is not None:
            return hit

        base_gray = self._templates_gray[name]
        base_edge = self._templates_edge[name]
        base_edge_soft = self._templates_edge_soft[name]

        if base_gray.shape[1] == w and base_gray.shape[0] == h:
            out = (base_gray, base_edge, base_edge_soft)
        else:
            out = (
                cv2.resize(base_gray, (w, h), interpolation=cv2.INTER_AREA),
                cv2.resize(base_edge, (w, h), interpolation=cv2.INTER_NEAREST),
                cv2.resize(base_edge_soft, (w, h), interpolation=cv2.INTER_NEAREST),
            )

        self._scaled_cache[key] = out
        return out

    def load_defaults(self) -> None:
        self.load("menu", IMG.MENU)
        self.load("skip1", IMG.SKIP1)
        self.load("skip2", IMG.SKIP2)
        self.load("read", IMG.READ)
        self.load("no_voice", IMG.NO_VOICE)
        self.load("lock", IMG.LOCK)
        self.load("enter", IMG.ENTER)

    def find_best(self, screen_bgr: np.ndarray, name: str, *, allow_ocr: bool = True) -> Optional[Match]:
        # 规则（强制路由）：
        # - menu/skip1/skip2/no_voice/read：只走 OCR（失败即视为未命中，不回退模板匹配）
        # - lock：只走图像识别（模板匹配），并优先灰度匹配（永不走 OCR）

        # lock：明确使用图像识别（模板匹配），并优先考虑灰度图。
        # 这里采用“多种匹配方法 + 形状校验（圆形）”，解决 lock 置信度偏低的问题。
        # 同时支持落盘调试：
        # - 创建 ./debug/lock/enable（推荐，无需环境变量）
        # - 或设置 BB_DEBUG_LOCK=1；可选 BB_DEBUG_DIR=... 指定输出父目录
        # 注意：即使外部通过 enable_ocr_for("lock") 误开启，也会被该分支屏蔽。
        if name == "lock":
            if name not in self._templates_gray:
                raise KeyError(f"模板未加载: {name}")

            sh, sw = screen_bgr.shape[:2]
            x0, y0, x1, y1 = self._roi_for(name, sw, sh)
            roi = screen_bgr[y0:y1, x0:x1]

            roi_gray_raw = _to_gray(roi)
            screen_gray = _prep_gray_match(roi_gray_raw)
            screen_edge = _prep_edge(roi_gray_raw)
            screen_edge_soft = _prep_edge_soft(roi_gray_raw)
            base_w, base_h = self._sizes[name]
            scales = self._multi_scales.get(name, (1.0,))

            best_score: Optional[float] = None
            best_loc: Optional[Tuple[int, int]] = None
            best_size: Optional[Tuple[int, int]] = None
            best_method: str = ""
            best_tpl: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
            best_res_map: Optional[np.ndarray] = None

            # 两阶段策略：
            # - 阶段 1：只用 edge 做定位（决定 best_loc/best_size）
            # - 阶段 2：在该位置计算灰度 ccorr/sqdiff 仅用于“信心评估”，不参与找位置
            edge_score: float = 0.0
            gray_ccorr_score: float = 0.0
            gray_sqdiff_score: float = 0.0

            # edge 定位可选上采样：同时放大 screen_edge 与 tpl_edge，提升定位稳定性。
            # 该缩放不会影响最终裁剪：best_loc 会映射回原 ROI 坐标系。
            edge_upscale = float(os.environ.get("BB_LOCK_EDGE_UPSCALE") or 2.0)
            if not (edge_upscale and edge_upscale > 0):
                edge_upscale = 1.0

            # 说明：lock 图标是“白底 + 灰色圆 + 圆内白色锁”。
            # 定位只用边缘相关（TM_CCORR_NORMED），灰度只参与信心评估。
            # 永远不用 TM_CCOEFF_NORMED。
            best_loc_u: Optional[Tuple[int, int]] = None
            best_wh: Optional[Tuple[int, int]] = None
            for s in scales:
                w = int(round(base_w * float(s)))
                h = int(round(base_h * float(s)))
                if w < 6 or h < 6:
                    continue
                if h > screen_gray.shape[0] or w > screen_gray.shape[1]:
                    continue

                tpl_gray, tpl_edge, tpl_edge_soft = self._get_scaled_tpl(name, w, h)

                # 阶段 1：只用边缘定位
                if edge_upscale != 1.0:
                    screen_edge_u = cv2.resize(
                        screen_edge,
                        None,
                        fx=edge_upscale,
                        fy=edge_upscale,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    tpl_edge_u = cv2.resize(
                        tpl_edge,
                        None,
                        fx=edge_upscale,
                        fy=edge_upscale,
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    screen_edge_u = screen_edge
                    tpl_edge_u = tpl_edge

                if tpl_edge_u.shape[0] > screen_edge_u.shape[0] or tpl_edge_u.shape[1] > screen_edge_u.shape[1]:
                    continue

                res = cv2.matchTemplate(screen_edge_u, tpl_edge_u, cv2.TM_CCORR_NORMED)
                _, max_val, _, max_loc_u = cv2.minMaxLoc(res)
                score = float(max_val)
                if best_score is None or score > best_score:
                    best_score = float(score)
                    edge_score = float(score)
                    best_loc_u = (int(max_loc_u[0]), int(max_loc_u[1]))
                    best_wh = (int(w), int(h))
                    best_method = "edge_locate"
                    best_tpl = (tpl_gray, tpl_edge, tpl_edge_soft)
                    best_res_map = res

            if best_score is None or best_loc_u is None or best_wh is None:
                return None

            # 将 edge 上采样坐标系映射回原 ROI 坐标系
            if edge_upscale != 1.0:
                best_loc = (
                    int(round(float(best_loc_u[0]) / float(edge_upscale))),
                    int(round(float(best_loc_u[1]) / float(edge_upscale))),
                )
            else:
                best_loc = best_loc_u

            # 语义尺寸：永远来自模板“语义尺寸”（base_w/base_h * best_scale），不从 edge resize 后的 shape 回推。
            best_size = best_wh

            # 保底：避免缩放/四舍五入导致越界
            bw = int(max(1, best_size[0]))
            bh = int(max(1, best_size[1]))
            bx = int(max(0, min(screen_gray.shape[1] - bw, best_loc[0])))
            by = int(max(0, min(screen_gray.shape[0] - bh, best_loc[1])))
            best_loc = (bx, by)
            best_size = (bw, bh)

            # 用最终语义尺寸重新取一份模板，保证后续灰度打分/调试输出尺寸一致
            best_tpl = self._get_scaled_tpl(name, best_size[0], best_size[1])

            # 阶段 2：固定在 edge 定位的位置，只做灰度打分（不参与找位置）
            if best_tpl is not None:
                tpl_gray, tpl_edge, tpl_edge_soft = best_tpl
                w, h = best_size
                bx, by = best_loc
                patch_gray = screen_gray[by : by + h, bx : bx + w]
                if patch_gray.size and patch_gray.shape == tpl_gray.shape:
                    # 灰度：CCORR_NORMED
                    res = cv2.matchTemplate(patch_gray, tpl_gray, cv2.TM_CCORR_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    gray_ccorr_score = float(max_val)
                    # 灰度：SQDIFF_NORMED（越小越好）
                    res = cv2.matchTemplate(patch_gray, tpl_gray, cv2.TM_SQDIFF_NORMED)
                    min_val, _, _, _ = cv2.minMaxLoc(res)
                    gray_sqdiff_score = float(1.0 - float(min_val))

            # 形状校验：在 best patch 上尝试检测“圆形”（灰色圆），没有检测到则略降置信度。
            # 目的：防止边缘匹配在其他区域“虚高”。ROI 很小，开销可接受。
            w, h = best_size
            bx, by = best_loc
            # 用对称 padding 的 patch 做形状校验，更抗锯齿/留白导致的“截不全”
            pad = int(max(2, round(min(w, h) * 0.08)))
            roi_h, roi_w = roi_gray_raw.shape[:2]
            px0 = int(max(0, bx - pad))
            py0 = int(max(0, by - pad))
            px1 = int(min(roi_w, bx + w + pad))
            py1 = int(min(roi_h, by + h + pad))
            patch = roi_gray_raw[py0:py1, px0:px1]
            circle_ok = False
            try:
                if patch.size:
                    blur = cv2.GaussianBlur(patch, (3, 3), 0)
                    min_r = max(6, int(round(min(w, h) * 0.22)))
                    max_r = max(min_r + 2, int(round(min(w, h) * 0.55)))
                    circles = cv2.HoughCircles(
                        blur,
                        cv2.HOUGH_GRADIENT,
                        dp=1.2,
                        minDist=float(min(w, h) * 0.6),
                        param1=80,
                        param2=18,
                        minRadius=min_r,
                        maxRadius=max_r,
                    )
                    if circles is not None and len(circles) > 0:
                        circle_ok = True
            except Exception:
                circle_ok = False

            # 置信度融合：位置由 edge 决定；信心由 edge + 灰度评估共同决定。
            final_score = float(max(0.0, min(1.0, 0.70 * float(edge_score) + 0.15 * float(gray_ccorr_score) + 0.15 * float(gray_sqdiff_score))))
            if not circle_ok:
                final_score = float(max(0.0, min(1.0, final_score * 0.85)))

            # 调试落盘
            if best_tpl is not None:
                tpl_gray, tpl_edge, tpl_edge_soft = best_tpl
                self._debug_dump_lock(
                    sw=sw,
                    sh=sh,
                    roi_rect=(x0, y0, x1, y1),
                    roi_bgr=roi,
                    roi_gray_raw=roi_gray_raw,
                    roi_gray=screen_gray,
                    roi_edge=screen_edge,
                    roi_edge_soft=screen_edge_soft,
                    best_method=best_method,
                    best_score=final_score,
                    edge_score=float(edge_score),
                    gray_ccorr_score=float(gray_ccorr_score),
                    gray_sqdiff_score=float(gray_sqdiff_score),
                    edge_upscale=float(edge_upscale),
                    patch_pad=int(pad),
                    best_loc=best_loc,
                    best_size=best_size,
                    tpl_gray=tpl_gray,
                    tpl_edge=tpl_edge,
                    tpl_edge_soft=tpl_edge_soft,
                    res_map=best_res_map,
                )

            return Match(
                name=name,
                score=float(final_score),
                top_left=(int(best_loc[0] + x0), int(best_loc[1] + y0)),
                size=best_size,
            )

        # OCR-only：这些按钮只用 OCR 判断；不命中就返回 None。
        if name in {"menu", "skip1", "skip2", "no_voice", "read"}:
            if not allow_ocr:
                return None
            return self._find_by_ocr(screen_bgr, name)

        # 其他目标：OCR（可选）→ 模板匹配回退
        # 性能敏感场景（例如实时循环）可通过 allow_ocr=False 强制走模板匹配。
        if allow_ocr and name in self._ocr_enabled_for:
            m = self._find_by_ocr(screen_bgr, name)
            if m is not None:
                return m

        if name not in self._templates_gray:
            raise KeyError(f"模板未加载: {name}")

        sh, sw = screen_bgr.shape[:2]
        x0, y0, x1, y1 = self._roi_for(name, sw, sh)
        roi = screen_bgr[y0:y1, x0:x1]

        roi_gray_raw = _to_gray(roi)
        screen_gray = _prep_gray_match(roi_gray_raw)
        screen_edge = _prep_edge(roi_gray_raw)
        screen_edge_soft = _prep_edge_soft(roi_gray_raw)

        base_w, base_h = self._sizes[name]

        scales = self._multi_scales.get(name, (1.0,))

        best_score: Optional[float] = None
        best_loc: Optional[Tuple[int, int]] = None
        best_size: Optional[Tuple[int, int]] = None

        for s in scales:
            w = int(round(base_w * float(s)))
            h = int(round(base_h * float(s)))
            if w < 6 or h < 6:
                continue
            if h > screen_gray.shape[0] or w > screen_gray.shape[1]:
                continue

            tpl_gray, tpl_edge, tpl_edge_soft = self._get_scaled_tpl(name, w, h)

            # 1) 灰度
            # CCOEFF_NORMED 范围为 [-1, 1]，将其映射到 [0, 1] 作为“置信度”
            res_gray = cv2.matchTemplate(screen_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val_g_raw, _, max_loc_g = cv2.minMaxLoc(res_gray)
            max_val_g = (float(max_val_g_raw) + 1.0) * 0.5
            if best_score is None or max_val_g > best_score:
                best_score = float(max_val_g)
                best_loc = (int(max_loc_g[0]), int(max_loc_g[1]))
                best_size = (w, h)

            # 2) 边缘（正常）
            # 对二值/稀疏边缘，CCORR_NORMED 通常比分布更“高、更可用”
            res_edge = cv2.matchTemplate(screen_edge, tpl_edge, cv2.TM_CCORR_NORMED)
            _, max_val_e, _, max_loc_e = cv2.minMaxLoc(res_edge)
            if best_score is None or float(max_val_e) > best_score:
                best_score = float(max_val_e)
                best_loc = (int(max_loc_e[0]), int(max_loc_e[1]))
                best_size = (w, h)

            # 3) 边缘（软）
            res_edge2 = cv2.matchTemplate(screen_edge_soft, tpl_edge_soft, cv2.TM_CCORR_NORMED)
            _, max_val_e2, _, max_loc_e2 = cv2.minMaxLoc(res_edge2)
            if best_score is None or float(max_val_e2) > best_score:
                best_score = float(max_val_e2)
                best_loc = (int(max_loc_e2[0]), int(max_loc_e2[1]))
                best_size = (w, h)

        # 若 ROI/模板尺寸导致无法匹配（没有任何 scale 可用），返回 None
        if best_score is None or best_loc is None or best_size is None:
            return None

        # 坐标从 ROI 转回全屏
        return Match(
            name=name,
            score=float(best_score),
            top_left=(int(best_loc[0] + x0), int(best_loc[1] + y0)),
            size=best_size,
        )


def decode_png(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("截图解码失败")
    return img


def decode_screencap_raw(raw_bytes: bytes) -> np.ndarray:
    """Decode `adb exec-out screencap` raw output into BGR image.

    Raw output format (common AOSP implementation):
    - 3x uint32 little-endian header: width, height, format
    - followed by width*height*4 bytes in RGBA.

    Note: Some devices/emulators may differ; caller should catch and fall back to PNG.
    """

    if raw_bytes is None or len(raw_bytes) < 12:
        raise ValueError("raw screencap too small")

    w, h, fmt = struct.unpack_from("<III", raw_bytes, 0)
    if w <= 0 or h <= 0 or w > 20000 or h > 20000:
        raise ValueError(f"invalid raw screencap size: {w}x{h}")

    # Most common is RGBA_8888 (fmt==1), but we primarily validate by payload size.
    expected = 12 + int(w) * int(h) * 4
    if len(raw_bytes) < expected:
        raise ValueError(f"raw screencap length mismatch: got={len(raw_bytes)} expected>={expected} fmt={fmt}")

    buf = memoryview(raw_bytes)[12:expected]
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape((int(h), int(w), 4))
    # Convert to BGR for downstream OpenCV code.
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
