from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from vision import TemplateMatcher


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="调试指定 name 的 ROI + OCR 识别结果")
    ap.add_argument("image", type=str, help="整屏截图路径（png/jpg）")
    ap.add_argument("--name", type=str, default="skip2", help="识别目标（默认 skip2）")
    ap.add_argument(
        "--roi-image",
        action="store_true",
        help="将输入图片视为已裁剪好的 ROI（不再按 name 做 ROI 裁剪）",
    )
    args = ap.parse_args()

    img_path = Path(args.image)
    screen_bgr = _read_image(img_path)

    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir / "img"

    matcher = TemplateMatcher(img_dir)
    engine = matcher._ensure_ocr()
    if engine is None:
        raise SystemExit("RapidOCR 未安装或初始化失败：请检查 requirements.txt 与安装环境")

    sh, sw = screen_bgr.shape[:2]
    # 小图（例如模板/按钮小截图）很可能不是“整屏截图”，自动切换到 ROI 模式避免裁剪跑偏。
    auto_roi_image = (sw <= 600 and sh <= 600)
    roi_mode = bool(args.roi_image or auto_roi_image)

    print(f"screen: {sw}x{sh}")

    if roi_mode:
        x0p, y0p, x1p, y1p = 0, 0, sw, sh
        print("roi:    (input-as-roi)")
        print(f"roi+pad:({x0p},{y0p})-({x1p},{y1p})")
        roi = screen_bgr
    else:
        x0, y0, x1, y1 = matcher._roi_for(args.name, sw, sh)
        pad_x = max(4, int(round((x1 - x0) * 0.04)))
        pad_y = max(4, int(round((y1 - y0) * 0.04)))
        x0p = max(0, x0 - pad_x)
        y0p = max(0, y0 - pad_y)
        x1p = min(sw, x1 + pad_x)
        y1p = min(sh, y1 + pad_y)
        print(f"roi:    ({x0},{y0})-({x1},{y1})")
        print(f"roi+pad:({x0p},{y0p})-({x1p},{y1p})")
        roi = screen_bgr[y0p:y1p, x0p:x1p]
    if roi.size == 0:
        raise SystemExit("ROI 为空：请确认分辨率与 ROI 配置")

    pattern = matcher._ocr_patterns.get(args.name)
    print(f"pattern: {pattern.pattern if pattern else 'N/A'}")

    variants, scale = matcher._ocr_build_variants(roi, aggressive=(args.name in {"skip1", "skip2"}))
    print(f"ocr input scale: {scale}")
    for i, v in enumerate(variants):
        print("\n" + "=" * 60)
        print(f"variant[{i}] shape={v.shape[1]}x{v.shape[0]}")

        items = matcher._ocr_extract(engine, v)
        if not items:
            print("(no ocr results)")
            continue

        # 打印 top 10
        items_sorted = sorted(items, key=lambda x: float(x[2]) if len(x) >= 3 else 0.0, reverse=True)
        for box, text, conf in items_sorted[:10]:
            ok = bool(pattern and pattern.search(text or ""))
            print(f"  conf={float(conf):.3f} match={ok} text={text!r}")


if __name__ == "__main__":
    main()
