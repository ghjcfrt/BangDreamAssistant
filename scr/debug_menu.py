from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from adb_client import AdbClient
from vision import TemplateMatcher, decode_png, decode_screencap_raw


def _capture_screen(adb: AdbClient, serial: str) -> np.ndarray:
    try:
        raw = adb.screencap_raw(serial)
        return decode_screencap_raw(raw)
    except Exception:
        png = adb.screencap_png(serial)
        return decode_png(png)


def main() -> None:
    parser = argparse.ArgumentParser(description="BBAssistant - menu 识别单独测试")
    parser.add_argument("--adb", default="adb", help="adb 可执行文件路径")
    parser.add_argument("--serial", required=True, help="adb 设备序列号（adb devices 查看）")
    parser.add_argument(
        "--mode",
        choices=["template", "ocr", "both"],
        default="both",
        help="测试模式：仅模板/仅OCR/两者都测",
    )
    parser.add_argument("--save", action="store_true", help="保存截图到 screenshots/ 目录")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir / "assets/images"

    adb = AdbClient(adb_path=args.adb)
    matcher = TemplateMatcher(img_dir)
    matcher.load_defaults()

    screen = _capture_screen(adb, args.serial)
    h, w = screen.shape[:2]

    tpl = None
    ocr = None
    if args.mode in {"template", "both"}:
        tpl = matcher.find_template_only(screen, "menu")
    if args.mode in {"ocr", "both"}:
        ocr = matcher.find_ocr_only(screen, "menu")

    def fmt(m):
        if m is None:
            return "N/A"
        x, y = m.center
        return f"score={m.score:.3f} at=({x},{y})"

    print(f"screen: {w}x{h}")
    print(f"menu template: {fmt(tpl)}")
    print(f"menu ocr:      {fmt(ocr)}")

    if args.save:
        out_dir = base_dir / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"menu_test_{w}x{h}.png"
        ok, buf = cv2.imencode(".png", screen)
        if ok:
            buf.tofile(str(out_path))
            print(f"saved: {out_path}")
        else:
            print("save failed: imencode")


if __name__ == "__main__":
    main()
