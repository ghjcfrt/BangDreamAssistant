from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class AdbDevice:
    serial: str
    status: str


class AdbClient:
    def __init__(self, adb_path: str = "adb") -> None:
        self.adb_path = adb_path

    def _run(self, args: List[str], timeout_s: float = 15.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.adb_path, *args],
            capture_output=True,
            text=False,
            timeout=timeout_s,
        )

    def _run_text(self, args: List[str], timeout_s: float = 15.0) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.adb_path, *args],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    def list_devices(self) -> List[AdbDevice]:
        cp = self._run_text(["devices"], timeout_s=10.0)
        out = cp.stdout or ""
        devices: List[AdbDevice] = []
        for line in out.splitlines():
            line = line.strip()
            if not line or line.startswith("List of devices"):
                continue
            # serial\tstatus
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                devices.append(AdbDevice(serial=parts[0], status=parts[1]))
        return devices

    def connect(self, address: str) -> str:
        cp = self._run_text(["connect", address], timeout_s=10.0)
        return (cp.stdout or "").strip() or (cp.stderr or "").strip()

    def screencap_png(self, serial: str) -> bytes:
        cp = self._run(["-s", serial, "exec-out", "screencap", "-p"], timeout_s=10.0)
        if cp.returncode != 0:
            raise RuntimeError((cp.stderr or b"").decode("utf-8", errors="ignore") or "adb screencap failed")
        return cp.stdout

    def screencap_raw(self, serial: str) -> bytes:
        """Capture raw framebuffer bytes.

        Compared to PNG (`screencap -p`), raw output avoids PNG decode CPU cost.
        Not all devices/emulators support/behave consistently; caller should fall back to PNG.
        """

        cp = self._run(["-s", serial, "exec-out", "screencap"], timeout_s=10.0)
        if cp.returncode != 0:
            raise RuntimeError((cp.stderr or b"").decode("utf-8", errors="ignore") or "adb screencap(raw) failed")
        return cp.stdout

    def tap(self, serial: str, x: int, y: int) -> None:
        cp = self._run_text(["-s", serial, "shell", "input", "tap", str(x), str(y)], timeout_s=5.0)
        if cp.returncode != 0:
            raise RuntimeError((cp.stderr or "").strip() or "adb tap failed")

    def get_device_model(self, serial: str) -> Optional[str]:
        cp = self._run_text(["-s", serial, "shell", "getprop", "ro.product.model"], timeout_s=5.0)
        if cp.returncode != 0:
            return None
        model = (cp.stdout or "").strip()
        return model or None
