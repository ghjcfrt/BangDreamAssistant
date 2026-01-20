"""Compatibility shim for the Python standard library module `operator`.

This workspace previously introduced a local `operator.py`, which can shadow
the stdlib `operator` module when `src/` is on `sys.path`.

To avoid breaking dependencies, we load and re-export the real stdlib module.
Project-specific automation helpers live in `bb_operator.py`.
"""

from __future__ import annotations

import importlib.util
import os
import sysconfig


def _load_stdlib_operator() -> object:
    # 这里不能用 pathlib：pathlib 在导入过程中会触发 collections/glob 等模块，
    # 而 collections 在初始化时会 import operator，若此处再 import pathlib 就会形成循环导入。
    stdlib_dir = sysconfig.get_path("stdlib")
    if not stdlib_dir:
        raise ImportError("Cannot locate stdlib directory")
    op_path = os.path.join(stdlib_dir, "operator.py")
    spec = importlib.util.spec_from_file_location("_stdlib_operator", op_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load stdlib operator module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stdlib_operator = _load_stdlib_operator()

for _name in dir(_stdlib_operator):
    globals()[_name] = getattr(_stdlib_operator, _name)
