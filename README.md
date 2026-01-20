# BBAssistant

基于 ADB + 模板匹配的简单自动点击器（Qt / PySide6 GUI）。

## 运行

1. 确保电脑可用 `adb`（Android SDK Platform-Tools），并能连接到 MuMu（常见：`127.0.0.1:16384`）。
2. 安装依赖（推荐在虚拟环境里）：

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
```

3. 启动 GUI：

```powershell
.\.venv\Scripts\python src\app.py
```

## Python 版本建议（Windows）

- 推荐：Python 3.12 或 3.13.1+。
- 不建议：Python 3.13.0（部分环境会在退出时出现 `Exception ignored on threading shutdown` 的 SystemError）。

本项目已在关闭窗口时主动停止并 join 后台线程，正常退出时不应再触发该报错；若你仍偶发看到，优先升级 Python 到更新的 3.13.x 或退回 3.12。 

## 性能/加速（降低 Python CPU 99% 的关键）

实时触发卡顿、CPU 飙高，主要来自：
1) `adb exec-out screencap -p` 的 PNG 数据量大
2) Python 端 PNG 解码 + OCR 推理非常吃 CPU

本项目已优先尝试使用 **raw screencap**（不走 PNG），能显著降低解码开销；若你的设备/模拟器不支持，会自动回退到 PNG，并在日志提示。

### OCR 推理加速（可选）

RapidOCR（onnxruntime）支持通过环境变量选择加速：

- CPU（默认）：不需要额外配置
- NVIDIA CUDA：
	- 安装：`pip install -r requirements.txt -r requirements-gpu.txt`
	- 启用：设置环境变量 `BB_OCR_ACCEL=cuda`
- DirectML（Windows 上 AMD/Intel/NVIDIA 都可用，推荐）：
	- 安装：`pip install -r requirements.txt -r requirements-gpu.txt`
	- 启用：设置环境变量 `BB_OCR_ACCEL=dml`

示例（PowerShell）：

```powershell
$env:BB_OCR_ACCEL = 'cuda'   # 或 'dml'
.\.venv\Scripts\python src\app.py
```

可用 provider 快速自检：

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## 模板图

模板图位于 `img/`：
- `menu.png`, `skip1.png`, `skip2.png`
- `read.png`, `no_voice.png`, `lock.png`

可在界面里调阈值（0~1）：越小越容易触发。
