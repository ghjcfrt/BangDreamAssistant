# Bang Dream Assistant

## 简介

基于 ADB + 模板匹配的Bang Dream助手。

目前仅支持自动剧情

其余功能请考虑：

- [自动演奏](https://github.com/kvarenzn/ssm "ssm")
- [歌曲谱面下载](https://github.com/ghjcfrt/BestdoriDownload)

内置两套识别方式：

- 模板匹配（默认，开销更低）
- OCR（对分辨率/缩放更鲁棒，但更吃算力）

分辨率最好是`1920*1080`，如果是其他分辨率最好大于该分辨率

因为不是经常玩邦邦，所以代码处于原始的混乱态，什么时候想起来了再说优化的事

## 运行

### 1) 准备 ADB

1. 安装/配置 `adb`（Android SDK Platform-Tools），确保命令行可直接运行 `adb`。
2. 连接模拟器/真机，并拿到TCP/IP或emulator（例：常见为 `127.0.0.1:xxxxx`或 `emulator-xxxx`）：

```powershell
adb devices
```

### 2) 安装依赖（推荐虚拟环境）

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

可选：安装 OCR 加速依赖（Windows 推荐 DirectML）：

```powershell
.\.venv\Scripts\python -m pip install -r requirements-gpu.txt
```

### 3) 启动 GUI

```powershell
.\.venv\Scripts\python scr\app.py
```

说明：请在仓库根目录运行（与 README.md 同级），否则可能会出现 `ModuleNotFoundError`。

## Python 版本建议（Windows）

- 推荐：Python 3.12 或 3.13.1+。

## 性能/加速（CPU 99% 的关键问题）

实时触发卡顿、CPU 飙高，主要来自：

1) `adb exec-out screencap -p` 的 PNG 数据量大
2) Python 端 PNG 解码 + OCR 推理非常吃 CPU
3) OCR推理使用的是CPU

本项目已优先尝试使用 **raw screencap**（不走 PNG），能轻微降低解码开销；若你的设备/模拟器不支持，会自动回退到 PNG，并在日志提示。

### OCR 推理加速（可选、推荐）

RapidOCR（onnxruntime）支持通过环境变量选择加速：

- DirectML（默认，Windows 推荐）：
  - 安装：`.\.venv\Scripts\python -m pip install -r requirements-gpu.txt`
  - 启用：不需要额外配置（默认即为 `dml`）
- NVIDIA CUDA（较麻烦，未测试）：
  - 安装：`.\.venv\Scripts\python -m pip install -r requirements-gpu.txt`
  - 启用：设置环境变量 `BB_OCR_ACCEL=cuda`
- CPU：
  - 启用：设置环境变量 `BB_OCR_ACCEL=cpu`

示例（PowerShell）：

```powershell
$env:BB_OCR_ACCEL = 'cuda'   # 或 'dml'
.\.venv\Scripts\python scr\app.py
```

可用 provider 快速自检：

```powershell
.\.venv\Scripts\python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## 识别模式

通过环境变量切换默认识别模式：

- `BB_VISION_MODE=template`：只做模板匹配（默认）
- `BB_VISION_MODE=ocr`：只做 OCR

示例：

```powershell
$env:BB_VISION_MODE = 'ocr'
.\.venv\Scripts\python scr\app.py
```

## 模板图

模板图位于 `assets/images/`：

- `menu.png`, `skip1.png`, `skip2.png`
- `read.png`, `no_voice.png`, `lock.png`, `enter.png`

可在界面里调阈值（0~1）：越小越容易触发。

## 调试脚本

- 菜单识别单测（模板/OCR/都测）：

```powershell
.\.venv\Scripts\python scr\debug_menu.py --serial 127.0.0.1:16384 --mode both --save
```

如果你使用的不是 MuMu，替换 `--serial` 为 `adb devices` 输出的序列号即可。
