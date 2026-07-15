# SkellyTracker GPU Setup Guide

This guide walks you through getting GPU-accelerated pose estimation working with SkellyTracker. GPU acceleration makes tracking significantly faster — but getting it set up requires picking the right install extra for your hardware.

## Which GPU do you have?

| Your GPU | Your OS | Install extra | Execution provider | Setup difficulty |
|---|---|---|---|---|
| NVIDIA | Windows or Linux | `skellytracker[all-cuda]` | `cuda` (auto-detected) | Easy (no system CUDA install needed) |
| NVIDIA (fastest) | Windows or Linux | `skellytracker[all-trt]` | `trt` (auto-detected) | Medium (first run compiles engines) |
| Apple Silicon (M1/M2/M3/M4) | macOS | `skellytracker[all-cpu]` | `coreml` (auto-detected) | Easy (automatic) |
| AMD / Intel / any GPU | Windows | `skellytracker[all-directml]` | `directml` (explicit) | Easy (no system install needed) |
| No dedicated GPU | Any | `skellytracker[all-cpu]` | `cpu` (auto-detected) | None |

> **Note on DirectML:** DirectML is not included in auto-detection — you must set `execution_provider="directml"` explicitly. See the DirectML section for details.

---

## Provider auto-detection

When you create a tracker without specifying an execution provider, SkellyTracker probes ONNX Runtime in this order and picks the best available option:

```
TensorRT → CUDA → CoreML (macOS only) → CPU
```

So on a machine with `all-trt` installed, inference runs on TensorRT automatically. On `all-cuda`, it runs on CUDA. On macOS with `all-cpu`, it uses CoreML on Apple Silicon. On any other machine, it falls back to CPU.

You can also specify the provider explicitly:

```python
from skellytracker.core.sessions.onnx_session import OnnxSessionConfig

config = OnnxSessionConfig(execution_provider="cuda", ...)
```

---

## Option 1: NVIDIA GPU (CUDA)

The `all-cuda` extra bundles the CUDA 12 and cuDNN 9 runtime libraries as pip packages. **You do not need to install the CUDA Toolkit or cuDNN manually.** SkellyTracker locates and loads the pip-installed DLLs automatically at session creation time.

### Prerequisites

- An NVIDIA GPU with CUDA support
- A recent NVIDIA driver (supporting CUDA 12.x) — check with `nvidia-smi`

That's it. No CUDA Toolkit download, no cuDNN download, no PATH changes.

### Install

```bash
pip install skellytracker[all-cuda]
```

Or with uv:

```bash
uv sync --extra all-cuda
```

### Verify

```python
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)
assert 'CUDAExecutionProvider' in providers, "CUDA not available!"
print("NVIDIA GPU (CUDA) is ready!")
```

---

## Option 2: NVIDIA GPU with TensorRT (fastest)

TensorRT is 2–5× faster than CUDA EP alone on NVIDIA hardware. The first run compiles and caches TRT engines (1–5 minutes depending on model and GPU). Subsequent runs load the cached engines instantly.

### Install

```bash
pip install skellytracker[all-trt]
```

Or with uv:

```bash
uv sync --extra all-trt
```

No system CUDA or cuDNN install needed — same as `all-cuda`.

### First run

The first time you run inference, expect a delay while TRT compiles engine files. You'll see log messages like:

```
Building ORT session: provider='trt' ...
[model] TRT session on device_id=0 (engine cache: ...) -- first-run compilation can take 1-5 minutes; subsequent runs load from cache instantly.
```

Engines are cached in `~/.cache/skellytracker/trt_engines/`. After the first run, startup is fast.

### Verify

```python
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)
assert 'TensorrtExecutionProvider' in providers, "TensorRT not available!"
print("NVIDIA GPU (TensorRT) is ready!")
```

---

## Option 3: Apple Silicon Mac (CoreML)

The base `onnxruntime` package (installed by `all-cpu`) includes CoreML support on macOS. When SkellyTracker detects `CoreMLExecutionProvider` is available, it uses it automatically — no extra configuration needed.

### Install

```bash
pip install skellytracker[all-cpu]
```

Or with uv:

```bash
uv sync --extra all-cpu
```

### Verify

```python
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)
# On Apple Silicon you should see 'CoreMLExecutionProvider' in the list
```

> **Note:** CoreML does not support batch sizes greater than 1 or FP16 inputs. SkellyTracker handles this automatically — batch size is forced to 1 when CoreML is active.

---

## Option 4: AMD / Intel GPU on Windows (DirectML) ⚠️ Experimental

> **Experimental:** DirectML support has not been tested on real hardware. It may work, but there are no guarantees around model compatibility, performance, or edge cases. Feedback welcome.

DirectML is Microsoft's hardware-agnostic GPU acceleration library. It works with any DirectX 12 GPU — AMD Radeon, Intel Arc, NVIDIA, Qualcomm — and requires **no system-level setup**. No CUDA Toolkit, no driver installs beyond what Windows already has. Just pip install and go.

### Install

```bash
pip install skellytracker[all-directml]
```

### Usage

DirectML is not included in auto-detection (which probes TRT → CUDA → CoreML → CPU). You must request it explicitly when creating a session:

```python
from skellytracker.core.sessions.onnx_session import OnnxSessionConfig

config = OnnxSessionConfig(execution_provider="directml", ...)
```

### Verify

```python
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)
assert 'DmlExecutionProvider' in providers, "DirectML not available!"
print("DirectML GPU is ready!")
```

### Limitations

- **Windows only** — DirectML is a DirectX 12 feature
- Must be specified explicitly; not auto-detected
- May be slightly slower than CUDA on equivalent NVIDIA hardware

---

## CPU fallback (any platform)

```bash
pip install skellytracker[all-cpu]
```

Works on Windows, Linux, and macOS. On Apple Silicon Macs, CoreML is used automatically (see Option 3).

---

## Troubleshooting

### `nvidia-smi` is not recognized (Windows) or not found (Linux)

Install NVIDIA drivers:
- **Windows**: Download from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx)
- **Linux (Ubuntu)**: `sudo apt install nvidia-driver-560` (or latest available), then `sudo reboot`

### `CUDAExecutionProvider` doesn't show up

Check that your NVIDIA driver supports CUDA 12.x — run `nvidia-smi` and look at the CUDA version in the top-right corner. If it shows 11.x, update your driver. No toolkit install is needed since SkellyTracker bundles the runtime via pip.

### `TensorrtExecutionProvider` doesn't show up

Requires `all-trt` or `onnx-trt` to be installed. The `all-cuda` extra does not include TensorRT.

### `onnxruntime` packages conflict with each other

The CPU (`onnxruntime`), CUDA (`onnxruntime-gpu`), and DirectML (`onnxruntime-directml`) packages all share the same import name and cannot coexist. If you switch extras, uninstall the old one first:

```bash
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml
pip install skellytracker[all-cuda]  # or whichever extra you want
```

### `AttributeError: module 'onnxruntime' has no attribute 'get_available_providers'`

Two conflicting `onnxruntime` builds are installed. See above — uninstall all, then reinstall one.

### CUDA out-of-memory errors

Your GPU doesn't have enough VRAM. Options: close other GPU-consuming apps, reduce input resolution, or fall back to CPU.

### Inference is slow (not using GPU)

Your code might be defaulting to CPU. Check the log output at startup — SkellyTracker logs which provider was selected:

```
auto_detect_provider: selected 'cuda' (available: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
```

If it says `cpu` and you expected GPU, verify that the right `onnxruntime` build is installed and that your GPU drivers are up to date.

### TRT engine compilation is slow

This is expected on first run. Cache is stored in `~/.cache/skellytracker/trt_engines/`. If you want to force a recompile (e.g. after a hardware change), delete that directory.

---

## Quick reference

| What you want | Install command | Provider |
|---|---|---|
| NVIDIA GPU (CUDA) | `pip install skellytracker[all-cuda]` | `cuda` (auto) |
| NVIDIA GPU (TensorRT, fastest) | `pip install skellytracker[all-trt]` | `trt` (auto) |
| Apple Silicon Mac | `pip install skellytracker[all-cpu]` | `coreml` (auto) |
| CPU (any platform) | `pip install skellytracker[all-cpu]` | `cpu` (auto) |
