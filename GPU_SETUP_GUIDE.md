# SkellyTracker GPU Setup Guide (RTMPose)

This guide walks you through getting GPU-accelerated pose estimation working with SkellyTracker's RTMPose tracker. GPU acceleration makes tracking significantly faster — but getting it set up requires some extra steps depending on your hardware.

## Which GPU do you have?

The setup path depends entirely on your hardware. Here's the quick version:

| Your GPU | Your OS | What to install | rtmlib device | Setup difficulty |
|---|---|---|---|---|
| NVIDIA | Windows or Linux | `skellytracker[rtmpose-gpu]` | `cuda` | Medium (CUDA + cuDNN) |
| AMD | Windows | `skellytracker[rtmpose-directml]` | `directml` | Easy (just pip) |
| AMD | Linux | `skellytracker[rtmpose-cpu]` + ROCm | `rocm` | Hard (limited GPU support) |
| Apple Silicon (M1/M2/M3/M4) | macOS | `skellytracker[rtmpose-cpu]` | `mps` | Easy (automatic) |
| Intel Arc | Windows | `skellytracker[rtmpose-directml]` | `directml` | Easy (just pip) |
| No dedicated GPU | Any | `skellytracker[rtmpose-cpu]` | `cpu` | None |

> **Note:** The `rtmpose-directml` extra doesn't exist yet in skellytracker — see the "AMD / DirectML" section below for what needs to happen to enable it.

---

## Option 1: NVIDIA GPU (CUDA)

This is the best-supported and fastest path.

### Prerequisites

You need an NVIDIA GPU with CUDA support and **three** things installed on your system:

1. **NVIDIA Driver** (recent version supporting CUDA 12.x)
2. **CUDA Toolkit 12.x**
3. **cuDNN 9.x** (separate download from CUDA — this is the one everyone forgets)

### Step 1: Check your GPU and driver

Open a terminal (Command Prompt on Windows) and run:

```
nvidia-smi
```

You should see a table with your GPU name and a CUDA version. If this command isn't found, you need to install NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx).

**You need CUDA 12.x** shown in the output. If it shows 11.x, update your NVIDIA driver.

### Step 2: Install CUDA Toolkit and cuDNN

#### Windows

1. Download and install the [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads)
2. Download and install [cuDNN 9.x](https://developer.nvidia.com/cudnn-downloads) (free NVIDIA developer account required)
   - If you download the zip version, copy the files into your CUDA Toolkit directory (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\`)
3. Install the [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) (latest x64)
4. **Restart your computer**

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

For cuDNN, download the `.deb` from [NVIDIA's cuDNN page](https://developer.nvidia.com/cudnn-downloads), then:

```bash
sudo dpkg -i cudnn-local-repo-<your-version>.deb
sudo cp /var/cudnn-local-repo-*/*.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libcudnn9-cuda-12
```

#### Shortcut: If you already have PyTorch with CUDA

If PyTorch with CUDA support is already installed, `onnxruntime-gpu` can reuse PyTorch's bundled CUDA and cuDNN libraries. Just make sure to `import torch` before creating any ONNX Runtime sessions, or call `onnxruntime.preload_dlls()`. This can save you from installing CUDA Toolkit and cuDNN separately.

### Step 3: Install SkellyTracker

```bash
pip install skellytracker[rtmpose-gpu]
```

Or combined with mediapipe:

```bash
pip install skellytracker[mediapipe,rtmpose-gpu]
```

### Step 4: Verify

```python
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)
assert 'CUDAExecutionProvider' in providers, "CUDA not available!"
print("NVIDIA GPU is ready!")
```

---

## Option 2: AMD GPU on Windows (DirectML)

DirectML is Microsoft's hardware-agnostic GPU acceleration library. It works with any DirectX 12 GPU — AMD, Intel, NVIDIA, Qualcomm — and requires **zero system-level setup**. No toolkit installs, no PATH wrangling, no cuDNN. Just pip install and go.

### Current status (action needed)

DirectML support requires two small changes — one to skellytracker, one to rtmlib.

**1. Add a new extra to skellytracker's `pyproject.toml`:**

```toml
[project.optional-dependencies]
rtmpose-directml = ["rtmlib==0.0.14", "onnxruntime-directml"]
```

And update the conflicts list (since `onnxruntime`, `onnxruntime-gpu`, and `onnxruntime-directml` all conflict):

```toml
[tool.uv]
conflicts = [
    [
        { extra = "rtmpose-cpu" },
        { extra = "rtmpose-gpu" },
        { extra = "rtmpose-directml" },
    ],
]
```

**2. Add one line to rtmlib's `base.py`:**

In `rtmlib/tools/base.py`, the `RTMLIB_SETTINGS` dict maps device strings to ONNX Runtime execution providers. It already has `cpu`, `cuda`, `rocm`, and `mps` — but not `directml`. The fix is adding one line:

```python
'onnxruntime': {
    'cpu': 'CPUExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'rocm': 'ROCMExecutionProvider',
    'directml': 'DmlExecutionProvider',  # <-- add this
    'mps': 'CoreMLExecutionProvider' if check_mps_support() else 'CPUExecutionProvider'
},
```

This is a one-line PR to rtmlib. Until that lands upstream, you can monkey-patch it:

```python
import rtmlib.tools.base as rtmlib_base
rtmlib_base.RTMLIB_SETTINGS['onnxruntime']['directml'] = 'DmlExecutionProvider'
```

### Setup (once the extra exists)

```bash
pip install skellytracker[rtmpose-directml]
```

That's it. No CUDA, no cuDNN, no toolkit, no driver installs beyond what Windows already has. Any DirectX 12 GPU works — AMD Radeon, Intel Arc, even NVIDIA (though CUDA is faster on NVIDIA hardware).

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
- DirectML is in maintenance mode at Microsoft (still works, still gets ONNX Runtime updates, but not actively getting new standalone features)
- May be slightly slower than CUDA on equivalent NVIDIA hardware, since CUDA has deeper NVIDIA-specific optimizations

---

## Option 3: Apple Silicon Mac (MPS/CoreML)

rtmlib already has built-in support for Apple Silicon via CoreML. When you set `device='mps'`, it automatically uses the `CoreMLExecutionProvider` if available, and falls back to CPU if not.

### Setup

```bash
pip install skellytracker[rtmpose-cpu]
```

You install the CPU extra — the base `onnxruntime` package (not `onnxruntime-gpu`) includes CoreML support on macOS automatically.

### Usage

When using rtmlib in your code, set `device='mps'`:

```python
device = 'mps'
backend = 'onnxruntime'
```

No other setup needed. Apple Silicon handles the acceleration natively.

---

## Option 4: AMD GPU on Linux (ROCm)

rtmlib has a `device='rocm'` option that maps to `ROCMExecutionProvider`. However, this path is significantly harder:

- ROCm only supports a [limited set of AMD GPUs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) (mostly datacenter and higher-end consumer cards like RX 7900 XTX)
- ROCm installation is complex and Linux-only
- You need a specific ONNX Runtime build for ROCm (not available as a simple pip package)

For most students with AMD GPUs on Linux, **the CPU backend is the practical choice**. RTMPose models are lightweight enough that CPU inference is still quite fast.

---

## Troubleshooting

### `nvidia-smi` is not recognized (Windows)

Install NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/download/index.aspx).

### `nvidia-smi` command not found (Linux)

```bash
sudo apt update
sudo apt install nvidia-driver-560  # or latest available version
sudo reboot
```

### `CUDAExecutionProvider` doesn't show up

This is the most common problem. Check:

1. **Is CUDA 12.x installed?** Run `nvcc --version`. If it says 11.x or "not found," install/update the CUDA Toolkit.
2. **Is cuDNN 9.x installed?** The CUDA Toolkit does NOT include cuDNN. It's a separate download.
3. **Are they on your PATH?**
   - **Windows**: CUDA bin directory (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`) and cuDNN bin directory must be in your system PATH.
   - **Linux**: Check with `ldconfig -p | grep cudnn`.
4. **Did you restart?** Especially on Windows.

### `nvcc --version` and `nvidia-smi` show different CUDA versions

This is normal and not a problem. `nvidia-smi` shows the max CUDA version your *driver* supports. `nvcc --version` shows the CUDA *Toolkit* version installed. As long as toolkit version is less than or equal to the driver version, you're fine (e.g. driver shows 12.6, toolkit shows 12.4 = okay).

### `LoadLibrary failed with error 126` (Windows)

A required DLL wasn't found. Common culprits:

- `cudnn64_9.dll` — cuDNN 9 not installed or not on PATH
- `cublas64_12.dll` — CUDA Toolkit not on PATH
- `onnxruntime_providers_cuda.dll` — try `pip install --force-reinstall onnxruntime-gpu`

### Everything installed but inference is slow (not using GPU)

Your code might be defaulting to CPU. Make sure you're setting `device='cuda'` (or `'directml'` or `'mps'`) when initializing rtmlib — not `'cpu'`.

### `onnxruntime` packages conflict with each other

The three GPU-related packages — `onnxruntime` (CPU), `onnxruntime-gpu` (CUDA), and `onnxruntime-directml` (DirectML) — all conflict. You can only have one installed. When switching, uninstall the old one first:

```bash
pip uninstall onnxruntime onnxruntime-gpu onnxruntime-directml
pip install skellytracker[rtmpose-gpu]  # or whichever extra you want
```

### CUDA out-of-memory errors

Your GPU doesn't have enough VRAM. Options: close other GPU-consuming apps, reduce input resolution, or fall back to CPU.

### Linux: `libcudnn.so` not found

```bash
ldconfig -p | grep cudnn  # check if installed
sudo ldconfig              # update library cache if installed but not found
```

---

## Quick reference

| What you want | Install command | rtmlib device |
|---|---|---|
| NVIDIA GPU | `pip install skellytracker[rtmpose-gpu]` | `cuda` |
| AMD/Intel GPU (Windows) | `pip install skellytracker[rtmpose-directml]` * | `directml` * |
| Apple Silicon Mac | `pip install skellytracker[rtmpose-cpu]` | `mps` |
| CPU (any platform) | `pip install skellytracker[rtmpose-cpu]` | `cpu` |
| Mediapipe + NVIDIA GPU | `pip install skellytracker[mediapipe,rtmpose-gpu]` | `cuda` |

\* Requires adding `rtmpose-directml` extra to skellytracker's pyproject.toml and a one-line PR to rtmlib — see the DirectML section.
