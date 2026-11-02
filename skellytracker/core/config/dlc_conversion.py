"""Optional offline DLC conversion, isolated from frame processing and training packages."""
from pathlib import Path
import json
import os
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field


class DlcConversionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pose_config_path: Path | None = None
    checkpoint_path: Path | None = None
    project_config_path: Path | None = None
    image_height: int | None = Field(default=None, ge=32, le=8192)
    image_width: int | None = Field(default=None, ge=32, le=8192)


def converter_python(python_executable: str | None = None) -> str:
    """Resolve optional conversion tooling without installing packages or using training GPUs."""
    explicit = python_executable or os.environ.get("SKELLYTRACKER_DLC_CONVERTER_PYTHON")
    if explicit:
        return explicit
    setting = Path.home() / '.config' / 'skellytracker' / 'dlc-converter.json'
    if setting.is_file():
        try:
            configured = Path(json.loads(setting.read_text(encoding='utf8'))['python_executable']).expanduser()
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Invalid converter settings in {setting}: expected python_executable') from error
        if not configured.is_absolute() or not configured.is_file():
            raise ValueError(f'Configured converter Python no longer exists: {configured}. Update {setting}.')
        return str(configured)
    tools = Path(__file__).resolve().parents[3] / 'tools' / 'dlc_inference'
    adjacent = tools / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    return str(adjacent) if adjacent.is_file() else sys.executable


def dlc_conversion_readiness(*, python_executable: str | None = None) -> dict:
    """Inspect optional conversion dependencies without importing them in Core."""
    default_script = Path(__file__).resolve().parents[3] / "tools/dlc_inference/export_model.py"
    script = Path(os.environ.get("SKELLYTRACKER_DLC_EXPORT_SCRIPT", str(default_script)))
    status = {"available": False, "supported_model": "DLC PyTorch single-subject ResNet50 GroupNorm",
              "missing_packages": [], "detail": ""}
    if not script.is_file():
        status["detail"] = "Configure SKELLYTRACKER_DLC_EXPORT_SCRIPT to the optional DLC exporter, or select an already converted sidecar."
        return status
    try:
        executable = converter_python(python_executable)
        probe = subprocess.run([executable, "-c", "import importlib.util,json; print(json.dumps([n for n in ('torch','timm','onnx','onnxscript','yaml') if importlib.util.find_spec(n) is None]))"],
                               capture_output=True, text=True, timeout=15, check=False,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if probe.returncode:
            raise ValueError(probe.stderr.strip()[-1000:])
        status["missing_packages"] = json.loads(probe.stdout)
        status["available"] = not status["missing_packages"]
        status["detail"] = ("Conversion is available on CPU; inference uses ONNX Runtime."
                            if status["available"] else
                            "Configure SKELLYTRACKER_DLC_CONVERTER_PYTHON to the optional conversion environment; missing: "
                            + ", ".join(status["missing_packages"]))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        status["detail"] = "Cannot inspect the DLC conversion Python; configure SKELLYTRACKER_DLC_CONVERTER_PYTHON. " + str(error)
    return status


def convert_dlc_model(request: DlcConversionRequest, output_directory: Path, *,
                      python_executable: str | None = None) -> Path:
    """Run the optional converter on CPU and return its validated sidecar.

    The independent converter environment supplies Torch/timm; deployed inference
    requires neither. Source checkouts include the tool. Other installations can
    configure SKELLYTRACKER_DLC_EXPORT_SCRIPT to its separately installed location.
    """
    if request.checkpoint_path is None or request.pose_config_path is None or request.image_height is None or request.image_width is None:
        from skellytracker.core.config.dlc_import import resolve_dlc_request
        request = resolve_dlc_request(request)
    for path in (request.pose_config_path, request.checkpoint_path, request.project_config_path):
        if path is not None and not path.is_file():
            raise ValueError(f"DLC input file does not exist: {path}")
    if output_directory.exists():
        raise ValueError("DLC conversion output already exists; choose a new model directory")
    default_script = Path(__file__).resolve().parents[3] / "tools/dlc_inference/export_model.py"
    script = Path(os.environ.get("SKELLYTRACKER_DLC_EXPORT_SCRIPT", str(default_script)))
    if not script.is_file():
        raise ValueError("DLC conversion tools are unavailable; configure SKELLYTRACKER_DLC_EXPORT_SCRIPT or import a converted model sidecar")
    executable = converter_python(python_executable)
    command = [executable, str(script), str(request.pose_config_path.resolve()),
               str(request.checkpoint_path.resolve()), str(output_directory.resolve()),
               "--height", str(request.image_height), "--width", str(request.image_width),
               "--resize-method", "fit"]
    if request.project_config_path is not None:
        command.extend(("--project-config", str(request.project_config_path.resolve())))
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf8", errors="replace",
                            env=environment, timeout=600, check=False,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-6000:]
        raise ValueError("DLC conversion failed. Only the verified ResNet50 GroupNorm single-subject model is supported; "
                         "conversion requires the optional Torch/timm/ONNX export environment "
                         "(SKELLYTRACKER_DLC_CONVERTER_PYTHON).\n" + detail)
    from skellytracker.core.tracker.tracker_presets import dlc_tracker_metadata
    sidecar_path = output_directory / "dlc_resnet50_gn.yaml"
    dlc_tracker_metadata(sidecar_path=sidecar_path)
    return sidecar_path.resolve()


def prepare_dlc_model(request: DlcConversionRequest, cache_directory: Path, *, on_progress=None,
                      python_executable: str | None = None) -> Path:
    """Resolve, reuse or convert a project/checkpoint through Tracker's import cache."""
    from skellytracker.core.config.dlc_import import prepare_model
    return prepare_model(request, cache_directory, on_progress=on_progress, python_executable=python_executable)
