"""Read-only evidence from an existing TensorRT cache directory.

This is not the mutating cache validator and never loads a native session. The
current cache manifest identifies runtime versions only. Even a matching manifest
and nonempty engine files cannot prove model, device, precision, shape-profile or
graph-partition compatibility. Callers must not treat CANDIDATE as ready or reuse
an engine solely because its directory is named ``batch_N``.

Pass the exact cache directory and known runtime versions explicitly. Inspection
does not choose a GPU, import ORT/TRT, resolve/download models, or prepare ONNX.
"""

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class CacheEvidenceState(StrEnum):
    MISSING = "missing"
    UNKNOWN = "unknown"
    RUNTIME_MISMATCH = "runtime_mismatch"
    CANDIDATE = "candidate"


@dataclass(frozen=True)
class RuntimeVersions:
    ort_version: str
    trt_version: str

    def __post_init__(self) -> None:
        for value in (self.ort_version, self.trt_version):
            if not value.strip() or value.strip().lower() == "unavailable":
                raise ValueError("Runtime versions must be known, nonempty version strings")


@dataclass(frozen=True)
class TrtCacheInspection:
    directory: Path
    state: CacheEvidenceState
    reason: str
    engine_files: tuple[Path, ...] = ()
    recorded_versions: RuntimeVersions | None = None

    @property
    def ready(self) -> bool:
        """Disk inspection cannot certify a loaded, validated, warmed session."""
        return False


def inspect_trt_cache(
    directory: Path, *, expected_versions: RuntimeVersions | None = None,
) -> TrtCacheInspection:
    """Inspect existing cache evidence without creating or modifying any files.

    Missing/unreadable/malformed evidence stays explicit. Matching versions are
    necessary but insufficient for compatibility. Files can change after this
    snapshot; preparation must revalidate them through its owning lifecycle.
    Native readiness and cross-capacity reuse are deliberately not certified here.
    """
    engines: tuple[Path, ...] = ()
    try:
        if not directory.exists():
            return TrtCacheInspection(directory, CacheEvidenceState.MISSING, "Cache directory does not exist")
        if not directory.is_dir():
            return TrtCacheInspection(directory, CacheEvidenceState.UNKNOWN, "Cache path is not a directory")
        engines = tuple(sorted(path for path in directory.glob("*.engine") if path.is_file()))
        manifest = directory / "cache_manifest.json"
        if not manifest.is_file():
            return TrtCacheInspection(directory, CacheEvidenceState.UNKNOWN,
                                      "Runtime manifest is missing", engines)
        # Existing manifests are tiny. Bound reads of malformed/external files.
        with manifest.open("rb") as source:
            payload = source.read(65537)
        if len(payload) > 65536:
            return TrtCacheInspection(directory, CacheEvidenceState.UNKNOWN,
                                      "Runtime manifest exceeds inspection size limit", engines)
        raw = json.loads(payload.decode("utf-8"))
        if (not isinstance(raw, dict) or set(raw) != {"ort_version", "trt_version"}
                or not all(isinstance(value, str) for value in raw.values())):
            return TrtCacheInspection(directory, CacheEvidenceState.UNKNOWN,
                                      "Runtime manifest format is not recognized", engines)
        recorded = RuntimeVersions(ort_version=raw["ort_version"], trt_version=raw["trt_version"])
        if expected_versions is None:
            return TrtCacheInspection(directory, CacheEvidenceState.UNKNOWN,
                                      "Current runtime versions were not supplied", engines, recorded)
        if recorded != expected_versions:
            return TrtCacheInspection(directory, CacheEvidenceState.RUNTIME_MISMATCH,
                                      "Recorded runtime versions differ; no files were invalidated", engines, recorded)
        if not engines:
            return TrtCacheInspection(directory, CacheEvidenceState.MISSING,
                                      "No engine files are present", engines, recorded)
        if any(path.stat().st_size == 0 for path in engines):
            return TrtCacheInspection(directory, CacheEvidenceState.UNKNOWN,
                                      "An engine file is empty", engines, recorded)
        return TrtCacheInspection(
            directory, CacheEvidenceState.CANDIDATE,
            "Runtime versions match; model/device/precision/profile compatibility and native readiness remain unverified",
            engines, recorded,
        )
    except (OSError, ValueError, RecursionError) as error:
        return TrtCacheInspection(directory, CacheEvidenceState.UNKNOWN,
                                  f"Cache evidence could not be read: {type(error).__name__}", engines)
