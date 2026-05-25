//! ONNX Runtime utilities: session management, model download, and inference.
//!
//! For Phase 1 (CPU single-image), this provides:
//! - Model download from OpenMMLab CDN (zip → cached .onnx)
//! - ORT session creation (CPU execution provider)
//! - Single-image YOLOX + RTMPose two-stage inference

pub mod preprocessing;
pub mod postprocessing;
pub mod session_builder;

use std::io::Read;
use std::path::PathBuf;

use ort::session::Session;

// ---------------------------------------------------------------------------
// Model URLs (from Python model_registry.py)
// ---------------------------------------------------------------------------

const MODEL_URLS: &[(&str, &str)] = &[
    (
        "yolox-m",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    ),
    (
        "yolox-tiny",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
    ),
    (
        "rtmw-x-l_384x288",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip",
    ),
    (
        "rtmw-x-l_256x192",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
    ),
    (
        "rtmw-l-m_256x192",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
    ),
];

// ---------------------------------------------------------------------------
// Mode → (det_key, det_input_size, pose_key, pose_input_size)
// ---------------------------------------------------------------------------

pub struct ModeConfig {
    pub det_key: &'static str,
    pub det_input_size: (u32, u32), // (H, W)
    pub pose_key: &'static str,
    pub pose_input_size: (u32, u32), // (H, W)
}

pub fn mode_config(mode: &str) -> ModeConfig {
    match mode {
        "performance" => ModeConfig {
            det_key: "yolox-m",
            det_input_size: (640, 640),
            pose_key: "rtmw-x-l_384x288",
            pose_input_size: (288, 384),
        },
        "lightweight" => ModeConfig {
            det_key: "yolox-tiny",
            det_input_size: (416, 416),
            pose_key: "rtmw-l-m_256x192",
            pose_input_size: (192, 256),
        },
        _ => ModeConfig {
            // "balanced" — the default
            det_key: "yolox-m",
            det_input_size: (640, 640),
            pose_key: "rtmw-x-l_256x192",
            pose_input_size: (192, 256),
        },
    }
}

// ---------------------------------------------------------------------------
// Model download + cache
// ---------------------------------------------------------------------------

fn default_cache_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("skellytracker")
        .join("models")
}

fn model_url(key: &str) -> Option<&'static str> {
    MODEL_URLS
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, url)| *url)
}

/// Download and cache an ONNX model from an OpenMMLab CDN URL (zip format).
/// Returns the path to the cached .onnx file.
pub fn resolve_model(key: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let url = model_url(key)
        .ok_or_else(|| format!("Unknown model key: {key}"))?;

    let cache_dir = default_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    // Derive the expected ONNX filename from the URL stem.
    let filename = url.rsplit('/').next().unwrap_or("model.onnx");
    let onnx_name = filename.replace(".zip", ".onnx");
    let cached_onnx = cache_dir.join(&onnx_name);

    if cached_onnx.exists() {
        eprintln!("[skellytracker-rust] Using cached model: {}", cached_onnx.display());
        return Ok(cached_onnx);
    }

    eprintln!("[skellytracker-rust] Downloading model: {url}");
    let response = ureq::get(url).call()?;
    let total_size: usize = response
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut zip_data = Vec::with_capacity(total_size.max(1024 * 1024));
    response.into_reader().read_to_end(&mut zip_data)?;

    // Extract the first .onnx from the zip.
    let cursor = std::io::Cursor::new(zip_data);
    let mut archive = zip::ZipArchive::new(cursor)?;

    let mut onnx_idx: Option<usize> = None;
    for i in 0..archive.len() {
        let entry = archive.by_index(i)?;
        if entry.name().ends_with(".onnx") {
            onnx_idx = Some(i);
            break;
        }
    }

    let onnx_idx = onnx_idx
        .ok_or_else(|| format!("No .onnx file found in zip: {url}"))?;

    let mut entry = archive.by_index(onnx_idx)?;
    let mut onnx_data = Vec::new();
    entry.read_to_end(&mut onnx_data)?;
    std::fs::write(&cached_onnx, &onnx_data)?;

    eprintln!("[skellytracker-rust] Model cached: {}", cached_onnx.display());
    Ok(cached_onnx)
}

// ---------------------------------------------------------------------------
// ORT session wrapper
// ---------------------------------------------------------------------------

/// Holds two ORT sessions (YOLOX detection + RTMPose pose estimation)
/// and the preprocessing parameters needed for the two-stage pipeline.
pub struct RtmPoseOrtSession {
    pub det_session: Session,
    pub pose_session: Session,
    pub det_input_size: (u32, u32),
    pub pose_input_size: (u32, u32),
}

/// RTMPose preprocessing constants (from Python).
pub const POSE_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
pub const POSE_STD: [f32; 3] = [58.395, 57.12, 57.375];
pub const DET_NMS_THR: f32 = 0.45;
pub const DET_SCORE_THR: f32 = 0.7;
pub const SIMCC_SPLIT_RATIO: f32 = 2.0;

impl RtmPoseOrtSession {
    /// Create a new GPU-tuned session pair, downloading models if needed.
    /// Uses CUDA → CPU fallback. TensorRT deferred to a later phase.
    pub fn new(mode: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use crate::onnx_utils::session_builder::{Provider, build_tuned_ort_session};

        let cfg = mode_config(mode);

        eprintln!("[skellytracker-rust] Downloading/resolving models...");
        let det_path = resolve_model(cfg.det_key)?;
        let pose_path = resolve_model(cfg.pose_key)?;

        eprintln!("[skellytracker-rust] Building YOLOX session (CUDA)...");
        let det_session = build_tuned_ort_session(
            &det_path, Provider::CUDA, None, true, "yolox",
        )?;

        eprintln!("[skellytracker-rust] Building RTMPose session (CUDA)...");
        let pose_session = build_tuned_ort_session(
            &pose_path, Provider::CUDA, None, true, "rtmpose",
        )?;

        Ok(Self {
            det_session,
            pose_session,
            det_input_size: cfg.det_input_size,
            pose_input_size: cfg.pose_input_size,
        })
    }
}
