//! Generic model resolution and download — framework-agnostic.
//!
//! Ported from `skellytracker/utilities/gpu_utils/model_registry.py`.
//!
//! `ModelSource` says *where* to get a model file (URL, Hugging Face Hub, or
//! local path). `resolve_model_path()` returns a local `Path`, downloading
//! and caching if necessary. `ModelSpec` bundles a source with the metadata
//! needed to run inference (input size, keypoint count, preprocessing mode).

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// TrackerPreset
// ---------------------------------------------------------------------------

/// High-level performance tier that bundles model choices for all components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackerPreset {
    Light,
    Medium,
    Heavy,
}

// ---------------------------------------------------------------------------
// ModelSource
// ---------------------------------------------------------------------------

/// Where to get a model file. Set exactly one field.
#[derive(Debug, Clone)]
pub struct ModelSource {
    /// Direct download URL (OpenMMLab CDN .zip or direct .onnx).
    pub url: Option<String>,
    /// Hugging Face Hub repository ID.
    pub hf_repo: Option<String>,
    /// File path within HF repo.
    pub hf_filename: Option<String>,
    /// Absolute or relative path to an already-downloaded model file.
    pub local_path: Option<PathBuf>,
}

impl ModelSource {
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            url: Some(url.into()),
            hf_repo: None,
            hf_filename: None,
            local_path: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocessing mode
// ---------------------------------------------------------------------------

/// Which preprocessing pipeline to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessMode {
    /// RTMO letterbox + BGR mean/std normalization → NMS decode.
    Rtmo,
    /// RTMPose/simple letterbox + BGR mean/std → SIMCC decode.
    RtmPoseLetterbox,
    /// Simple letterbox, no normalization → caller handles output decode.
    SimpleLetterbox,
    /// RGB conversion + [0,1] scaling + resize → direct coordinate regression.
    MediaPipe,
    /// No preprocessing — raw input.
    None_,
}

// ---------------------------------------------------------------------------
// ModelSpec
// ---------------------------------------------------------------------------

/// Descriptor for a single ML model used by a tracker. Framework-agnostic.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub source: ModelSource,
    pub input_size: (u32, u32), // (height, width)
    pub num_keypoints: u32,
    pub preprocess_mode: PreprocessMode,
    pub mean: Option<[f32; 3]>, // BGR channel-wise mean
    pub std: Option<[f32; 3]>,  // BGR channel-wise std
    pub simcc_split_ratio: Option<f32>,
    pub supports_batching: Option<bool>, // None = probe at runtime
}

impl ModelSpec {
    // -- Body (RTMO one-stage) --------------------------------------------------

    pub fn rtmo_light() -> Self {
        Self {
            source: ModelSource::from_url(MODEL_URLS["rtmo-s"].clone()),
            input_size: (640, 640),
            num_keypoints: 17,
            preprocess_mode: PreprocessMode::Rtmo,
            mean: Some([123.675, 116.28, 103.53]),
            std: Some([58.395, 57.12, 57.375]),
            simcc_split_ratio: None,
            supports_batching: None,
        }
    }

    pub fn rtmo_medium() -> Self {
        Self {
            source: ModelSource::from_url(MODEL_URLS["rtmo-m"].clone()),
            input_size: (640, 640),
            num_keypoints: 17,
            preprocess_mode: PreprocessMode::Rtmo,
            mean: Some([123.675, 116.28, 103.53]),
            std: Some([58.395, 57.12, 57.375]),
            simcc_split_ratio: None,
            supports_batching: None,
        }
    }

    pub fn rtmo_heavy() -> Self {
        Self {
            source: ModelSource::from_url(MODEL_URLS["rtmo-l"].clone()),
            input_size: (640, 640),
            num_keypoints: 17,
            preprocess_mode: PreprocessMode::Rtmo,
            mean: Some([123.675, 116.28, 103.53]),
            std: Some([58.395, 57.12, 57.375]),
            simcc_split_ratio: None,
            supports_batching: None,
        }
    }

    // -- Hand (MediaPipe hand landmark — default) ----------------------------------

    pub fn mediapipe_hand_landmark() -> Self {
        Self {
            source: ModelSource::from_url(MODEL_URLS["mediapipe-hand-landmark"].clone()),
            input_size: (224, 224),
            num_keypoints: 21,
            preprocess_mode: PreprocessMode::MediaPipe,
            mean: None,
            std: None,
            simcc_split_ratio: None,
            supports_batching: None,
        }
    }

    // -- Face (RTMPose SIMCC, LaPa 106-point) -------------------------------------

    pub fn rtmpose_face() -> Self {
        Self {
            source: ModelSource::from_url(MODEL_URLS["rtmpose-face"].clone()),
            input_size: (256, 256),
            num_keypoints: 106,
            preprocess_mode: PreprocessMode::RtmPoseLetterbox,
            mean: Some([123.675, 116.28, 103.53]),
            std: Some([58.395, 57.12, 57.375]),
            simcc_split_ratio: Some(2.0),
            supports_batching: None,
        }
    }

    // -- Preset dispatch ----------------------------------------------------------

    pub fn body_for_preset(preset: TrackerPreset) -> Self {
        match preset {
            TrackerPreset::Light => Self::rtmo_light(),
            TrackerPreset::Medium => Self::rtmo_medium(),
            TrackerPreset::Heavy => Self::rtmo_heavy(),
        }
    }
}

// ---------------------------------------------------------------------------
// Well-known ONNX model URLs
// ---------------------------------------------------------------------------

static MODEL_URLS: LazyLock<HashMap<&'static str, String>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("rtmo-s", "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip".into());
    m.insert("rtmo-m", "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip".into());
    m.insert("rtmo-l", "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip".into());
    m.insert("rtmpose-hand", "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip".into());
    m.insert("rtmpose-face", "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.zip".into());
    m.insert("yolox-tiny", "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip".into());
    m.insert("yolox-m", "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip".into());
    m.insert("rtmpose-s", "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip".into());
    m.insert("rtmpose-m", "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip".into());
    m.insert("rtmw-l-m_256x192", "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip".into());
    m.insert("rtmw-x-l_256x192", "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip".into());
    m.insert("rtmw-x-l_384x288", "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip".into());
    m.insert("mediapipe-hand-landmark", "https://raw.githubusercontent.com/PINTO0309/hand-gesture-recognition-using-onnx/main/model/hand_landmark/hand_landmark_sparse_Nx3x224x224.onnx".into());
    m
});

// ---------------------------------------------------------------------------
// Model resolution — download + cache
// ---------------------------------------------------------------------------

fn default_cache_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("skellytracker")
        .join("models")
}

/// Download and cache a model from a URL, returning the local path.
///
/// For .zip URLs (OpenMMLab CDN): extracts the first .onnx in the archive.
/// For direct .onnx URLs: downloads as-is.
pub fn resolve_model_path(key: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let url = MODEL_URLS
        .get(key)
        .cloned()
        .ok_or_else(|| format!("Unknown model key: {key}"))?;

    let cache_dir = default_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    let is_zip = url.ends_with(".zip");

    let filename = url.rsplit('/').next().unwrap_or("model.onnx");
    let cached_name = if is_zip {
        filename.replace(".zip", ".onnx")
    } else {
        filename.to_string()
    };
    let cached_path = cache_dir.join(&cached_name);

    if cached_path.exists() {
        eprintln!("[skellytracker-rust] Using cached model: {}", cached_path.display());
        return Ok(cached_path);
    }

    eprintln!("[skellytracker-rust] Downloading model: {key} <- {url}");
    let response = ureq::get(&url).call()?;
    let total_size: usize = response
        .header("Content-Length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let mut data = Vec::with_capacity(total_size.max(1024 * 1024));
    response.into_reader().read_to_end(&mut data)?;

    if is_zip {
        let cursor = std::io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)?;
        let onnx_idx = (0..archive.len())
            .find(|&i| archive.by_index(i).map(|e| e.name().ends_with(".onnx")).unwrap_or(false))
            .ok_or_else(|| format!("No .onnx file found in zip: {url}"))?;

        let mut entry = archive.by_index(onnx_idx)?;
        let mut onnx_data = Vec::new();
        entry.read_to_end(&mut onnx_data)?;
        std::fs::write(&cached_path, &onnx_data)?;
    } else {
        std::fs::write(&cached_path, &data)?;
    }

    eprintln!("[skellytracker-rust] Model cached: {}", cached_path.display());
    Ok(cached_path)
}

/// Resolve multiple models sequentially. Downloads are I/O-bound (network),
/// so sequential is fine — parallel gains mostly from concurrent HTTP connections
/// which ureq's blocking model doesn't exploit well.
pub fn resolve_model_paths(
    keys: &[&str],
) -> Result<HashMap<String, PathBuf>, Box<dyn std::error::Error>> {
    let mut map = HashMap::new();
    for &key in keys {
        match resolve_model_path(key) {
            Ok(path) => {
                map.insert(key.to_string(), path);
            }
            Err(e) => {
                eprintln!("[skellytracker-rust] Failed to resolve model '{key}': {e}");
            }
        }
    }
    Ok(map)
}
