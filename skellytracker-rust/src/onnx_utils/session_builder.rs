//! GPU-aware ORT session construction matching Python's `build_tuned_ort_session`.
//!
//! Ports provider resolution, CUDA/TRT options, engine cache management from
//! `skellytracker/utilities/gpu_utils/ort_session_utils.py`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use ort::ep::{self, ArbitrarilyConfigurableExecutionProvider, ExecutionProviderDispatch};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

// ---------------------------------------------------------------------------
// Provider resolution
// ---------------------------------------------------------------------------

/// Which execution provider to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// TensorRT — best GPU performance, requires engine compilation
    TensorRT,
    /// CUDA — GPU inference (no TRT engine compilation)
    CUDA,
    /// CPU fallback
    CPU,
}

/// Resolve the requested provider against what's available on this system.
/// Mirrors Python's `resolve_provider()`: TRT → CUDA → CPU.
pub fn resolve_provider(requested: Provider) -> Provider {
    // For now, always try the requested provider.
    // The ort crate's SessionBuilder will handle fallback internally
    // when the requested EP isn't available.
    requested
}

// ---------------------------------------------------------------------------
// Engine cache management
// ---------------------------------------------------------------------------

fn default_engine_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("skellytracker")
        .join("trt_engines")
}

/// Build a tuned ORT session with CUDA/TensorRT execution providers.
///
/// Provider fallback chain: TRT → CUDA → CPU (matching Python exactly).
pub fn build_tuned_ort_session(
    onnx_path: &Path,
    provider: Provider,
    engine_cache_dir: Option<&Path>,
    fp16: bool,
    log_label: &str,
) -> Result<Session, Box<dyn std::error::Error>> {
    let engine_cache_dir = engine_cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_engine_cache_dir);
    std::fs::create_dir_all(&engine_cache_dir)?;

    eprintln!(
        "[skellytracker-rust] Building ORT session for '{log_label}' with provider={:?}, fp16={fp16}",
        provider
    );

    let start = Instant::now();

    // Build provider list with fallback chain
    let providers: Vec<ExecutionProviderDispatch> = match provider {
        Provider::TensorRT => {
            eprintln!(
                "[skellytracker-rust]   TRT: engine cache dir: {}",
                engine_cache_dir.display()
            );
            vec![
                build_trt_ep(&engine_cache_dir, fp16),
                build_cuda_ep(),
                ep::CPUExecutionProvider::default().build(),
            ]
        }
        Provider::CUDA => vec![
            build_cuda_ep(),
            ep::CPUExecutionProvider::default().build(),
        ],
        Provider::CPU => vec![ep::CPUExecutionProvider::default().build()],
    };

    let provider_names: Vec<&str> = match provider {
        Provider::TensorRT => vec!["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        Provider::CUDA => vec!["CUDAExecutionProvider", "CPUExecutionProvider"],
        Provider::CPU => vec!["CPUExecutionProvider"],
    };
    eprintln!(
        "[skellytracker-rust]   provider chain: {}",
        provider_names.join(" → ")
    );
    eprintln!("[skellytracker-rust]   loading model: {} ...", onnx_path.display());

    eprintln!("[skellytracker-rust]   building session + loading model...");
    let before_commit = Instant::now();
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?
        .with_execution_providers(providers)?
        .commit_from_file(onnx_path)?;
    let commit_elapsed = before_commit.elapsed();
    eprintln!(
        "[skellytracker-rust]   session ready in {:.3}s — running warmup pass...",
        commit_elapsed.as_secs_f64()
    );

    let elapsed = start.elapsed();
    let elapsed_s = elapsed.as_secs_f64();
    eprintln!(
        "[skellytracker-rust]   '{log_label}' session ready in {elapsed_s:.1}s"
    );

    if matches!(provider, Provider::TensorRT) && elapsed_s > 30.0 {
        eprintln!(
            "[skellytracker-rust]   TRT engine for '{log_label}' compiled and cached — \
             next run will load in seconds."
        );
    }

    Ok(session)
}

/// Build the CUDA execution provider with tuned options.
fn build_cuda_ep() -> ExecutionProviderDispatch {
    ep::CUDA::default()
        .with_device_id(0)
        .with_arbitrary_config("cudnn_conv_algo_search", "EXHAUSTIVE")
        .with_arbitrary_config("arena_extend_strategy", "kSameAsRequested")
        .with_arbitrary_config("do_copy_in_default_stream", "1")
        .with_arbitrary_config("gpu_mem_limit", (2u64 * 1024 * 1024 * 1024).to_string()) // 2 GiB
        .build()
}

/// Build the TensorRT execution provider with engine caching enabled.
fn build_trt_ep(engine_cache_dir: &Path, fp16: bool) -> ExecutionProviderDispatch {
    let cache_dir = engine_cache_dir.display().to_string();
    ep::TensorRT::default()
        .with_device_id(0)
        .with_arbitrary_config("trt_fp16_enable", if fp16 { "1" } else { "0" })
        .with_arbitrary_config("trt_engine_cache_enable", "1")
        .with_arbitrary_config("trt_engine_cache_path", cache_dir.as_str())
        .with_arbitrary_config("trt_timing_cache_enable", "1")
        .with_arbitrary_config("trt_timing_cache_path", cache_dir.as_str())
        .with_arbitrary_config("trt_max_workspace_size", (2u64 * 1024 * 1024 * 1024).to_string()) // 2 GiB
        .build()
}
