use std::path::PathBuf;

fn main() {
    let opencv_root = PathBuf::from("C:/tools/opencv/build");

    if !opencv_root.exists() {
        println!("cargo:warning=OpenCV not found at C:/tools/opencv/build");
        return;
    }

    let bin_dir = opencv_root.join("x64/vc16/bin");

    if !bin_dir.exists() {
        println!("cargo:warning=OpenCV bin dir not found at {}", bin_dir.display());
        return;
    }

    // Derive the target profile directory from OUT_DIR.
    // OUT_DIR is typically: target/<profile>/build/<crate>-<hash>/out
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_profile_dir = out_dir
        .ancestors()
        .nth(3) // walk up: out -> <hash> -> build -> <profile> -> target
        .expect("Failed to derive target dir from OUT_DIR");

    let target_deps_dir = target_profile_dir.join("deps");

    // Copy OpenCV DLLs to cargo output dirs so `cargo test` and `cargo run` work.
    // At Python import time, the hot-swappable adapter calls os.add_dll_directory()
    // with the OpenCV bin path, so the .pyd finds the DLLs without bundling.
    let dlls = ["opencv_world4130.dll", "opencv_videoio_ffmpeg4130_64.dll"];

    for dll in &dlls {
        let src = bin_dir.join(dll);
        if !src.exists() {
            println!("cargo:warning=OpenCV DLL not found: {}", src.display());
            continue;
        }
        let _ = std::fs::copy(&src, target_profile_dir.join(dll));
        let _ = std::fs::copy(&src, target_deps_dir.join(dll));
    }

    println!("cargo:rerun-if-changed=build.rs");
}
