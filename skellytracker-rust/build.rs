fn main() {
    if std::env::var("OPENCV_LINK_PATHS").is_ok() {
        return; // User provided explicit config — trust it
    }

    // Common OpenCV install locations on Windows
    let candidates = [
        "C:/tools/opencv",
        "C:/opencv",
        "C:/Program Files/OpenCV",
    ];

    for root in &candidates {
        let include = std::path::Path::new(root).join("build/include");
        let lib = std::path::Path::new(root).join("build/x64/vc16/lib");
        let bin = std::path::Path::new(root).join("build/x64/vc16/bin");

        if include.exists() && lib.exists() {
            println!("cargo:warning=Found OpenCV at {}", root);

            // Headers
            println!("cargo:rustc-env=OPENCV_INCLUDE_PATHS={}", include.display());

            // Libs
            println!("cargo:rustc-env=OPENCV_LINK_PATHS={}", lib.display());

            // The Chocolatey build uses opencv_world (everything in one lib).
            // The version suffix matches the install (4.13.0 → 4130).
            // Try world first, fall back to per-module libs.
            let world_lib = lib.join("opencv_world4130.lib");
            if world_lib.exists() {
                println!("cargo:rustc-env=OPENCV_LINK_LIBS=opencv_world4130");
            } else {
                // Older installs with per-module libs
                println!("cargo:rustc-env=OPENCV_LINK_LIBS=opencv_core4,opencv_imgproc4,opencv_objdetect4,opencv_calib3d4,opencv_imgcodecs4");
            }

            // Put DLLs on PATH for runtime
            println!("cargo:rustc-env=PATH={};{}", bin.display(), std::env::var("PATH").unwrap_or_default());

            return;
        }
    }

    // Also check VCPKG_ROOT
    if let Ok(vcpkg_root) = std::env::var("VCPKG_ROOT") {
        let installed = std::path::Path::new(&vcpkg_root).join("installed/x64-windows");
        if installed.exists() {
            println!("cargo:warning=Found OpenCV via vcpkg at {}", installed.display());
            println!("cargo:rustc-env=OPENCV_LINK_PATHS={}/lib", installed.display());
            println!("cargo:rustc-env=OPENCV_LINK_LIBS=opencv_core4,opencv_imgproc4,opencv_objdetect4,opencv_calib3d4,opencv_imgcodecs4");
            println!("cargo:rustc-env=OPENCV_INCLUDE_PATHS={}/include", installed.display());
            return;
        }
    }

    // Linux/macOS: pkg-config
    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        println!("cargo:warning=On Linux/macOS, install OpenCV via your package manager");
        println!("cargo:warning=  Ubuntu: sudo apt install libopencv-dev");
        println!("cargo:warning=  macOS:  brew install opencv");
    } else {
        println!("cargo:warning=OpenCV not found. Install via:");
        println!("cargo:warning=  choco install opencv");
        println!("cargo:warning=  or: vcpkg install opencv[contrib]:x64-windows");
    }
}
