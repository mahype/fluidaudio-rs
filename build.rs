use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell Cargo to rerun if Swift files change
    println!("cargo:rerun-if-changed=swift/");
    println!("cargo:rerun-if-changed=Package.swift");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    // Build the Swift package first to get FluidAudio dependency
    println!("cargo:warning=Building Swift package...");

    let swift_build_dir = out_dir.join("swift-build");
    std::fs::create_dir_all(&swift_build_dir).expect("Failed to create swift-build directory");

    // Build Swift package in release mode
    let status = Command::new("swift")
        .args(&[
            "build",
            "-c",
            "release",
            "--build-path",
            swift_build_dir.to_str().unwrap(),
        ])
        .current_dir(&manifest_dir)
        .status()
        .expect("Failed to run swift build");

    if !status.success() {
        panic!("Swift package build failed");
    }

    // Find the built library
    let lib_path = swift_build_dir.join("release");

    // Link the Swift library
    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:rustc-link-lib=static=FluidAudioBridge");

    // FluidAudio >= 0.15 depends on NemoTextProcessing, shipped as a binary
    // xcframework holding a universal static library. SwiftPM links it into
    // Swift products, but our static FluidAudioBridge only records the
    // references, so the final Rust link has to pull it in explicitly.
    let nemo_lib = swift_build_dir.join(
        "artifacts/fluidaudio/NemoTextProcessing/NemoTextProcessing.xcframework/macos-arm64_x86_64/libtext_processing_rs.a",
    );
    if nemo_lib.exists() {
        let local_dir = out_dir.join("nemo-local");
        localize_nemo(&nemo_lib, &local_dir);
        println!("cargo:rustc-link-search=native={}", local_dir.display());
        println!("cargo:rustc-link-lib=static=nemo_text_processing");
    }

    // Link Apple frameworks
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=AVFoundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // Link Swift runtime
    println!("cargo:rustc-link-lib=dylib=swiftCore");

    // Link C++ standard library (needed for FastClusterWrapper.cpp in FluidAudio)
    println!("cargo:rustc-link-lib=c++");
}

/// `libtext_processing_rs.a` is itself a Rust static library with its own copy
/// of the Rust runtime. Linked next to another Rust static library (as in any
/// app that embeds this crate through a `staticlib`), unmangled runtime symbols
/// such as `_rust_eh_personality` collide. Pre-link it into one relocatable
/// object for the target architecture that exports only the `nemo_*` C API
/// and keeps everything else private, then wrap that in a static library.
fn localize_nemo(nemo_lib: &std::path::Path, out: &std::path::Path) {
    std::fs::create_dir_all(out).expect("create nemo-local directory");
    let arch = match std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() {
        Ok("aarch64") => "arm64",
        Ok("x86_64") => "x86_64",
        Ok(other) => panic!("unsupported target architecture for NemoTextProcessing: {other}"),
        Err(_) => panic!("CARGO_CFG_TARGET_ARCH is not set"),
    };
    let thin = out.join("libtext_processing_rs-thin.a");
    run(Command::new("lipo")
        .arg(nemo_lib)
        .args(["-thin", arch, "-output"])
        .arg(&thin));
    let exports = out.join("nemo-exports.txt");
    std::fs::write(&exports, "_nemo_*\n").expect("write nemo export list");
    let object = out.join("nemo_text_processing.o");
    // Go through clang so the linker gets the platform version it requires.
    let min_macos = std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".into());
    run(Command::new("xcrun")
        .args(["clang", "-r", "-nostdlib", "-arch", arch])
        .arg(format!("-mmacosx-version-min={min_macos}"))
        .arg("-Wl,-all_load")
        .arg(format!("-Wl,-exported_symbols_list,{}", exports.display()))
        .arg(&thin)
        .arg("-o")
        .arg(&object));
    let archive = out.join("libnemo_text_processing.a");
    let _ = std::fs::remove_file(&archive);
    run(Command::new("libtool")
        .args(["-static", "-o"])
        .arg(&archive)
        .arg(&object));
}

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|err| panic!("failed to run {command:?}: {err}"));
    assert!(status.success(), "{command:?} failed with {status}");
}
