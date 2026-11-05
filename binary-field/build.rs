fn main() {
    // The GPU grinder is linked only with the `gpu-grind` feature.
    if std::env::var_os("CARGO_FEATURE_GPU_GRIND").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=ZKM_GRIND_LIB_DIR");
    let dir = std::env::var("ZKM_GRIND_LIB_DIR")
        .expect("the gpu-grind feature needs ZKM_GRIND_LIB_DIR, the directory of libzkmgrind.so");
    println!("cargo:rustc-link-search=native={dir}");
    println!("cargo:rustc-link-lib=dylib=zkmgrind");
}
