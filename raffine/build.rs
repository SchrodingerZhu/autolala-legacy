use std::env;
use std::error::Error;
use std::path::Path;
use std::process::Command;

const LLVM_MAJOR_VERSION: usize = 22;

fn llvm_config(argument: &str) -> Result<String, Box<dyn Error>> {
    let prefix = env::var(format!("MLIR_SYS_{LLVM_MAJOR_VERSION}0_PREFIX"))
        .map(|path| Path::new(&path).join("bin"))
        .unwrap_or_default();
    let llvm_config_exe = if cfg!(target_os = "windows") {
        "llvm-config.exe"
    } else {
        "llvm-config"
    };

    let call = format!(
        "{} --link-static {argument}",
        prefix.join(llvm_config_exe).display(),
    );

    Ok(str::from_utf8(
        &if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", &call]).output()?
        } else {
            Command::new("sh").arg("-c").arg(&call).output()?
        }
        .stdout,
    )?
    .trim()
    .to_string())
}

fn main() -> Result<(), Box<dyn Error>> {
    let includes = llvm_config("--includedir")?
        .split_whitespace()
        .map(String::from)
        .collect::<Vec<_>>();
    let flags = llvm_config("--cxxflags")?
        .split_whitespace()
        .filter(|x| !x.starts_with("-I") && !x.contains("no-exceptions"))
        .map(String::from)
        .collect::<Vec<_>>();
    let cxx_sources = Path::new("src/cxx")
        .read_dir()?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if !entry.file_type().ok()?.is_file() {
                return None;
            }
            let path = entry.path();
            if path.extension()?.to_str()? == "cpp" {
                Some(path)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut build = cxx_build::bridge("src/cxx.rs");
    for flag in flags {
        build.flag_if_supported(flag);
    }
    let cxx_source_dir = Path::new("src/cxx").canonicalize()?;
    build
        .files(cxx_sources)
        .includes(includes)
        .include(cxx_source_dir)
        .cpp(true)
        .std("gnu++20")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("raffine-cxx");
    println!("cargo:rerun-if-changed=src/cxx");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MLIR_SYS_{LLVM_MAJOR_VERSION}0_PREFIX");
    Ok(())
}
