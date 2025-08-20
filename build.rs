#[cfg(target_os = "windows")]
fn main() {
    use std::ffi::OsString;

    let macros = [
        OsString::from(format!("VER_MAJOR={}", env!("CARGO_PKG_VERSION_MAJOR"))),
        OsString::from(format!("VER_MINOR={}", env!("CARGO_PKG_VERSION_MINOR"))),
    ];
    embed_resource::compile("assets/scanmower.rc", macros);
}
