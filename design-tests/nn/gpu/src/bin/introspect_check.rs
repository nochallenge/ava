// TESTING / DESIGN ONLY — introspect_check (F10): is her confidence caused by her state? See src/introspect.rs.
//
// introspect_check --model FILE --val DIR [--ctx 256] [--tokens 8192] [--block K] [--dirs 8]
//                  [--strengths -1,-0.5,-0.25,0.25,0.5,1] [--seed 1] [--workers 3] [--call-tokens 512] [--out DIR]
//
// CPU only. The model needs block options with --confidence (the probe that is checked).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match ava_gpu::introspect::check(&args) {
        Ok(r) => print!("{r}"),
        Err(e) => {
            eprintln!("introspect_check: {e}");
            std::process::exit(1);
        }
    }
}
