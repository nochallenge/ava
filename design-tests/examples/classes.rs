// TESTING / DESIGN ONLY.
// What does she think is similar to what, once similarity is context and not spelling?
use design_tests::seq;
use design_tests::wordclass::WordClasses;
use std::fs;

fn main() -> std::io::Result<()> {
    let raw = fs::read_to_string("../data/replies.tsv")?;
    let mut docs: Vec<Vec<String>> = Vec::new();
    for line in raw.lines().take(120_000) {
        if let Some((q, r)) = line.split_once('\t') {
            docs.push(seq::tokenize(q));
            docs.push(seq::tokenize(r));
        }
    }
    let refs: Vec<&[String]> = docs.iter().map(|d| d.as_slice()).collect();
    let wc = WordClasses::build(refs.into_iter(), 2, 20);
    println!("profiled {} words from {} sentences\n", wc.vocab, docs.len());

    // who shares a class with these? that is the whole question.
    for probe in ["dog", "happy", "monday", "eat", "she", "hospital", "quickly"] {
        let Some(k) = wc.class.get(probe) else { println!("{probe}: too rare"); continue };
        let mut same: Vec<&str> = wc.class.iter()
            .filter(|(w, l)| w.as_str() != probe && l[0] == k[0] && l[1] == k[1])
            .map(|(w, _)| w.as_str()).collect();
        same.sort();
        same.truncate(12);
        println!("{:<10} -> {}", probe, if same.is_empty() { "(alone)".into() } else { same.join(" ") });
    }

    // Is the problem my HASHING, or the profiles themselves? Compute exact cosine
    // neighbours over the same PMI vectors. If these are good the LSH is too coarse;
    // if these are bad too, the corpus cannot support distributional similarity and
    // no amount of clever hashing will rescue it.
    println!("
--- exact cosine neighbours over the same profiles ---");
    let (prof, freq, total) = design_tests::wordclass::profiles(
        docs.iter().map(|d| d.as_slice()), 2, 20);
    for probe in ["dog", "happy", "monday", "eat", "she", "hospital", "quickly"] {
        let Some(pv) = prof.get(probe) else { println!("{probe}: too rare"); continue };
        let mut best: Vec<(f32, &str)> = prof.iter()
            .filter(|(w, _)| w.as_str() != probe)
            .map(|(w, v)| (design_tests::wordclass::cosine(pv, v, &freq, total), w.as_str()))
            .collect();
        best.sort_by(|a, b| b.0.total_cmp(&a.0));
        best.truncate(6);
        println!("{:<10} -> {}", probe,
            best.iter().map(|(s, w)| format!("{w} {s:.2}")).collect::<Vec<_>>().join("  "));
    }

    let sizes = wc.band0_sizes();
    let used = sizes.iter().filter(|&&n| n > 0).count();
    let biggest = sizes.iter().max().unwrap();
    println!("\nband 0: {used} classes in use, biggest holds {biggest} words");
    Ok(())
}
