// TESTING / DESIGN ONLY.
//
// CATASTROPHIC FORGETTING — the claim that was made and never measured.
//
//   cargo run --release --example forget
//
// v14 reported "4x the data cost 0.27 points on prior knowledge" and called it no
// forgetting. It was not a forgetting test: those were three models each trained FROM
// SCRATCH on 15k/30k/60k exchanges. Nothing was ever learned AFTER anything else.
//
// Forgetting is a claim about ORDER (McCloskey & Cohen 1989): learn A, then learn B
// with no access to A, then ask about A. A network that stores knowledge in shared
// weights loses A because B's learning overwrites them. The design argument for Ava is
// that she allocates fresh units for new facts, so B has nothing of A's to overwrite.
//
// There is a specific way that argument could fail here, and exam.rs already met it
// inside a single run: the allocator is NON-STATIONARY. Its homeostatic `duty` and its
// candidate weights keep moving while it learns, and "by the end of training the same
// query produced a DIFFERENT code and every early memory was unreachable". The fix
// there was to freeze the allocator before writing. Continual learning cannot freeze
// it — learning B means letting it run again — so the drift comes back, and whether
// A survives it is exactly the question.
//
// Three numbers, plus the mechanism:
//   1. A right after learning A               the starting point
//   2. A after then learning B, no replay     forgetting = (1) - (2)
//   3. A when A and B are learned TOGETHER    the interleaved reference
// and, for every A fact that was lost: did its query code change (allocation drift),
// or did it keep its code and get outvoted (readout interference)?
use design_tests::living::Fact;
use design_tests::v6::*;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/unimorph-eng.tsv";
const N_ID: u32 = 10_000_000;
const N_GRAM: u32 = 200_000;
const K_ID: usize = 10;
const PER_TAG: usize = 3;
const SAMPLE: usize = 3000;
const MIN_FREE_GB: f64 = 3.0;

fn free_gb() -> f64 {
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command",
               "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1KB"])
        .output().ok().and_then(|o| String::from_utf8(o.stdout).ok());
    out.and_then(|t| t.trim().parse::<f64>().ok()).map(|b| b / 1e9).unwrap_or(f64::INFINITY)
}

struct Model { fac: Factored, ro: OpReadout }

impl Model {
    fn new() -> Self {
        Model { fac: Factored::new(N_ID, N_GRAM, K_ID, PER_TAG), ro: OpReadout::default() }
    }
    /// exactly exam.rs's procedure: two passes of competition, then a frozen write.
    fn learn(&mut self, facts: &[&Fact]) -> Result<(), String> {
        for _ in 0..2 {
            for (i, f) in facts.iter().enumerate() {
                if i % 100_000 == 0 && i > 0 && free_gb() < MIN_FREE_GB {
                    return Err(format!("stopped: under {MIN_FREE_GB} GB free"));
                }
                self.fac.learn_code(&Fields { lemma: &f.lemma, form: Some(&f.form), tags: &f.tags });
            }
        }
        for f in facts {
            let c = self.code(f);
            self.ro.learn(&c, &edit_op(&f.lemma, &f.form));
        }
        Ok(())
    }
    fn code(&self, f: &Fact) -> Vec<NeuronId> {
        let (idc, gc) = self.fac.query_code(&Fields { lemma: &f.lemma, form: None, tags: &f.tags }, K_ID);
        let mut c = idc;
        c.extend_from_slice(&gc);
        c
    }
    /// The code as a SET. The first version compared codes in order and reported 20.9%
    /// of A's codes "moved" with 100.0% of their units still present: every one of those
    /// was a reordering, because `compete` returns winners sorted by a score that B's
    /// duty updates nudge. Order is irrelevant to readout; membership is what matters.
    fn code_set(&self, f: &Fact) -> Vec<NeuronId> {
        let mut c = self.code(f);
        c.sort_unstable();
        c.dedup();
        c
    }
    /// exam.rs's `ask(.., 1).first()`: top operation, applied to the lemma.
    fn right(&self, f: &Fact) -> bool {
        self.ro.decode(&self.code(f), 1).into_iter().next()
            .and_then(|(op, _)| apply_op(&f.lemma, &op))
            .as_deref() == Some(f.form.as_str())
    }
    fn acc(&self, s: &[&Fact]) -> f64 {
        s.iter().filter(|f| self.right(f)).count() as f64 / s.len().max(1) as f64 * 100.0
    }
}

fn sample<'a>(v: &[&'a Fact]) -> Vec<&'a Fact> {
    v.iter().step_by((v.len() / SAMPLE).max(1)).take(SAMPLE).copied().collect()
}

fn main() -> std::io::Result<()> {
    let raw = fs::read_to_string(CORPUS)?;
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).collect();
    drop(raw);

    // Split by LEMMA, deterministically, so A and B share no word: every A fact's
    // identity key is one B never touches. Whatever A loses, it loses through the shared
    // machinery — grammar units, the readout, the allocator's duty — not because B
    // relearned the same word. Bit 33 of the hash, not bit 0: FNV-1a's low bit never
    // mixes (v12), and `h & 1` here would split by the parity of the spelling.
    let in_a = |f: &Fact| (fnv1a(f.lemma.as_bytes(), 0xF06E7) >> 33) & 1 == 0;
    let a: Vec<&Fact> = all.iter().filter(|f| in_a(f)).collect();
    let b: Vec<&Fact> = all.iter().filter(|f| !in_a(f)).collect();
    let (sa, sb) = (sample(&a), sample(&b));

    // THE CONFOUND IN THE FIRST RUN. B scored 99.20% before she had seen a single B
    // word, because most English inflection is regular and the GRAMMAR route derives it
    // from the tags alone. So "A survived" could not distinguish remembering A from
    // re-deriving A. Forgetting is only observable on facts that REQUIRE memory.
    //
    // Those are the irregulars, defined exactly as exam.rs defines them: a fact whose
    // edit operation is not the most common operation for its tag-set, so grammar alone
    // gets it wrong. The modal op is computed over the whole corpus; it is a property of
    // English, not of either training set.
    let tagkey = |f: &Fact| {
        let mut t: Vec<&str> = f.tags.iter().map(|s| s.as_str()).collect();
        t.sort();
        t.join(",")
    };
    let mut modal: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for f in &all {
        *modal.entry(tagkey(f)).or_default().entry(edit_op(&f.lemma, &f.form)).or_insert(0) += 1;
    }
    let common: HashMap<String, String> = modal.iter()
        .filter_map(|(k, m)| m.iter()
            .max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(x.0)))
            .map(|(op, _)| (k.clone(), op.clone())))
        .collect();
    let irregular = |f: &Fact| common.get(&tagkey(f)) != Some(&edit_op(&f.lemma, &f.form));
    let ai: Vec<&Fact> = a.iter().copied().filter(|f| irregular(f)).collect();
    let bi: Vec<&Fact> = b.iter().copied().filter(|f| irregular(f)).collect();
    let (sai, sbi) = (sample(&ai), sample(&bi));
    println!("   irregular (grammar alone gets them wrong): {} in A, {} in B", ai.len(), bi.len());
    println!();
    println!("== forgetting: {} facts in A, {} in B, disjoint lemmas, {} sampled each ==\n",
        a.len(), b.len(), sa.len());

    // ---- SEQUENTIAL: A, then B with no replay of A.
    let t0 = Instant::now();
    let mut seq = Model::new();
    if let Err(e) = seq.learn(&a) { println!("   !! {e}"); return Ok(()); }
    let a1 = seq.acc(&sa);
    let b1 = seq.acc(&sb);
    let ai1 = seq.acc(&sai);
    let bi1 = seq.acc(&sbi);
    let irr_before: Vec<bool> = sai.iter().map(|f| seq.right(f)).collect();
    let codes_before: Vec<Vec<NeuronId>> = sa.iter().map(|f| seq.code_set(f)).collect();
    let right_before: Vec<bool> = sa.iter().map(|f| seq.right(f)).collect();
    println!("   learned A                       {:>7.2?}", t0.elapsed());
    println!("     A  {:>6.2}%   (facts she was just taught)", a1);
    println!("     B  {:>6.2}%   (never seen: these words are new to her)", b1);
    println!("     A irregulars  {:>6.2}%   <- the facts that NEED memory", ai1);
    println!("     B irregulars  {:>6.2}%   (never seen: what grammar alone gets on irregulars)", bi1);

    if let Err(e) = seq.learn(&b) { println!("   !! {e}"); return Ok(()); }
    let a2 = seq.acc(&sa);
    let b2 = seq.acc(&sb);
    let ai2 = seq.acc(&sai);
    let bi2 = seq.acc(&sbi);
    let irr_lost = sai.iter().zip(&irr_before).filter(|(f, &was)| was && !seq.right(f)).count();
    println!("   then learned B, no replay of A  {:>7.2?}", t0.elapsed());
    println!("     A  {:>6.2}%   <- forgetting is the drop from {:.2}%", a2, a1);
    println!("     B  {:>6.2}%", b2);
    println!("     A irregulars  {:>6.2}%   <- forgetting where it can actually show: from {:.2}%", ai2, ai1);
    println!("     B irregulars  {:>6.2}%", bi2);
    println!("     A irregulars right before B and wrong after: {irr_lost} of {}", sai.len());

    // ---- MECHANISM. For each A fact that was right before B and wrong after it:
    // did its code move, or did it keep its code and get outvoted?
    let (mut lost, mut gained, mut drift, mut lost_drift) = (0usize, 0usize, 0usize, 0usize);
    let mut overlap_sum = 0.0f64;
    for (i, f) in sa.iter().enumerate() {
        let now = seq.code_set(f);
        let moved = now != codes_before[i];
        if moved {
            drift += 1;
            let kept = now.iter().filter(|u| codes_before[i].contains(u)).count();
            overlap_sum += kept as f64 / codes_before[i].len().max(1) as f64;
        }
        let ok = seq.right(f);
        if right_before[i] && !ok { lost += 1; if moved { lost_drift += 1; } }
        if !right_before[i] && ok { gained += 1; }
    }
    println!("\n   mechanism, over the {} sampled A facts:", sa.len());
    println!("     codes that moved while B was learned   {:>5}  ({:.1}%)",
        drift, drift as f64 / sa.len() as f64 * 100.0);
    if drift > 0 {
        println!("       of a moved code, units still shared  {:.1}% on average",
            overlap_sum / drift as f64 * 100.0);
    }
    println!("     right before B, wrong after            {:>5}  ({} of them had moved codes)",
        lost, lost_drift);
    println!("     wrong before B, right after            {:>5}", gained);
    drop(seq);

    // ---- INTERLEAVED reference: the same facts, learned together.
    let t1 = Instant::now();
    let mut joint = Model::new();
    let both: Vec<&Fact> = all.iter().collect();
    if let Err(e) = joint.learn(&both) { println!("   !! {e}"); return Ok(()); }
    let aj = joint.acc(&sa);
    let bj = joint.acc(&sb);
    let aij = joint.acc(&sai);
    let bij = joint.acc(&sbi);
    println!("\n   interleaved, A and B together    {:>7.2?}", t1.elapsed());
    println!("     A  {:>6.2}%", aj);
    println!("     B  {:>6.2}%", bj);
    println!("     A irregulars  {:>6.2}%", aij);
    println!("     B irregulars  {:>6.2}%", bij);
    println!();
    println!("   forgetting on irregulars                {:+.2} points", ai2 - ai1);

    println!("\n   forgetting (A before B -> after B)     {:+.2} points", a2 - a1);
    println!("   sequential vs interleaved, on A        {:+.2} points", a2 - aj);
    println!("\nTSV\tforget\t{}\t{}\t{a1:.2}\t{a2:.2}\t{aj:.2}\t{ai1:.2}\t{ai2:.2}\t{aij:.2}\t{bi1:.2}\t{drift}\t{lost}\t{irr_lost}",
        a.len(), b.len());
    Ok(())
}
