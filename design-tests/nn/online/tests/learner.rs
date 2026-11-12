// TESTING / DESIGN ONLY
//! The whole learner loop, run as a program: import, learn from a diary (dry run), a gate that
//! blocks, a gate that passes, erasure that deletes the brains that learned and rolls back.

use std::path::Path;
use std::process::Command;

fn learner(args: &[&str]) -> (bool, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_ava_learner")).args(args).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    (o.status.success(), text)
}

fn diary_day(path: &Path) {
    let mut s = String::from("time\tmodel\tturn\tuser\tava\n");
    for k in 1..=40 {
        s.push_str(&format!("2026-10-27T10:{:02}:00\tv41/m\t{k}\tWhat is {k} plus 1?\t{}\n", k, k + 1));
        if k % 5 == 0 {
            s.push_str(&format!("2026-10-27T10:{:02}:30\tv41/m\t{k}\t/correct\t{}\n", k, k + 1));
        }
    }
    std::fs::write(path, s).unwrap();
}

#[test]
fn learns_gates_and_erases() {
    let d = std::env::temp_dir().join(format!("ava_learner_it_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("diary")).unwrap();
    let (root, diary, brain) = (d.join("learner"), d.join("diary"), d.join("brain.safetensors"));
    ava_online::safet::write_dummy(&brain, &[("emb", &[8, 4])]).unwrap();
    diary_day(&diary.join("2026-10-27.tsv"));
    std::fs::write(diary.join("2026-10-27.gen.tsv"), "not a day file\n").unwrap();
    let (r, d_, b) = (root.to_str().unwrap(), diary.to_str().unwrap(), brain.to_str().unwrap());

    let (ok, out) = learner(&["--root", r, "import", b]);
    assert!(ok && out.contains("brain 1 is live"), "{out}");

    // Dry run: no training command, no gates.
    let (ok, out) = learner(&["--root", r, "run", "--once", "--no-gates", "--batch", "16", "--diary", d_]);
    assert!(ok && out.contains("+88 episodes") && out.contains("brain 2 is live (rehearsed 16"), "{out}");

    // Without a gate command and without --no-gates, nothing is promoted.
    let (ok, out) = learner(&["--root", r, "run", "--once", "--batch", "16", "--diary", d_]);
    assert!(ok && out.contains("brain 3 rejected: no gate command"), "{out}");

    // A gate that sees 20% of the facts lost blocks the swap.
    let bin = env!("CARGO_BIN_EXE_ava_learner");
    let drop = format!("\"{bin}\" fake-gate {{out}} drop");
    let (ok, out) = learner(&["--root", r, "run", "--once", "--batch", "16", "--diary", d_, "--gate-cmd", &drop]);
    assert!(ok && out.contains("brain 4 rejected: skills dropped: facts"), "{out}");

    let same = format!("\"{bin}\" fake-gate {{out}} same");
    let (ok, out) = learner(&["--root", r, "run", "--once", "--batch", "16", "--diary", d_, "--gate-cmd", &same]);
    assert!(ok && out.contains("brain 5 is live"), "{out}");

    let (ok, out) = learner(&["--root", r, "status"]);
    assert!(ok && out.contains("live brain: 5 (lineage [1, 2, 5])") && out.contains("self-questions anchored: 1"), "{out}");
    assert!(out.contains("facts: 100% now"), "{out}");

    // Erasing the person deletes every brain that learned from them and goes back to brain 1.
    let (ok, out) = learner(&["--root", r, "erase-person", "owner"]);
    assert!(ok && out.contains("erased 88 episodes") && out.contains("live brain: Some(1)"), "{out}");
    let (ok, out) = learner(&["--root", r, "status"]);
    assert!(ok && out.contains("live brain: 1") && out.contains("episodes: 0"), "{out}");
    let raw = std::fs::read(root.join("episodes.log")).unwrap();
    assert!(!raw.windows(13).any(|w| w == b"What is 7 plu"));
    assert!(!root.join("ckpt/000002.safetensors").exists());

    // Lessons (J8): shared, imported once, then quizzed (F14) and in her life story (F11).
    let lessons = d.join("lessons_001.tsv");
    std::fs::write(&lessons, "question\tanswer\nWhat is 7 times 8?\t56.\nbroken line\nWhy do we have day and night?\tThe Earth spins.\n").unwrap();
    let l = lessons.to_str().unwrap();
    let (ok, out) = learner(&["--root", r, "teach", l]);
    assert!(ok && out.contains("taught 2 new lessons (1 bad lines), shared"), "{out}");
    let (ok, out) = learner(&["--root", r, "teach", l]);
    assert!(ok && out.contains("taught 0 new lessons"), "{out}");
    let (ok, out) = learner(&["--root", r, "--person", "kim", "quiz", "5"]);
    assert!(ok && out.contains("lesson/lessons_001#2") && out.contains("What is 7 times 8?\t56."), "{out}");
    let (ok, out) = learner(&["--root", r, "life"]);
    assert!(ok && out.contains("been taught 2"), "{out}");
    let (ok, out) = learner(&["--root", r, "person", "owner"]);
    assert!(ok && out.contains("I have not talked with owner before."), "{out}");

    // Standing goals (F5): added, cued by what is said, kept per person, finished.
    let (ok, out) = learner(&["--root", r, "goal", "add", "sam", "ask how the exam went", "exam test"]);
    assert!(ok && out.contains("goal g1 added"), "{out}");
    let (ok, _) = learner(&["--root", r, "goal", "add", "-", "learn to count past 100", "count", "2099-01-01", "0.6"]);
    assert!(ok);
    let (ok, _) = learner(&["--root", r, "goal", "add", "kim", "send the recipe", "recipe"]);
    assert!(ok);
    let (ok, out) = learner(&["--root", r, "--person", "sam", "goals", "my", "exam", "is", "tomorrow"]);
    let rows: Vec<&str> = out.lines().skip(1).collect();
    assert!(ok && rows.len() == 2 && rows[0].starts_with("g1\t") && rows[1].contains("\ther\t"), "{out}");
    let (ok, _) = learner(&["--root", r, "goal", "done", "g1"]);
    assert!(ok);
    let (ok, out) = learner(&["--root", r, "--person", "sam", "goals"]);
    assert!(ok && !out.contains("g1\t") && out.contains("g2\t"), "{out}");
    let (ok, out) = learner(&["--root", r, "goal", "done", "g9"]);
    assert!(!ok && out.contains("no goal g9"), "{out}");

    // Care (J9): someone she has never talked with leans on her not at all.
    let (ok, out) = learner(&["--root", r, "care", "nobody"]);
    assert!(ok && out.contains("reliance 0.00") && out.contains("turns outward 0.02"), "{out}");

    // L1 surprise: the unmeasured lessons are measured once before rehearsal, then never again.
    let sur = format!("\"{bin}\" fake-surprise {{ids}} {{out}}");
    let (ok, out) = learner(&["--root", r, "run", "--once", "--no-gates", "--batch", "16", "--diary", d_, "--surprise-cmd", &sur]);
    assert!(ok && out.contains("(surprise measured for 2 episodes)"), "{out}");
    let (ok, out) = learner(&["--root", r, "run", "--once", "--no-gates", "--batch", "16", "--diary", d_, "--surprise-cmd", &sur]);
    assert!(ok && !out.contains("surprise measured"), "{out}");
    let raw = std::fs::read(root.join("episodes.log")).unwrap();
    assert_eq!(raw.windows(9).filter(|w| w == b"surprise\t").count(), 2, "one surprise record per lesson");
}
