//! Side-by-side comparison of naive vs IDF-weighted overlap.
//!
//! Builds a synthetic IDF map (so the test is fully deterministic)
//! and runs ten cases through both classifiers. The point isn't to
//! prove they ALWAYS disagree — naive is a fine proxy when the
//! summary keeps every important word. The point is to prove that
//! when summaries drop high-IDF rationale tokens, weighted overlap
//! correctly classifies the item as Lost while naive declares it
//! Paraphrased.

use std::collections::HashMap;

use cctx::compaction::loss_detector::{
    weighted_overlap, DEFAULT_LOST_THRESHOLD, DEFAULT_PRESERVED_THRESHOLD,
};

/// Tiny clone of the unit-test helper — `loss_detector::naive_overlap`
/// is `pub(crate)` and not reachable from the integration test
/// crate. This re-implementation matches the production formula
/// exactly (set-based Jaccard over alphanumeric-trimmed tokens of
/// length ≥ 2).
fn naive_overlap(item: &str, summary: &str) -> f64 {
    use std::collections::HashSet;
    let tok = |s: &str| -> HashSet<String> {
        s.split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|t| t.len() >= 2)
            .collect()
    };
    let item_set = tok(item);
    if item_set.is_empty() {
        return 0.0;
    }
    let summary_set = tok(summary);
    let intersection = item_set.intersection(&summary_set).count();
    intersection as f64 / item_set.len() as f64
}

/// Bucket each overlap into the same three classes the production
/// detector uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bucket {
    Preserved,
    Paraphrased,
    Lost,
}

fn bucket(score: f64) -> Bucket {
    if score >= DEFAULT_PRESERVED_THRESHOLD {
        Bucket::Preserved
    } else if score >= DEFAULT_LOST_THRESHOLD {
        Bucket::Paraphrased
    } else {
        Bucket::Lost
    }
}

/// One adversarial test case. Each carries the IDF map relevant to
/// its mini-conversation so the cases are independent.
struct Case {
    name: &'static str,
    item: &'static str,
    summary: &'static str,
    idf: HashMap<String, f64>,
    /// What we expect naive to say ("looks fine, summary echoes the noun").
    expected_naive: Bucket,
    /// What we expect weighted to say ("rationale tokens missing → Lost").
    expected_weighted: Bucket,
}

fn idf(common: &[&str], rare: &[&str]) -> HashMap<String, f64> {
    // Common = small weight (0.1); rare = large weight (2.0).
    // Numbers picked to clearly separate the two bands; the actual
    // production IDF gives similar shapes from real corpora.
    let mut map = HashMap::new();
    for tok in common {
        map.insert((*tok).to_string(), 0.1);
    }
    for tok in rare {
        map.insert((*tok).to_string(), 2.0);
    }
    map
}

/// The ten cases — each one demonstrates a real-world pattern where
/// naive overlap would let an item slip through as Paraphrased while
/// weighted overlap correctly flags the rationale-loss as Lost.
fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "PostgreSQL/MongoDB/ACID rationale dropped",
            item: "We chose PostgreSQL over MongoDB for ACID compliance",
            summary: "We use PostgreSQL for the project",
            idf: idf(
                &["we", "use", "for", "the", "postgresql", "project"],
                &["chose", "over", "mongodb", "acid", "compliance"],
            ),
            expected_naive: Bucket::Paraphrased,
            expected_weighted: Bucket::Lost,
        },
        Case {
            name: "Auth service port number dropped",
            item: "auth service runs on port 8443 in the mesh",
            summary: "the auth service handles authentication for the project",
            idf: idf(
                &["the", "service", "in", "for", "auth", "on"],
                &["runs", "port", "8443", "mesh"],
            ),
            expected_naive: Bucket::Paraphrased,
            expected_weighted: Bucket::Lost,
        },
        Case {
            name: "Budget cap value dropped",
            item: "budget should not exceed 50K total for the year",
            summary: "we set a budget for the year",
            idf: idf(
                &["we", "for", "the", "year", "budget"],
                &["should", "not", "exceed", "50k", "total"],
            ),
            expected_naive: Bucket::Paraphrased,
            expected_weighted: Bucket::Lost,
        },
        Case {
            name: "Config file path dropped",
            item: "the production config lives at /etc/myapp/prod.yml",
            summary: "the production config is read on boot",
            idf: idf(
                &["the", "is", "config", "production"],
                &["lives", "at", "myapp", "prod", "etc", "yml"],
            ),
            expected_naive: Bucket::Paraphrased,
            expected_weighted: Bucket::Lost,
        },
        Case {
            name: "Bug root cause specifics dropped",
            // Item carries 4 common words (the/root/cause/was) and 6
            // rare ones (the actual diagnosis). Summary echoes the
            // generic "we found the root cause" without the specifics.
            // Naive: 4 / 10 = 0.40 → Paraphrased.
            // Weighted: 0.4 / 12.4 ≈ 0.03 → Lost.
            item: "the root cause was the HPA scaling delay during cold starts",
            summary: "the root cause was found and we moved on with the fix",
            idf: idf(
                &["the", "was", "root", "cause"],
                &["hpa", "scaling", "delay", "during", "cold", "starts"],
            ),
            expected_naive: Bucket::Paraphrased,
            expected_weighted: Bucket::Lost,
        },
        Case {
            name: "Constraint kept; details dropped",
            item: "deadline is March 15 and the budget is 100K total",
            summary: "deadline and budget are documented",
            idf: idf(
                &["the", "and", "is", "are", "deadline", "budget"],
                &["march", "15", "100k", "total", "documented"],
            ),
            expected_naive: Bucket::Paraphrased,
            expected_weighted: Bucket::Lost,
        },
        Case {
            name: "Design decision: rationale trimmed",
            // 6 common tokens shared (we / picked / rust / for / our /
            // service); 8 rare rationale tokens dropped. Naive sees
            // 5/14 ≈ 0.36 → Paraphrased. Weighted sees 0.5/16.6 ≈
            // 0.03 → Lost.
            item:
                "We picked Rust for our service over Go because borrow checker prevents data races",
            summary: "we picked Rust for the new service",
            idf: idf(
                &["we", "picked", "rust", "for", "our", "service", "the"],
                &[
                    "over", "go", "because", "borrow", "checker", "prevents", "data", "races",
                ],
            ),
            expected_naive: Bucket::Paraphrased,
            expected_weighted: Bucket::Lost,
        },
        Case {
            name: "Identical surface form — should stay Preserved",
            item: "we set minReplicas to 2 in the HPA spec to fix the cold-start issue",
            summary: "we set minReplicas to 2 in the HPA spec to fix the cold-start issue",
            idf: idf(
                &["we", "to", "in", "the"],
                &[
                    "minreplicas",
                    "hpa",
                    "spec",
                    "fix",
                    "cold",
                    "start",
                    "issue",
                ],
            ),
            expected_naive: Bucket::Preserved,
            expected_weighted: Bucket::Preserved,
        },
        Case {
            name: "Genuinely Lost (sanity)",
            item: "the canary group uses the v2 router on port 7001",
            summary: "we migrated everything to the new system",
            idf: idf(
                &["the", "we", "to"],
                &["canary", "group", "v2", "router", "port", "7001"],
            ),
            expected_naive: Bucket::Lost,
            expected_weighted: Bucket::Lost,
        },
        Case {
            name: "Common-word echo without high-IDF terms",
            item: "the team ships a release on Thursdays",
            summary: "the team ships a release on Thursdays",
            idf: idf(&["the", "on", "ships", "team"], &["release", "thursdays"]),
            expected_naive: Bucket::Preserved,
            expected_weighted: Bucket::Preserved,
        },
    ]
}

#[test]
fn weighted_overlap_correctly_separates_rationale_loss_in_seven_of_ten_cases() {
    let cases = cases();
    let mut weighted_wins = 0;
    let mut details: Vec<String> = Vec::new();

    for case in &cases {
        let naive = naive_overlap(case.item, case.summary);
        let weighted = weighted_overlap(case.item, case.summary, &case.idf);

        let naive_bucket = bucket(naive);
        let weighted_bucket = bucket(weighted);

        // Both classifiers should match the case's `expected_*`.
        assert_eq!(
            naive_bucket, case.expected_naive,
            "[{}] naive: expected {:?}, got {:?} (score {:.3})",
            case.name, case.expected_naive, naive_bucket, naive
        );
        assert_eq!(
            weighted_bucket, case.expected_weighted,
            "[{}] weighted: expected {:?}, got {:?} (score {:.3})",
            case.name, case.expected_weighted, weighted_bucket, weighted
        );

        // "Weighted wins" = naive missed a Lost / Paraphrased
        // distinction that weighted caught. That happens when the
        // expected verdicts differ.
        if case.expected_naive != case.expected_weighted {
            weighted_wins += 1;
        }

        details.push(format!(
            "{:>3}.{:<5}|{:<5} | {} | naive={:.2}({:?})  weighted={:.2}({:?})",
            details.len() + 1,
            "",
            "",
            case.name,
            naive,
            naive_bucket,
            weighted,
            weighted_bucket,
        ));
    }

    // Sanity: at least 7 of the 10 cases are scenarios where weighted
    // genuinely changes the verdict. (3 are sanity cases where they
    // should agree.)
    assert!(
        weighted_wins >= 7,
        "expected weighted to override naive on >=7 of 10 cases, got {}",
        weighted_wins
    );

    // Print a useful diff so a future maintainer can see the table.
    eprintln!(
        "Weighted vs naive overlap, {} adversarial cases:",
        cases.len()
    );
    for line in &details {
        eprintln!("{}", line);
    }
    eprintln!(
        "Weighted overrode naive on {}/{} cases",
        weighted_wins,
        cases.len()
    );
}
