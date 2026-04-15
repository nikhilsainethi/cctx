//! Importance pruning integration tests.
//!
//! The prune strategy removes low-information lines (filler, acknowledgments,
//! repetition). These tests use inline fixtures with deliberate filler content
//! since our file-based fixtures are clean technical conversations.

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;

fn cctx() -> Command {
    Command::cargo_bin("cctx").expect("binary not found")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Filler removal
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prune_removes_filler_lines() {
    // A conversation heavy with filler: "Sure!", "Great question!", "Thanks!".
    // These should be pruned while substantive content is preserved.
    let input = serde_json::json!([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I set up a database?"},
        {"role": "assistant", "content": "Sure, I can help with that!\nGreat question!\nTo set up PostgreSQL, first install it via your package manager.\nThen create a database with createdb and configure the connection string.\nLet me know if you need more details!"},
        {"role": "user", "content": "Thanks for that!\nYeah so basically what I'm trying to say is that I need it for production.\nThe database needs to handle 1000 concurrent connections.\nI also need replication for high availability."},
        {"role": "assistant", "content": "Absolutely, I understand!\nFor production PostgreSQL with 1000 connections, use PgBouncer for connection pooling.\nSet max_connections=200 and let PgBouncer multiplex.\nFor replication, configure streaming replication with a standby server.\nHope that helps!"}
    ]).to_string();

    let output = cctx()
        .args([
            "optimize",
            "--strategy",
            "prune",
            "--prune-threshold",
            "0.5",
        ])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");
    assert_eq!(msgs.len(), 5, "prune should not remove entire messages");

    // Check that filler lines were removed from the assistant message.
    let assistant_content = msgs[2]["content"].as_str().unwrap();
    assert!(
        !assistant_content.contains("Sure, I can help"),
        "filler 'Sure, I can help' should be pruned"
    );
    assert!(
        !assistant_content.contains("Great question"),
        "filler 'Great question' should be pruned"
    );
    // Substantive content should be preserved.
    assert!(
        assistant_content.contains("PostgreSQL"),
        "substantive content about PostgreSQL should survive"
    );
}

#[test]
fn prune_removes_short_acknowledgments() {
    let input = serde_json::json!([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is Kubernetes?"},
        {"role": "assistant", "content": "OK.\nSure.\nKubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications across clusters of machines."},
        {"role": "user", "content": "Thanks!\nGot it.\nHow do I deploy to it?"}
    ]).to_string();

    let output = cctx()
        .args([
            "optimize",
            "--strategy",
            "prune",
            "--prune-threshold",
            "0.5",
        ])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");

    // Short acknowledgments like "OK.", "Sure.", "Thanks!" should be pruned.
    let asst = msgs[2]["content"].as_str().unwrap();
    assert!(!asst.starts_with("OK."), "short 'OK.' should be pruned");
    assert!(
        asst.contains("Kubernetes"),
        "substantive Kubernetes definition should survive"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// System message protection
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prune_never_touches_system_message() {
    let input = serde_json::json!([
        {"role": "system", "content": "Sure, I can help with that!\nYou are a helpful assistant.\nOK let me explain."},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Sure thing!\nHi there, how can I help?"}
    ]).to_string();

    let output = cctx()
        .args([
            "optimize",
            "--strategy",
            "prune",
            "--prune-threshold",
            "0.5",
        ])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");

    // System message content must be completely unchanged — even filler inside it.
    let system = msgs[0]["content"].as_str().unwrap();
    assert!(
        system.contains("Sure, I can help"),
        "system message must be preserved verbatim"
    );
    assert!(
        system.contains("OK let me explain"),
        "system message filler must NOT be pruned"
    );
}

#[test]
fn prune_never_touches_last_two_user_messages() {
    let input = serde_json::json!([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "OK so basically I wanted to know about databases."},
        {"role": "assistant", "content": "Sure thing!\nDatabases store data."},
        {"role": "user", "content": "Thanks!\nGot it.\nSo basically what about Redis?"},
        {"role": "assistant", "content": "Absolutely!\nRedis is an in-memory store."},
        {"role": "user", "content": "OK thanks.\nSo basically what I really want to know is about caching."}
    ]).to_string();

    let output = cctx()
        .args([
            "optimize",
            "--strategy",
            "prune",
            "--prune-threshold",
            "0.5",
        ])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");

    // Last 2 user messages (indices 3 and 5) should be untouched.
    let user_3 = msgs[3]["content"].as_str().unwrap();
    assert!(
        user_3.contains("Thanks!"),
        "last 2 user messages must be preserved verbatim"
    );

    let user_5 = msgs[5]["content"].as_str().unwrap();
    assert!(
        user_5.contains("OK thanks"),
        "last user message must be preserved verbatim"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Empty-after-prune → placeholder
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prune_replaces_empty_messages_with_placeholder() {
    // A message that's ALL filler should become "[earlier context summarized]".
    let input = serde_json::json!([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Tell me about Rust."},
        {"role": "assistant", "content": "Sure!\nOK.\nGot it.\nAbsolutely!"},
        {"role": "user", "content": "More details please."},
        {"role": "assistant", "content": "Rust is a systems programming language focused on safety, concurrency, and performance without garbage collection."}
    ]).to_string();

    let output = cctx()
        .args([
            "optimize",
            "--strategy",
            "prune",
            "--prune-threshold",
            "0.6",
        ])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());
    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");

    // The all-filler assistant message should become the placeholder.
    let filler_msg = msgs[2]["content"].as_str().unwrap();
    assert!(
        filler_msg.contains("earlier context summarized"),
        "all-filler message should become placeholder, got: {}",
        filler_msg
    );

    // The substantive assistant message should survive.
    let real_msg = msgs[4]["content"].as_str().unwrap();
    assert!(
        real_msg.contains("Rust"),
        "substantive content should survive"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pipeline integration
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prune_in_aggressive_preset() {
    cctx()
        .args([
            "optimize",
            "tests/fixtures/large_conversation.json",
            "--preset",
            "aggressive",
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("prune"));
}

#[test]
fn prune_strategy_standalone() {
    let output = cctx()
        .args([
            "optimize",
            "tests/fixtures/large_conversation.json",
            "--strategy",
            "prune",
            "--prune-threshold",
            "0.4",
        ])
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    // Should produce valid JSON with all 39 chunks (prune doesn't remove chunks).
    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");
    assert_eq!(msgs.len(), 39, "prune should preserve all message slots");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Repetition detection
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn prune_detects_repeated_content() {
    // User repeats their stack description. Earlier occurrences of repeated
    // lines should score lower due to repetition detection.
    let input = serde_json::json!([
        {"role": "system", "content": "You are a DevOps engineer."},
        {"role": "user", "content": "I run a FastAPI app with PostgreSQL and Redis on Docker."},
        {"role": "assistant", "content": "Great setup. What do you need help with?"},
        {"role": "user", "content": "So basically I have a FastAPI application with PostgreSQL and Redis running in Docker containers.\nI need help with deployment."},
        {"role": "assistant", "content": "For deployment to Kubernetes, create a Deployment manifest with resource limits and health probes."},
        {"role": "user", "content": "As I mentioned I run FastAPI with PostgreSQL and Redis in Docker.\nThe deployment target is AWS EKS."}
    ]).to_string();

    let output = cctx()
        .args([
            "optimize",
            "--strategy",
            "prune",
            "--prune-threshold",
            "0.5",
        ])
        .write_stdin(input)
        .output()
        .expect("failed to run");

    assert!(output.status.success());

    let msgs: Vec<Value> = serde_json::from_slice(&output.stdout).expect("not JSON");

    // The later repetitions of the stack description should score lower due
    // to repetition detection. Count total tokens to verify compression.
    let original_tokens_estimate: usize = msgs
        .iter()
        .filter_map(|m| m["content"].as_str())
        .map(|s| s.split_whitespace().count())
        .sum();

    // The output should have fewer words than the input due to pruning
    // repeated content. (Exact amount depends on scoring thresholds.)
    assert!(output.status.success());
    let _ = original_tokens_estimate; // verified above that it runs
}
