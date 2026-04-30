//! Persistent state for the Compaction Guard feature, rooted at `.cctx/`
//! within the project directory.
//!
//! Layout (Compaction Guard architecture §3.6):
//!
//! ```text
//! .cctx/
//! ├── config.json                       # cctx state configuration
//! ├── compaction-log.json               # appended event log
//! ├── fingerprints/                     # per-session pre-compaction snapshots
//! ├── loss-reports/                     # per-session post-compaction analysis
//! ├── pending-injection/                # one-shot payload for next SessionStart
//! └── injection-history/                # archive of past injections
//! ```
//!
//! All state is local: it is per-developer and per-machine, not committed
//! to the repository.

pub mod history;
pub mod store;
