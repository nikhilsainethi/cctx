//! Stateful token-budget tracker used by [`super::injection_builder`].
//!
//! Most callers can use the simple greedy pattern:
//!
//! ```rust
//! use cctx::compaction::budget::BudgetManager;
//!
//! let mut budget = BudgetManager::new(4096);
//! for tokens in [800usize, 1200, 1500, 2000] {
//!     if budget.allocate(tokens).is_ok() {
//!         // …include the item…
//!     } else {
//!         break;
//!     }
//! }
//! assert!(budget.allocated() <= 4096);
//! ```
//!
//! The struct deliberately stays tiny (limit + counter). Anything more
//! sophisticated belongs in the caller — for example, the injection
//! builder bypasses `allocate` for a "single oversized item" case
//! where it truncates the item to fit the *remaining* budget instead.

use anyhow::{anyhow, Result};

/// Tracks how many tokens have been allocated against a fixed limit.
///
/// `allocate` is a try-and-record operation: it succeeds and records
/// the allocation only when the requested tokens fit inside the
/// remaining budget. There is no de-allocation — the typical use is
/// a single greedy pass, after which the manager is dropped.
#[derive(Debug, Clone)]
pub struct BudgetManager {
    limit: usize,
    used: usize,
}

impl BudgetManager {
    /// Build a new manager with `limit` tokens available.
    pub fn new(limit: usize) -> Self {
        Self { limit, used: 0 }
    }

    /// Total budget the manager was created with.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// How many tokens have been allocated so far.
    pub fn allocated(&self) -> usize {
        self.used
    }

    /// Tokens still available — `limit - allocated()`.
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.used)
    }

    /// Whether `tokens` would fit in the current remaining budget.
    /// Pure read — does not mutate state.
    pub fn can_fit(&self, tokens: usize) -> bool {
        tokens <= self.remaining()
    }

    /// Reserve `tokens` from the remaining budget.
    ///
    /// # Errors
    ///
    /// Returns `Err` (without mutating state) if `tokens > remaining()`.
    pub fn allocate(&mut self, tokens: usize) -> Result<()> {
        if !self.can_fit(tokens) {
            return Err(anyhow!(
                "budget exhausted: requested {} tokens, only {} remaining",
                tokens,
                self.remaining()
            ));
        }
        self.used += tokens;
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_manager_reports_full_remaining() {
        let m = BudgetManager::new(1000);
        assert_eq!(m.limit(), 1000);
        assert_eq!(m.allocated(), 0);
        assert_eq!(m.remaining(), 1000);
        assert!(m.can_fit(500));
        assert!(m.can_fit(1000));
        assert!(!m.can_fit(1001));
    }

    #[test]
    fn allocate_consumes_remaining() {
        let mut m = BudgetManager::new(1000);
        m.allocate(300).unwrap();
        assert_eq!(m.allocated(), 300);
        assert_eq!(m.remaining(), 700);
        m.allocate(700).unwrap();
        assert_eq!(m.remaining(), 0);
        assert!(m.allocate(1).is_err(), "over-budget alloc must error");
    }

    #[test]
    fn failed_allocation_does_not_mutate_state() {
        let mut m = BudgetManager::new(100);
        m.allocate(60).unwrap();
        assert!(m.allocate(50).is_err());
        // Still 60, NOT 60+50.
        assert_eq!(m.allocated(), 60);
        assert_eq!(m.remaining(), 40);
    }

    #[test]
    fn zero_token_alloc_is_a_noop_success() {
        let mut m = BudgetManager::new(100);
        m.allocate(0).unwrap();
        assert_eq!(m.allocated(), 0);
    }
}
