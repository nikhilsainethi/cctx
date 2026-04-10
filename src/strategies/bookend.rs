use crate::core::context::{Chunk, Context};

/// Apply bookend reordering to a context.
///
/// Algorithm (Liu et al., TACL 2024):
///   1. Sort chunks by relevance score, highest first.
///   2. Alternate placement between the front and back of the output list:
///        rank 1 → position 0  (beginning)
///        rank 2 → position N-1 (end)
///        rank 3 → position 1  (near beginning)
///        rank 4 → position N-2 (near end)
///        … and so on inward.
///
/// Cost: zero tokens added or removed — pure reordering.
/// Expected benefit: 10–30% recall improvement for content that was in the dead zone.
///
/// Returns a new Vec<Chunk> in the reordered order.
/// The original `context` is borrowed immutably; we .clone() chunks to build the output.
pub fn apply(context: &Context) -> Vec<Chunk> {
    // Clone the chunks so we can sort without modifying the original context.
    // Vec::clone() is possible because Chunk derives Clone.
    let mut by_relevance: Vec<Chunk> = context.chunks.clone();

    // Sort descending by relevance_score.
    // f64 doesn't implement Ord (because NaN != NaN), so we use partial_cmp.
    // unwrap_or(Equal) treats NaN comparisons as ties — safe here.
    by_relevance.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n = by_relevance.len();
    if n == 0 {
        return vec![];
    }

    // Allocate result slots as Option<Chunk>.
    // Option<T> is either Some(value) or None — we fill slots one by one.
    // vec![None; n] creates a Vec of n Nones.
    let mut result: Vec<Option<Chunk>> = vec![None; n];

    let mut front = 0usize;
    // usize can't go negative, so we use n - 1 only after checking n > 0 (done above).
    let mut back = n - 1;

    for (rank, chunk) in by_relevance.into_iter().enumerate() {
        if front > back {
            break; // All slots filled (handles odd n correctly).
        }
        if rank % 2 == 0 {
            result[front] = Some(chunk);
            front += 1;
        } else {
            result[back] = Some(chunk);
            // Prevent underflow: back is usize, so 0 - 1 would wrap/panic in debug.
            if back == 0 {
                break;
            }
            back -= 1;
        }
    }

    // flatten() on an iterator of Option<T> drops the Nones and unwraps the Somes.
    result.into_iter().flatten().collect()
}
