import Foundation

/// Long-form overlap-add chunking helpers for the verified audio-separator Roformer
/// demix recipe — the one that produced the approved separation quality on Mac
/// (Step ③ / Step 2 verification). Pure Swift (no MLX), so they unit-test under plain
/// `swift test` without a staged metallib.
///
/// Recipe (kim_vocal_2 path, 44.1kHz):
///   chunk_size = 485100 (11s) / step = 352800 (8s) / symmetric Hamming(485100) /
///   end-anchored final chunks / overlap-add → divide-by-counter.
enum OverlapAdd {

    /// A symmetric Hamming window of `size` samples.
    ///
    /// Matches `scipy.signal.windows.hamming(size)` (default `sym=True`):
    /// `w[n] = 0.54 - 0.46·cos(2π·n / (size-1))`, `n = 0..size-1`.
    /// Endpoints are 0.08, midpoint 1.0.
    ///
    /// This is the OLA *chunk* window. It is deliberately NOT the STFT analysis window
    /// (which is a periodic Hann in ``WindowFunctions`` and divides by `size`, giving a
    /// different endpoint). Reusing the periodic form here would shift every overlap
    /// blend.
    static func hammingWindow(_ size: Int) -> [Float] {
        precondition(size > 1, "Hamming window needs at least 2 samples")
        let denom = Double(size - 1)
        var w = [Float](repeating: 0, count: size)
        for n in 0..<size {
            w[n] = Float(0.54 - 0.46 * cos(2.0 * Double.pi * Double(n) / denom))
        }
        return w
    }

    /// Where one chunk reads from the mix and writes into the output buffer.
    struct ChunkPlacement: Equatable {
        /// First input sample this chunk reads.
        let readStart: Int
        /// First output sample this chunk contributes to.
        let writeStart: Int
        /// Chunk length in samples (always `chunkSize` here).
        let length: Int
        /// Final chunk re-anchored to the tail (`mix[-chunkSize:]`).
        let endAnchored: Bool
    }

    /// Plan the chunk layout matching the Python reference (`step3_chunk.py`):
    ///
    /// ```
    /// starts = range(0, sampleCount, step)
    /// ```
    ///
    /// A chunk whose nominal end `i + chunkSize` exceeds `sampleCount` is END-ANCHORED:
    /// it reads the last `chunkSize` samples and writes at `sampleCount - chunkSize`.
    /// (The reference can enumerate two trailing starts that both anchor to the same
    /// position — that redundancy is faithful and cancels under divide-by-counter.)
    ///
    /// - Precondition: `sampleCount > chunkSize`. The single-chunk fast path handles
    ///   `sampleCount <= chunkSize`, which also avoids the negative write-start the
    ///   Python reference would compute there.
    static func chunkPlan(sampleCount: Int, chunkSize: Int, step: Int) -> [ChunkPlacement] {
        precondition(sampleCount > chunkSize, "chunkPlan requires sampleCount > chunkSize")
        precondition(step > 0, "step must be positive")

        var plan: [ChunkPlacement] = []
        var i = 0
        while i < sampleCount {
            if i + chunkSize > sampleCount {
                let start = sampleCount - chunkSize
                plan.append(ChunkPlacement(readStart: start, writeStart: start,
                                           length: chunkSize, endAnchored: true))
            } else {
                plan.append(ChunkPlacement(readStart: i, writeStart: i,
                                           length: chunkSize, endAnchored: false))
            }
            i += step
        }
        return plan
    }
}
