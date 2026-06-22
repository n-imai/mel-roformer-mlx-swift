import Foundation
import Testing
@testable import SwiftRoFormer

/// Pins the long-form chunking convention to the verified audio-separator Roformer
/// demix recipe — the one that produced the approved separation quality:
///   chunk_size = 485100 (11s) / step = 352800 (8s) / scipy SYMMETRIC Hamming(485100) /
///   end-anchored final chunks / overlap-add → divide-by-counter.
///
/// These are PURE helpers (no MLX), so they run under plain `swift test` without a
/// staged metallib. The window CONVENTION (symmetric vs periodic) and the end-anchor
/// PLACEMENT are the two silent-bug surfaces that a global residual dB can mask, so
/// they get direct value/placement assertions here (Step 2 verification, 2026-06-22).
struct OverlapAddTests {

    // MARK: - Hamming window (scipy.signal.windows.hamming default = symmetric)

    @Test func hammingWindowTinyMatchesScipyExactly() {
        // scipy.signal.windows.hamming(5) == [0.08, 0.54, 1.0, 0.54, 0.08]
        // Formula: 0.54 - 0.46*cos(2πn/(M-1)), n=0..M-1, sym=True (NOT periodic ÷M).
        let w = OverlapAdd.hammingWindow(5)
        let expected: [Float] = [0.08, 0.54, 1.0, 0.54, 0.08]
        #expect(w.count == 5)
        for (got, want) in zip(w, expected) {
            #expect(abs(got - want) < 1e-6)
        }
    }

    @Test func hammingWindowFullChunkEndpointsAreScipyValues() {
        // The real OLA window. scipy reference (computed in the PoC venv):
        //   w[0] = w[M-1] = 0.08 (symmetric endpoints), w[M/2] = 1.0.
        // A periodic Hann/Hamming (÷size) would give w[0] != 0.08 — guards against
        // accidentally reusing WindowFunctions.hannWindow (periodic).
        let M = 485100
        let w = OverlapAdd.hammingWindow(M)
        #expect(w.count == M)
        #expect(abs(w[0] - 0.08) < 1e-4)
        #expect(abs(w[M - 1] - 0.08) < 1e-4)
        #expect(abs(w[M / 2] - 1.0) < 1e-4)
        #expect(w[1] >= w[0])  // rising from the start
    }

    // MARK: - Chunk plan (range(0,N,step) + end-anchored final chunks)

    @Test func chunkPlanSmallCaseMatchesPythonEnumeration() {
        // Mirror the Python reference on small numbers (chunkSize=10, step=7, N=20):
        //   starts = range(0,20,7) = [0, 7, 14]
        //   i=0 : 0+10<=20  → read 0,  write 0,  len 10, anchored=false
        //   i=7 : 7+10<=20  → read 7,  write 7,  len 10, anchored=false
        //   i=14: 14+10>20  → read N-10=10, write 10, len 10, anchored=true (end-anchor)
        let plan = OverlapAdd.chunkPlan(sampleCount: 20, chunkSize: 10, step: 7)
        #expect(plan.count == 3)

        #expect(plan[0].readStart == 0  && plan[0].writeStart == 0  && plan[0].length == 10 && plan[0].endAnchored == false)
        #expect(plan[1].readStart == 7  && plan[1].writeStart == 7  && plan[1].length == 10 && plan[1].endAnchored == false)
        #expect(plan[2].readStart == 10 && plan[2].writeStart == 10 && plan[2].length == 10 && plan[2].endAnchored == true)
    }

    @Test func chunkPlanEndAnchorPinsFinalWriteToTail() {
        // With the real params, the final chunk must be anchored so its WRITE region
        // ends exactly at N (writeStart = N - chunkSize), never past it.
        let chunkSize = 485_100
        let step = 352_800
        let N = 3 * step + 12_345  // forces a final partial → end-anchored chunk
        let plan = OverlapAdd.chunkPlan(sampleCount: N, chunkSize: chunkSize, step: step)

        // Starts enumerated as range(0, N, step).
        let expectedStarts = Array(stride(from: 0, to: N, by: step))
        #expect(plan.count == expectedStarts.count)

        let last = plan.last!
        #expect(last.endAnchored == true)
        #expect(last.writeStart == N - chunkSize)
        #expect(last.writeStart + last.length == N)  // write region ends exactly at N

        // Every placement stays in bounds and uses a full-length chunk.
        for p in plan {
            #expect(p.readStart >= 0)
            #expect(p.readStart + p.length <= N)
            #expect(p.length == chunkSize)
        }
    }
}
