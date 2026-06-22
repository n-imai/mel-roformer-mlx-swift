import Foundation
import MLX
import MLXNN
import SwiftRoFormer

// Step 2 long-form chunking harness (Swift side).
//
// Runs the PRODUCTION chunked overlap-add path (RoFormerSeparator.separateChunked)
// on a full-length mix loaded byte-identically from a safetensors written by the
// Python reference, and writes the separated vocals to a safetensors for residual
// comparison against the mlx-audio Python chunker (step3_chunk.py).
//
// Single source of truth: this exercises the real shipping code path, not a
// re-implementation — so a PASS here certifies the production wrapper.
//
// Usage: roformer-chunk <weights.safetensors> <mix.safetensors> <out.safetensors> [fp32|fp16]
//   mix.safetensors: key "audio", shape [1,2,N] or [2,N], 44.1kHz stereo.
//   For a faithful fp16 run, pass fp16 weights AND "fp16" (RMSNorm fp32-reduction
//   patch is built into the model).

guard CommandLine.arguments.count == 4 || CommandLine.arguments.count == 5 else {
    FileHandle.standardError.write(Data("usage: roformer-chunk <weights> <mix.safetensors> <out.safetensors> [fp32|fp16]\n".utf8))
    exit(2)
}

let weightsURL = URL(fileURLWithPath: CommandLine.arguments[1])
let mixURL = URL(fileURLWithPath: CommandLine.arguments[2])
let outURL = URL(fileURLWithPath: CommandLine.arguments[3])
let bodyDType: DType = (CommandLine.arguments.count == 5 && CommandLine.arguments[4] == "fp16") ? .float16 : .float32

// kim_vocal_2 architecture (the VoicyCare PoC checkpoint). WeightLoader rejects a
// mismatched checkpoint with a strict-load error.
let model = MelRoFormer()
try WeightLoader.loadWeights(into: model, from: weightsURL)

let mixArrays = try MLX.loadArrays(url: mixURL)
guard let rawMix = mixArrays["audio"] else {
    FileHandle.standardError.write(Data("mix safetensors missing key 'audio'\n".utf8))
    exit(3)
}
// Normalize to [1, 2, N] fp32.
var mix = rawMix.asType(.float32)
if mix.ndim == 2 { mix = mix.expandedDimensions(axis: 0) }
guard mix.ndim == 3, mix.shape[1] == 2 else {
    FileHandle.standardError.write(Data("expected mix [1,2,N] or [2,N], got \(mix.shape)\n".utf8))
    exit(3)
}
let n = mix.shape[2]
MLX.eval(mix)

let separator = RoFormerSeparator(model: model)

let t0 = CFAbsoluteTimeGetCurrent()
let out = try await separator.separate(samples: mix, bodyDType: bodyDType)
MLX.eval(out)
let computeSec = CFAbsoluteTimeGetCurrent() - t0

let hasNaN = MLX.any(MLX.isNaN(out)).item(Bool.self)
let maxAbs = MLX.max(MLX.abs(out)).item(Float.self)         // NaN-poisoned if hasNaN
let sumSq = MLX.sum(MLX.multiply(out, out)).item(Float.self)
let rms = (sumSq / Float(2 * n)).squareRoot()
let hasInf = maxAbs.isInfinite

let audioSec = Double(n) / 44100.0

// Non-finite metrics (NaN/Inf — exactly the failure this harness exists to catch) are
// not valid JSON, so JSONSerialization would throw and a swallowed `try?` would emit
// NOTHING. Encode them as string tokens so the JSON stays valid, and ALSO print the
// critical flags on a plain line that never depends on serialization.
func jsonSafe(_ x: Float) -> Any { x.isFinite ? Double(x) : "\(x)" }   // -> "nan"/"inf"/"-inf"

FileHandle.standardError.write(Data(
    "STATS nan=\(hasNaN) inf=\(hasInf) peak=\(maxAbs) rms=\(rms) n=\(n)\n".utf8))

let report: [String: Any] = [
    "body_dt": (bodyDType == .float16 ? "float16" : "float32"),
    "n_samples": n,
    "nan": hasNaN,
    "inf": hasInf,
    "peak": jsonSafe(maxAbs),
    "rms": jsonSafe(rms),
    "audio_sec": (audioSec * 100).rounded() / 100,
    "compute_sec": (computeSec * 100).rounded() / 100,
    "rtf_total": computeSec > 0 ? ((audioSec / computeSec * 1000).rounded() / 1000) : 0,
]
if let json = try? JSONSerialization.data(withJSONObject: report),
   let s = String(data: json, encoding: .utf8) {
    print(s)
} else {
    print("{\"nan\":\(hasNaN),\"inf\":\(hasInf),\"n_samples\":\(n)}")
}

try MLX.save(arrays: ["audio": out.asType(.float32)], url: outURL)
print("WROTE \(outURL.lastPathComponent)")
