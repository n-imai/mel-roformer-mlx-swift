import Foundation
import MLX
import MLXNN
import SwiftRoFormer

// Step 0 fidelity harness (Swift side).
// Runs the REAL MelRoFormer.forward on a fixed input (loaded byte-identically
// from a safetensors written by the Python reference) and dumps every
// intermediate stage to a safetensors for stage-by-stage residual comparison
// against the mlx-audio Python reference.
//
// Usage: roformer-dump <weights.safetensors> <input.safetensors> <out_stages.safetensors> [fp32|fp16]
//   The optional 4th arg sets the transformer-body precision (default fp32).
//   For a faithful fp16 run, pass fp16 weights AND "fp16".

guard CommandLine.arguments.count == 4 || CommandLine.arguments.count == 5 else {
    FileHandle.standardError.write(Data("usage: roformer-dump <weights> <input.safetensors> <out_stages.safetensors> [fp32|fp16]\n".utf8))
    exit(2)
}

let weightsURL = URL(fileURLWithPath: CommandLine.arguments[1])
let inputURL = URL(fileURLWithPath: CommandLine.arguments[2])
let outURL = URL(fileURLWithPath: CommandLine.arguments[3])
let bodyDType: DType = (CommandLine.arguments.count == 5 && CommandLine.arguments[4] == "fp16") ? .float16 : .float32

let model = MelRoFormer()
try WeightLoader.loadWeights(into: model, from: weightsURL)

let inArrays = try MLX.loadArrays(url: inputURL)
guard let audio = inArrays["audio"] else {
    FileHandle.standardError.write(Data("input safetensors missing key 'audio'\n".utf8))
    exit(3)
}
// Force fp32, [1, 2, N]
let audioF = audio.asType(.float32)

var stages: [String: MLXArray] = [:]
let out = model.forward(audioF, bodyDType: bodyDType) { name, arr in
    let a = arr.asType(.float32)
    MLX.eval(a)
    stages[name] = a
}
MLX.eval(out)
let hasNaN = MLX.any(MLX.isNaN(out)).item(Bool.self)
let maxAbs = MLX.max(MLX.abs(out)).item(Float.self)
FileHandle.standardError.write(Data("bodyDType=\(bodyDType) NaN=\(hasNaN) maxAbs=\(maxAbs)\n".utf8))

try MLX.save(arrays: stages, url: outURL)
print("wrote \(stages.count) stages -> \(outURL.lastPathComponent)")
for k in stages.keys.sorted() {
    print("  \(k): \(stages[k]!.shape)")
}
