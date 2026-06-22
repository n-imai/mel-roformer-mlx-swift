import MLX
import MLXNN

/// Custom RMSNorm matching lucidrains' L2-normalize-and-scale formulation.
///
/// **NOT** equivalent to `MLXNN.RMSNorm`. lucidrains uses L2-normalization:
///
/// ```python
/// output = F.normalize(x, dim=-1) * sqrt(dim) * gamma
/// ```
///
/// This normalizes each vector to unit length, then scales by `sqrt(dim)` and
/// a learnable `gamma` parameter. Standard RMSNorm divides by `sqrt(mean(x²) + eps)`
/// which is a fundamentally different operation.
///
/// ### Epsilon convention
///
/// PyTorch `F.normalize` applies eps as `max(||x||₂, eps)` — clamping the
/// *denominator* after the sqrt, not the sum-of-squares before it. Clamping
/// inside the sqrt (e.g. `sqrt(clip(||x||₂², min: 1e-12))`) diverges from
/// F.normalize for small norms and caps the SDR ceiling against a PyTorch
/// reference. The mlx-audio Python port (SDR 58 dB vs PyTorch) matches the
/// denominator-clamp convention exactly, so this implementation mirrors it.
///
/// Weight key: `gamma` (shape `[dim]`)
class RoFormerRMSNorm: Module {

    /// Learnable scale parameter, loaded from checkpoint.
    let gamma: MLXArray

    /// Constant scale factor: `sqrt(dim)`.
    let scale: Float

    /// `F.normalize` default epsilon. Clamps the denominator after the sqrt.
    private static let eps: Float = 1e-12

    init(dim: Int) {
        self.scale = Float(dim).squareRoot()
        self.gamma = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // fp32-reduction patch: compute the L2 norm (sum of squares) in fp32 even
        // when x is fp16. The unnormalized sum(x²) reaches ~1.5e5 on some bands,
        // which overflows fp16 (max 65504) → inf/NaN. This is the ONLY unbounded
        // fp16 reduction in the model. Identical to fp32 when x is already fp32
        // (the casts are no-ops). Mirrors mlx-audio's patched RMSNorm.
        let xf = x.asType(.float32)
        let squaredSum = sum(xf * xf, axis: -1, keepDims: true)
        let norm = sqrt(squaredSum)
        let normalized = xf / maximum(norm, MLXArray(Self.eps))
        // Scale by sqrt(dim) and learnable gamma
        let out = normalized * scale * gamma.asType(.float32)
        return out.asType(x.dtype)
    }
}
