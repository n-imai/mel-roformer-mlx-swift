import Foundation

/// Configuration for the Kim Vocal 2 Mel-RoFormer model.
///
/// Default values match the Kim Vocal 2 checkpoint (228M parameters):
/// - dim=384, depth=6, heads=8, dimHead=64
/// - 60 mel bands, n_fft=2048, hop_length=441
/// - 44.1kHz stereo input
public struct RoFormerConfiguration: Sendable {

    // MARK: - Model Architecture

    /// Hidden dimension of the transformer.
    public var dim: Int = 384

    /// Number of dual-axis transformer depth levels.
    public var depth: Int = 6

    /// Number of attention heads.
    public var heads: Int = 8

    /// Dimension per attention head.
    public var dimHead: Int = 64

    /// Number of mel bands for band splitting.
    public var numBands: Int = 60

    /// Number of output stems (1 = vocals only).
    public var numStems: Int = 1

    /// Feed-forward expansion multiplier.
    public var ffMult: Int = 4

    /// MLP expansion factor in mask estimator.
    public var mlpExpansionFactor: Int = 4

    /// Depth of MLP in mask estimator (number of hidden layers).
    public var maskEstimatorDepth: Int = 2

    // MARK: - STFT Parameters

    /// FFT size.
    public var nFFT: Int = 2048

    /// Hop length between STFT frames.
    public var hopLength: Int = 441

    /// Window length for STFT.
    public var winLength: Int = 2048

    /// Sample rate in Hz.
    public var sampleRate: Double = 44100.0

    // MARK: - Derived Properties

    /// Inner dimension of attention (heads × dimHead).
    public var dimInner: Int { heads * dimHead }  // 512

    /// Feed-forward hidden dimension (dim × ffMult).
    public var ffDim: Int { dim * ffMult }  // 1536

    /// MLP hidden dimension in mask estimator.
    public var mlpHidden: Int { dim * mlpExpansionFactor }  // 1536

    /// Number of frequency bins from STFT (nFFT/2 + 1).
    public var freqBins: Int { nFFT / 2 + 1 }  // 1025

    // MARK: - Processing

    /// GPU memory cache limit in bytes.
    public var gpuCacheLimit: Int = 512 * 1024 * 1024  // 512 MB

    /// Overlap-add chunk size in samples (~11.0s at 44.1kHz).
    ///
    /// This is the VERIFIED kim_vocal_2 demix recipe value, reproducing the
    /// audio-separator Roformer chunking that produced the approved separation
    /// quality: `chunkSize = stft_hop(441) × (dim_t(1101) - 1) = 485100`. It is a
    /// fixed constant for this checkpoint, NOT derived from `hopLength` (other
    /// presets have a different hop and no validated chunking params).
    public var chunkSize: Int = 485_100

    /// Overlap-add hop between consecutive chunks in samples (~8.0s at 44.1kHz).
    ///
    /// `chunkStep = min(overlap(8) × sampleRate, chunkSize) = 352800`, giving a 3.0s
    /// overlap region per the verified recipe. Final chunks are end-anchored and the
    /// per-sample Hamming weights are removed by divide-by-counter (see
    /// ``OverlapAdd``).
    public var chunkStep: Int = 352_800

    // MARK: - Presets

    /// Kim Vocal 2 checkpoint defaults (GPL-3.0 weights).
    ///
    /// 228 M parameters. dim=384, depth=6, mask_estimator_depth=2, hop=441.
    public static let kimVocal2 = RoFormerConfiguration()

    /// ZFTurbo v1.0.0 vocals checkpoint — the MIT-licensed preset.
    ///
    /// Matches release asset `model_vocals_mel_band_roformer_sdr_8.42.ckpt`
    /// from ZFTurbo/Music-Source-Separation-Training v1.0.0. Smaller than
    /// Kim Vocal 2 (~128 MB) with a narrower transformer and single-hidden
    /// mask estimator MLP — runs faster and is redistributable under MIT.
    ///
    /// Architecture differences vs `kimVocal2`:
    /// - `dim: 192` (vs 384)
    /// - `depth: 8` (vs 6)
    /// - `hopLength: 512` (vs 441)
    /// - `maskEstimatorDepth: 1` (vs 2)
    public static let zfturboVocalsV1: RoFormerConfiguration = {
        var config = RoFormerConfiguration()
        config.dim = 192
        config.depth = 8
        config.hopLength = 512
        config.maskEstimatorDepth = 1
        return config
    }()

    public init() {}
}
