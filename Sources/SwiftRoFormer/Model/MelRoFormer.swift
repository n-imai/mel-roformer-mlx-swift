import MLX
import MLXNN

/// Kim Mel-RoFormer model for vocal source separation.
///
/// Full inference pipeline:
/// ```
/// [B, 2, samples] → STFT → CaC interleave → BandSplit → 6× DualAxis →
/// MaskEstimate → scatter_add merge → complex multiply → iSTFT → [B, 2, samples]
/// ```
///
/// Weight key structure (708 total):
/// ```
/// band_split.to_features.{0-59}.{0,1}.*               (180 keys)
/// layers.{0-5}.{0,1}.layers.0.{0,1}.*                  (168 keys)
/// layers.{0-5}.{0,1}.norm.gamma
/// mask_estimators.0.to_freqs.{0-59}.0.{0,2,4}.*       (360 keys)
/// ```
public class MelRoFormer: Module {

    @ModuleInfo(key: "band_split") var bandSplit: BandSplit
    @ModuleInfo var layers: [[Transformer]]
    @ModuleInfo(key: "mask_estimators") var maskEstimators: [MaskEstimator]

    let config: RoFormerConfiguration

    /// Hann window for STFT/iSTFT (not learnable).
    let window: MLXArray

    public init(config: RoFormerConfiguration = .kimVocal2) {
        self.config = config
        self.window = WindowFunctions.hannWindow(size: config.nFFT)

        // BandSplit: 60 per-band projections
        self._bandSplit.wrappedValue = BandSplit(config: config)

        // DualAxisTransformer layers: 6 depth × [time, freq]
        // We store directly as [[Transformer]] to match key paths:
        //   layers.{0-5}.{0=time, 1=freq}.*
        self._layers.wrappedValue = (0..<config.depth).map { _ in
            [
                Transformer(
                    dim: config.dim, depth: 1,
                    heads: config.heads, dimHead: config.dimHead,
                    ffMult: config.ffMult
                ),
                Transformer(
                    dim: config.dim, depth: 1,
                    heads: config.heads, dimHead: config.dimHead,
                    ffMult: config.ffMult
                )
            ]
        }

        // MaskEstimator: 1 stem (vocals only), 60 bands
        let bandDims = self._bandSplit.wrappedValue.filterbank.bandDims
        self._maskEstimators.wrappedValue = [
            MaskEstimator(config: config, bandDims: bandDims)
        ]
    }

    /// Run the full separation pipeline on a single chunk.
    ///
    /// - Parameter audio: Input audio `[batch, 2, samples]` (stereo, 44.1kHz).
    /// - Returns: Separated vocal audio `[batch, 2, samples]`.
    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        forward(audio, bodyDType: .float32, capture: nil)
    }

    /// Same forward as `callAsFunction`, with:
    /// - `bodyDType`: precision of the transformer body. STFT/iSTFT always run in
    ///   fp32 (complex path); after STFT the CaC representation is cast to
    ///   `bodyDType` and the masked result is cast back to fp32 before iSTFT.
    ///   Pass `.float16` for the on-device shipping path (requires fp16 weights
    ///   loaded and the RMSNorm fp32-reduction patch). Mirrors mlx-audio step2.
    /// - `capture`: optional hook receiving each intermediate stage (name, tensor).
    ///   Production passes `nil` (no-op); the dump path passes a closure. Single
    ///   source of truth — no separate "debug forward" to drift.
    public func forward(_ audio: MLXArray, bodyDType: DType = .float32,
                        capture: ((String, MLXArray) -> Void)?) -> MLXArray {
        let originalLength = audio.shape[2]

        // Step 1: STFT → complex spectrogram [B, 2, freqBins, T]
        let stftComplex = STFT.stft(
            audio,
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            window: window
        )

        let B = stftComplex.shape[0]
        let freqBins = stftComplex.shape[2]  // 1025
        let T = stftComplex.shape[3]

        // Step 2: Convert complex to real/imaginary channels
        // stftComplex is complex-valued [B, 2, freqBins, T]
        // Extract real and imaginary parts: each [B, 2, freqBins, T]
        let stftReal = stftComplex.realPart()
        let stftImag = stftComplex.imaginaryPart()
        capture?("stft_R", stftReal)
        capture?("stft_I", stftImag)

        // Step 3: CaC interleave — rearrange "b s f t -> b (f s) t"
        // Interleave stereo channels per frequency: [f0_L, f0_R, f1_L, f1_R, ...]
        // stftReal/stftImag: [B, 2, freqBins, T]
        // → transpose to [B, freqBins, 2, T] then reshape to [B, freqBins*2, T]
        let realInterleaved = stftReal.transposed(0, 2, 1, 3).reshaped([B, freqBins * 2, T])
        let imagInterleaved = stftImag.transposed(0, 2, 1, 3).reshaped([B, freqBins * 2, T])

        // Stack real/imag as last dim: [B, freqBins*2, T, 2], then cast the CaC
        // representation to the body precision. STFT above stayed fp32; the body
        // (BandSplit → transformers → mask → complex-multiply) runs in bodyDType,
        // and the masked result is cast back to fp32 (complex64) before iSTFT.
        let stftRepr = stacked([realInterleaved, imagInterleaved], axis: -1).asType(bodyDType)

        // Step 4: BandSplit → [B, T, numBands, dim]
        var x = bandSplit.split(stftRepr)
        capture?("bandsplit_x", x)

        // Step 5: 6× Dual-axis transformer
        let Nb = x.shape[2]
        let D = x.shape[3]

        for (i, pair) in layers.enumerated() {
            let timeTransformer = pair[0]
            let freqTransformer = pair[1]

            // Time attention: [B, T, Nb, D] → [B*Nb, T, D]
            var timeInput = x.transposed(0, 2, 1, 3)  // [B, Nb, T, D]
            timeInput = timeInput.reshaped([B * Nb, T, D])
            let timeOutput = timeTransformer(timeInput)
            x = timeOutput.reshaped([B, Nb, T, D]).transposed(0, 2, 1, 3)

            // Frequency attention: [B, T, Nb, D] → [B*T, Nb, D]
            let freqInput = x.reshaped([B * T, Nb, D])
            let freqOutput = freqTransformer(freqInput)
            x = freqOutput.reshaped([B, T, Nb, D])
            capture?("layer\(i)_x", x)
        }

        // Step 6: Mask estimation → [B, T, totalBandDim]
        let masks = maskEstimators[0](x)
        capture?("masks", masks)

        // Step 7: Merge masks back to full spectrum → [B, freqBins*2, T, 2]
        let fullMask = bandSplit.merge(bandMasks: masks, freqBinsTimesTwo: freqBins * 2)
        capture?("full_mask", fullMask)

        // Step 8: Apply mask via complex multiplication
        // stftRepr: [B, freqBins*2, T, 2] (real/imag of input)
        // fullMask: [B, freqBins*2, T, 2] (real/imag of mask)
        // Complex multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        let inputReal = stftRepr[0..., 0..., 0..., 0]   // [B, freqBins*2, T]
        let inputImag = stftRepr[0..., 0..., 0..., 1]   // [B, freqBins*2, T]
        let maskReal = fullMask[0..., 0..., 0..., 0]     // [B, freqBins*2, T]
        let maskImag = fullMask[0..., 0..., 0..., 1]     // [B, freqBins*2, T]

        let outReal = inputReal * maskReal - inputImag * maskImag
        let outImag = inputReal * maskImag + inputImag * maskReal

        // Step 9: De-interleave CaC back to stereo channels
        // [B, freqBins*2, T] → [B, freqBins, 2, T] → [B, 2, freqBins, T]
        let realDeinterleaved = outReal.reshaped([B, freqBins, 2, T]).transposed(0, 2, 1, 3)
        let imagDeinterleaved = outImag.reshaped([B, freqBins, 2, T]).transposed(0, 2, 1, 3)

        // Step 10: Reconstruct complex spectrogram for iSTFT
        // realDeinterleaved + i * imagDeinterleaved
        // Use asImaginary() to convert imag part, then add
        let maskedComplex = realDeinterleaved.asType(.complex64)
            + imagDeinterleaved.asImaginary()

        // Step 11: iSTFT → [B, 2, samples]
        let separated = ISTFT.istft(
            maskedComplex,
            nFFT: config.nFFT,
            hopLength: config.hopLength,
            window: window,
            length: originalLength
        )
        capture?("out", separated)

        return separated
    }
}
