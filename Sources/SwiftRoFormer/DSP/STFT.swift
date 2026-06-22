import MLX
import MLXFFT

/// Short-Time Fourier Transform implementation using MLXFFT primitives.
///
/// Converts a time-domain signal into a complex-valued spectrogram
/// by windowing overlapping frames and applying rfft to each.
///
/// Kim Mel-RoFormer uses n_fft=2048, hop_length=441, Hann window.
enum STFT {

    /// Compute the Short-Time Fourier Transform of a signal.
    ///
    /// - Parameters:
    ///   - signal: Real-valued input of shape `[batch, channels, samples]`.
    ///   - nFFT: FFT size (default 2048).
    ///   - hopLength: Hop size between frames (default 441).
    ///   - window: Window function array of shape `[nFFT]`.
    /// - Returns: Complex-valued spectrogram of shape `[batch, channels, freq_bins, frames]`
    ///            where `freq_bins = nFFT / 2 + 1`.
    static func stft(
        _ signal: MLXArray,
        nFFT: Int = 2048,
        hopLength: Int = 441,
        window: MLXArray
    ) -> MLXArray {
        precondition(signal.ndim == 3, "STFT input must be 3D [batch, channels, samples], got \(signal.ndim)D")
        precondition(window.ndim == 1 && window.shape[0] == nFFT,
                     "Window must be 1D with length \(nFFT), got shape \(window.shape)")

        let shape = signal.shape
        let batch = shape[0]
        let channels = shape[1]
        let samples = shape[2]

        // Reshape to [batch * channels, samples] for frame extraction
        let flat = signal.reshaped([batch * channels, samples])

        // Pad signal: center reflect padding with nFFT/2 on each side.
        // Matches torch.stft(center=True, pad_mode="reflect") and mlx-audio
        // dsp.stft — zero padding here corrupts the edge frames and (via time
        // attention) contaminates the whole spectrogram. See Step 0 fidelity.
        let padAmount = nFFT / 2
        let padded = reflectPadded(flat, pad: padAmount)

        let paddedLength = padded.shape[1]

        // Calculate number of frames
        let numFrames = (paddedLength - nFFT) / hopLength + 1

        // Extract frames using strided indexing
        // Build frame indices: for each frame f, extract samples [f*hop .. f*hop+nFFT)
        var frames = [MLXArray]()
        frames.reserveCapacity(numFrames)
        for f in 0..<numFrames {
            let start = f * hopLength
            let frame = padded[0..., start..<(start + nFFT)]  // [batch*channels, nFFT]
            frames.append(frame)
        }
        // Stack to [numFrames, batch*channels, nFFT] then transpose to [batch*channels, numFrames, nFFT]
        var stacked = stacked(frames, axis: 0)
        stacked = stacked.transposed(1, 0, 2)  // [batch*channels, numFrames, nFFT]

        // Apply window: broadcast [nFFT] over frames
        let windowed = stacked * window

        // Apply rfft along the last axis
        let spectrum = MLXFFT.rfft(windowed, axis: -1)  // [batch*channels, numFrames, nFFT/2+1]

        // Reshape to [batch, channels, numFrames, freq_bins] then transpose to [batch, channels, freq_bins, numFrames]
        let freqBins = nFFT / 2 + 1
        let result = spectrum.reshaped([batch, channels, numFrames, freqBins])
        return result.transposed(0, 1, 3, 2)  // [batch, channels, freq_bins, frames]
    }

    /// Reflect-pad a 2D array `[rows, samples]` on the last axis with torch /
    /// mlx-audio "reflect" semantics (boundary sample excluded):
    ///   left  = reverse(x[:, 1 : pad+1])
    ///   right = reverse(x[:, samples-pad-1 : samples-1])
    private static func reflectPadded(_ x: MLXArray, pad: Int) -> MLXArray {
        if pad == 0 { return x }
        let samples = x.shape[1]
        let revIdx = MLXArray((0..<pad).reversed().map { Int32($0) })
        let left = x[0..., 1 ..< (pad + 1)].take(revIdx, axis: 1)
        let right = x[0..., (samples - pad - 1) ..< (samples - 1)].take(revIdx, axis: 1)
        return concatenated([left, x, right], axis: 1)
    }
}
