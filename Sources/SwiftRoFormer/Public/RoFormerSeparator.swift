import Foundation
import MLX
import MLXNN
import os

// MARK: - Error Types

/// Errors that can occur during RoFormer vocal separation.
public enum RoFormerError: Error, Sendable, LocalizedError {
    case weightsNotFound(String)
    case audioReadFailed(Error)
    case mlxInferenceFailed(Error)
    case outputWriteFailed(Error)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .weightsNotFound(let path):
            return "RoFormer weights not found at: \(path)"
        case .audioReadFailed(let error):
            return "Failed to read audio file: \(error.localizedDescription)"
        case .mlxInferenceFailed(let error):
            return "MLX inference failed: \(error.localizedDescription)"
        case .outputWriteFailed(let error):
            return "Failed to write output file: \(error.localizedDescription)"
        case .cancelled:
            return "Separation was cancelled"
        }
    }
}

// MARK: - Progress Types

/// Processing stages for progress reporting.
public enum RoFormerStage: String, Sendable {
    case loading = "Loading audio"
    case stft = "STFT"
    case bandSplit = "Band split"
    case transformer = "Transformer"
    case maskEstimate = "Mask estimation"
    case reconstruct = "Reconstruction"
    case writing = "Writing output"
}

/// Progress update during vocal separation.
public struct RoFormerProgress: Sendable {
    /// Overall progress fraction from 0.0 to 1.0.
    public let fraction: Float

    /// Current processing stage.
    public let stage: RoFormerStage

    /// Elapsed time since separation started (seconds).
    public let elapsedSeconds: Double
}

// MARK: - Delegate Protocol

/// Delegate protocol for fire-and-forget separation progress.
@MainActor
public protocol RoFormerProgressDelegate: AnyObject {
    /// Called when separation progress updates.
    func roFormer(
        _ separator: RoFormerSeparator,
        didUpdateProgress progress: Float,
        stage: RoFormerStage
    )

    /// Called when separation completes (success or failure).
    func roFormer(
        _ separator: RoFormerSeparator,
        didCompleteWithResult result: Result<StemResult, RoFormerError>
    )
}

// MARK: - RoFormerSeparator

/// Kim Mel-RoFormer vocal separator.
///
/// Separates vocals from music using the Kim Vocal 2 Mel-RoFormer model
/// (228M parameters, ~12.6 dB SDR). Processes audio in 11-second chunks with an
/// 8-second hop (3s overlap), end-anchored final chunks, and a symmetric-Hamming
/// overlap-add normalized by divide-by-counter — the verified audio-separator demix
/// recipe (see ``OverlapAdd``).
///
/// Three API styles are available:
///
/// **Async/await** (recommended):
/// ```swift
/// let separator = try await RoFormerSeparator(weightsDirectory: weightsURL)
/// let result = try await separator.separateVocals(from: inputURL, to: outputDir)
/// print("Vocals at: \(result.vocalsURL)")
/// ```
///
/// **AsyncThrowingStream** (progress tracking):
/// ```swift
/// for try await progress in separator.separationStream(from: inputURL, to: outputDir) {
///     print("Progress: \(progress.fraction)")
/// }
/// ```
///
/// **Delegate** (fire-and-forget):
/// ```swift
/// separator.delegate = self
/// separator.separateVocals(from: inputURL, to: outputDir)
/// ```
public final class RoFormerSeparator: @unchecked Sendable {

    // MARK: - Properties

    /// Delegate for fire-and-forget progress reporting.
    @MainActor public weak var delegate: RoFormerProgressDelegate?

    private let model: MelRoFormer
    private let audioIO: AudioIO
    private let config: RoFormerConfiguration
    private let cancelFlag: OSAllocatedUnfairLock<Bool>

    // MARK: - Initialization

    /// Create a separator around an already-loaded model.
    ///
    /// Use this when the model's lifecycle is owned elsewhere — e.g. a host that paged the
    /// weights in via ``MelRoFormer/fromPretrained(_:configuration:hub:progress:)`` and wants the
    /// separator to reuse that single resident instance rather than load weights again. The
    /// `configuration` must match the one the model was built with (it drives chunking / STFT).
    ///
    /// - Parameters:
    ///   - model: A `MelRoFormer` with weights already loaded.
    ///   - configuration: Processing configuration (must match `model`'s; default: Kim Vocal 2).
    public init(
        model: MelRoFormer,
        configuration: RoFormerConfiguration = .kimVocal2
    ) {
        self.config = configuration
        self.audioIO = AudioIO()
        self.cancelFlag = OSAllocatedUnfairLock(initialState: false)
        MLX.Memory.cacheLimit = configuration.gpuCacheLimit
        self.model = model
    }

    /// Create a new RoFormer separator, loading model weights from disk.
    ///
    /// - Parameters:
    ///   - weightsDirectory: Directory containing `mel_roformer_vocals.safetensors`.
    ///   - configuration: Model and processing configuration (default: Kim Vocal 2).
    /// - Throws: `RoFormerError.weightsNotFound` if weights file is missing.
    public init(
        weightsDirectory: URL,
        configuration: RoFormerConfiguration = .kimVocal2
    ) async throws {
        self.config = configuration
        self.audioIO = AudioIO()
        self.cancelFlag = OSAllocatedUnfairLock(initialState: false)

        // Set GPU memory cache limit
        MLX.Memory.cacheLimit = configuration.gpuCacheLimit

        // Build model architecture
        let model = MelRoFormer(config: configuration)

        // Load weights
        let weightsURL = weightsDirectory.appendingPathComponent(WeightLoader.vocalsWeightsFile)
        do {
            try WeightLoader.loadWeights(into: model, from: weightsURL)
        } catch {
            throw RoFormerError.weightsNotFound(weightsURL.path)
        }

        self.model = model
    }

    // MARK: - Public API: Async/Await

    /// Separate vocals from an audio file.
    ///
    /// - Parameters:
    ///   - inputURL: Path to input audio file (WAV, MP3, FLAC, M4A).
    ///   - outputURL: Directory for output files.
    /// - Returns: `StemResult` with URLs to separated vocals and accompaniment.
    /// - Throws: `RoFormerError` on failure or cancellation.
    public func separateVocals(
        from inputURL: URL,
        to outputURL: URL
    ) async throws -> StemResult {
        cancelFlag.withLock { $0 = false }
        let startTime = CFAbsoluteTimeGetCurrent()

        // Load audio
        let (audio, sampleCount) = try loadAudio(from: inputURL)
        let durationSeconds = Double(sampleCount) / config.sampleRate

        try checkCancelled()

        // Run separation
        let separated = try await separateChunked(
            audio,
            sampleCount: sampleCount,
            startTime: startTime,
            progressHandler: nil
        )

        try checkCancelled()

        // Write output
        let result = try writeOutput(
            vocals: separated,
            original: audio,
            to: outputURL,
            sampleRate: config.sampleRate,
            durationSeconds: durationSeconds,
            startTime: startTime
        )

        return result
    }

    // MARK: - Public API: AsyncThrowingStream

    /// Separate vocals with progress streaming.
    ///
    /// - Parameters:
    ///   - inputURL: Path to input audio file.
    ///   - outputURL: Directory for output files.
    /// - Returns: Stream of `RoFormerProgress` updates, yielding final result on completion.
    public func separationStream(
        from inputURL: URL,
        to outputURL: URL
    ) -> AsyncThrowingStream<RoFormerProgress, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    self.cancelFlag.withLock { $0 = false }
                    let startTime = CFAbsoluteTimeGetCurrent()

                    // Load audio
                    let (audio, sampleCount) = try self.loadAudio(from: inputURL)
                    let durationSeconds = Double(sampleCount) / self.config.sampleRate

                    continuation.yield(RoFormerProgress(
                        fraction: 0.05,
                        stage: .loading,
                        elapsedSeconds: CFAbsoluteTimeGetCurrent() - startTime
                    ))

                    try self.checkCancelled()

                    // Run separation with progress
                    let separated = try await self.separateChunked(
                        audio,
                        sampleCount: sampleCount,
                        startTime: startTime
                    ) { fraction, stage in
                        continuation.yield(RoFormerProgress(
                            fraction: fraction,
                            stage: stage,
                            elapsedSeconds: CFAbsoluteTimeGetCurrent() - startTime
                        ))
                    }

                    try self.checkCancelled()

                    // Write output
                    _ = try self.writeOutput(
                        vocals: separated,
                        original: audio,
                        to: outputURL,
                        sampleRate: self.config.sampleRate,
                        durationSeconds: durationSeconds,
                        startTime: startTime
                    )

                    continuation.yield(RoFormerProgress(
                        fraction: 1.0,
                        stage: .writing,
                        elapsedSeconds: CFAbsoluteTimeGetCurrent() - startTime
                    ))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                self.cancel()
            }
        }
    }

    // MARK: - Public API: Delegate (Fire-and-Forget)

    /// Separate vocals using delegate-based progress reporting.
    ///
    /// Returns immediately. Progress and completion are reported via ``delegate``.
    ///
    /// - Parameters:
    ///   - inputURL: Path to input audio file.
    ///   - outputURL: Directory for output files.
    public func separateVocals(from inputURL: URL, to outputURL: URL) {
        Task.detached {
            do {
                self.cancelFlag.withLock { $0 = false }
                let startTime = CFAbsoluteTimeGetCurrent()

                let (audio, sampleCount) = try self.loadAudio(from: inputURL)
                let durationSeconds = Double(sampleCount) / self.config.sampleRate

                await MainActor.run {
                    self.delegate?.roFormer(self, didUpdateProgress: 0.05, stage: .loading)
                }

                try self.checkCancelled()

                let separated = try await self.separateChunked(
                    audio,
                    sampleCount: sampleCount,
                    startTime: startTime
                ) { fraction, stage in
                    Task { @MainActor in
                        self.delegate?.roFormer(self, didUpdateProgress: fraction, stage: stage)
                    }
                }

                try self.checkCancelled()

                let result = try self.writeOutput(
                    vocals: separated,
                    original: audio,
                    to: outputURL,
                    sampleRate: self.config.sampleRate,
                    durationSeconds: durationSeconds,
                    startTime: startTime
                )

                await MainActor.run {
                    self.delegate?.roFormer(self, didCompleteWithResult: .success(result))
                }
            } catch let error as RoFormerError {
                await MainActor.run {
                    self.delegate?.roFormer(self, didCompleteWithResult: .failure(error))
                }
            } catch {
                await MainActor.run {
                    self.delegate?.roFormer(
                        self,
                        didCompleteWithResult: .failure(.mlxInferenceFailed(error))
                    )
                }
            }
        }
    }

    // MARK: - Public API: Raw Samples

    /// Separate vocals from raw audio samples (in-memory).
    ///
    /// - Parameters:
    ///   - samples: Stereo audio `[1, 2, samples]` at 44.1kHz.
    ///   - bodyDType: Transformer-body precision. `.float32` (default) or `.float16`
    ///     for the on-device shipping path (requires fp16 weights loaded and the
    ///     RMSNorm fp32-reduction patch). STFT/iSTFT always run in fp32.
    /// - Returns: Separated vocals `[1, 2, samples]`.
    /// - Throws: `RoFormerError` on failure or cancellation.
    public func separate(samples: MLXArray, bodyDType: DType = .float32) async throws -> MLXArray {
        cancelFlag.withLock { $0 = false }
        let startTime = CFAbsoluteTimeGetCurrent()
        return try await separateChunked(
            samples,
            sampleCount: samples.shape[2],
            startTime: startTime,
            bodyDType: bodyDType,
            progressHandler: nil
        )
    }

    // MARK: - Cancellation

    /// Cancel the current separation operation.
    ///
    /// Stops at the next checkpoint (between chunks). The operation will throw
    /// `RoFormerError.cancelled`.
    public func cancel() {
        cancelFlag.withLock { $0 = true }
    }

    // MARK: - Private: Chunked Processing

    private func separateChunked(
        _ audio: MLXArray,
        sampleCount: Int,
        startTime: CFAbsoluteTime,
        bodyDType: DType = .float32,
        progressHandler: ((Float, RoFormerStage) -> Void)?
    ) async throws -> MLXArray {
        let chunkSize = config.chunkSize   // 485100 (~11s)
        let step = config.chunkStep        // 352800 (~8s) → 3s overlap

        // Single-chunk fast path: the whole clip fits in one chunk. Run the model
        // directly (length-preserving) — no windowing/OLA needed. This also avoids the
        // negative write-start the reference would compute for sampleCount < chunkSize.
        if sampleCount <= chunkSize {
            progressHandler?(0.10, .stft)
            let result = model.forward(audio, bodyDType: bodyDType, capture: nil)
            MLX.eval(result)
            progressHandler?(0.90, .reconstruct)
            return result
        }

        // Multi-chunk overlap-add, faithful to the Python reference (step3_chunk.py):
        // per-sample symmetric-Hamming weights accumulate on CPU float32 buffers in the
        // same chunk order, then divide-by-counter. The window cancels exactly where a
        // sample is covered once and blends smoothly across the 3s overlap regions —
        // no amplitude modulation, no boundary click. CPU float32 mirrors numpy and
        // keeps result/counter as plain buffers (the future disk-streaming surface).
        let window = OverlapAdd.hammingWindow(chunkSize)
        let plan = OverlapAdd.chunkPlan(sampleCount: sampleCount, chunkSize: chunkSize, step: step)

        var resultL = [Float](repeating: 0, count: sampleCount)
        var resultR = [Float](repeating: 0, count: sampleCount)
        var counter = [Float](repeating: 0, count: sampleCount)

        for (chunkIdx, placement) in plan.enumerated() {
            try checkCancelled()

            let chunk = audio[0..., 0..., placement.readStart ..< (placement.readStart + chunkSize)]
            let separated = model.forward(chunk, bodyDType: bodyDType, capture: nil)
            MLX.eval(separated)

            let sep = separated[0]  // [2, chunkSize]
            let outL = sep[0].asType(.float32).asArray(Float.self)
            let outR = sep[1].asType(.float32).asArray(Float.self)
            let safe = min(placement.length, outL.count, window.count)
            let w = placement.writeStart

            for k in 0..<safe {
                let win = window[k]
                resultL[w + k] += outL[k] * win
                resultR[w + k] += outR[k] * win
                counter[w + k] += win
            }

            let frac = 0.10 + (Float(chunkIdx + 1) / Float(plan.count)) * 0.80
            progressHandler?(frac, .transformer)
        }

        // Normalize by accumulated window weight (np.clip(counter, 1e-10, None)).
        for n in 0..<sampleCount {
            let c = max(counter[n], Float(1e-10))
            resultL[n] /= c
            resultR[n] /= c
        }

        let out = stacked([MLXArray(resultL), MLXArray(resultR)], axis: 0)
            .expandedDimensions(axis: 0)  // [1, 2, sampleCount]
        MLX.eval(out)
        progressHandler?(0.90, .reconstruct)
        return out
    }

    // MARK: - Private: Helpers

    private func loadAudio(from url: URL) throws -> (MLXArray, Int) {
        do {
            return try audioIO.loadAudio(from: url)
        } catch {
            throw RoFormerError.audioReadFailed(error)
        }
    }

    private func writeOutput(
        vocals: MLXArray,
        original: MLXArray,
        to outputDirectory: URL,
        sampleRate: Double,
        durationSeconds: Double,
        startTime: CFAbsoluteTime
    ) throws -> StemResult {
        // Create output directory if needed
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )

        let vocalsURL = outputDirectory.appendingPathComponent("vocals.wav")
        let accompanimentURL = outputDirectory.appendingPathComponent("accompaniment.wav")

        do {
            // Write vocals
            try audioIO.saveAudio(vocals, to: vocalsURL, sampleRate: sampleRate)

            // Write accompaniment (original minus vocals)
            let accompaniment = original - vocals
            MLX.eval(accompaniment)
            try audioIO.saveAudio(accompaniment, to: accompanimentURL, sampleRate: sampleRate)
        } catch {
            throw RoFormerError.outputWriteFailed(error)
        }

        let inferenceTimeMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

        return StemResult(
            vocalsURL: vocalsURL,
            accompanimentURL: accompanimentURL,
            sampleRate: sampleRate,
            durationSeconds: durationSeconds,
            inferenceTimeMs: inferenceTimeMs
        )
    }

    private func checkCancelled() throws {
        if cancelFlag.withLock({ $0 }) {
            throw RoFormerError.cancelled
        }
    }
}
