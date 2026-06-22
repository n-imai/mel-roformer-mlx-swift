import Testing
@testable import SwiftRoFormer

@Test func kimVocal2Defaults() {
    let config = RoFormerConfiguration.kimVocal2
    #expect(config.dim == 384)
    #expect(config.depth == 6)
    #expect(config.heads == 8)
    #expect(config.dimHead == 64)
    #expect(config.dimInner == 512)
    #expect(config.ffDim == 1536)
    #expect(config.freqBins == 1025)
    #expect(config.nFFT == 2048)
    #expect(config.hopLength == 441)
    #expect(config.numBands == 60)
}

/// The OLA chunk recipe is validated ONLY for kim_vocal_2. The base default carries
/// kim's constants, so a preset that only overrides model dims (zfturbo) must NOT
/// silently inherit them — its STFT hop differs (512 vs 441), so the same chunk_size
/// would be a different chunk geometry. Pins the fix for Codex PR #2 P2.
@Test func chunkParamsAreKimSpecificNotSilentlyInherited() {
    // kim_vocal_2: the validated recipe (stft_hop 441 × (dim_t 1101 − 1)).
    #expect(RoFormerConfiguration.kimVocal2.chunkSize == 485_100)
    #expect(RoFormerConfiguration.kimVocal2.chunkStep == 352_800)
    // zfturbo must override, not inherit kim's 485100.
    #expect(RoFormerConfiguration.zfturboVocalsV1.chunkSize != 485_100)
    // zfturbo's chunk stays consistent with its OWN hop (512 × (1101 − 1)).
    #expect(RoFormerConfiguration.zfturboVocalsV1.chunkSize == 512 * (1101 - 1))
    #expect(RoFormerConfiguration.zfturboVocalsV1.chunkStep == 352_800)  // min(8×44100, chunkSize)
}
