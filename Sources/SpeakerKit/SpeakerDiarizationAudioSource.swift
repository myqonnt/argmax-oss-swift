//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// A seekable 16 kHz mono audio source for memory-bounded diarization.
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public protocol SpeakerDiarizationAudioSource: Sendable {
    /// The sample rate of the source. SpeakerKit expects 16 kHz mono PCM samples.
    var sampleRate: Int { get }

    /// The total number of samples available in the source.
    var sampleCount: Int { get }

    /// Reads samples in `[startSample, endSample)`.
    func readSamples(startSample: Int, endSample: Int) throws -> [Float]
}

@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
struct ArraySpeakerDiarizationAudioSource: SpeakerDiarizationAudioSource {
    let samples: [Float]
    let sampleRate: Int

    var sampleCount: Int { samples.count }

    func readSamples(startSample rawStart: Int, endSample rawEnd: Int) throws -> [Float] {
        let startSample = max(0, min(rawStart, samples.count))
        let endSample = max(0, min(rawEnd, samples.count))
        guard startSample < endSample else { return [] }
        return Array(samples[startSample..<endSample])
    }
}
