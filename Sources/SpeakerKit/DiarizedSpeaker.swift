//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation

/// Describes how a speaker voiceprint was derived.
public enum SpeakerVoiceprintSource: String, Sendable {
    /// Mean-pooled Pyannote window embeddings for a clustered speaker.
    case pyannoteWindowMean
}

/// A model-derived speaker feature vector for similarity comparison.
///
/// Voiceprints are not stable speaker identities. They are embeddings produced
/// for the current model and audio conditions, and are best used with similarity
/// thresholds calibrated for the target use case.
public struct SpeakerVoiceprint: Sendable {
    public let embedding: [Float]
    public let pldaEmbedding: [Float]?
    public let sampleCount: Int
    public let source: SpeakerVoiceprintSource

    public init(
        embedding: [Float],
        pldaEmbedding: [Float]? = nil,
        sampleCount: Int,
        source: SpeakerVoiceprintSource
    ) {
        self.embedding = embedding
        self.pldaEmbedding = pldaEmbedding
        self.sampleCount = sampleCount
        self.source = source
    }
}

/// Summary information for a diarized speaker in a single result.
public struct DiarizedSpeaker: Identifiable, Sendable {
    public let id: Int
    public let voiceprint: SpeakerVoiceprint?
    public let segmentCount: Int
    public let speakingDuration: Float

    public init(
        id: Int,
        voiceprint: SpeakerVoiceprint? = nil,
        segmentCount: Int,
        speakingDuration: Float
    ) {
        self.id = id
        self.voiceprint = voiceprint
        self.segmentCount = segmentCount
        self.speakingDuration = speakingDuration
    }

    func replacingStats(segmentCount: Int, speakingDuration: Float) -> DiarizedSpeaker {
        DiarizedSpeaker(
            id: id,
            voiceprint: voiceprint,
            segmentCount: segmentCount,
            speakingDuration: speakingDuration
        )
    }
}
