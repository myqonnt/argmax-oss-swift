//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

extension VoiceActivityDetector: @unchecked Sendable {}

public enum VoiceActivityControlledStreamingState: Sendable {
    case nonVoice
    case voice
}

public struct VoiceActivityControlOptions: Sendable {
    public var speechPaddingSeconds: Float
    public var minSilenceSeconds: Float
    public var processDuringSpeech: Bool

    public init(
        speechPaddingSeconds: Float = 0.2,
        minSilenceSeconds: Float = 0.5,
        processDuringSpeech: Bool = true
    ) {
        self.speechPaddingSeconds = speechPaddingSeconds
        self.minSilenceSeconds = minSilenceSeconds
        self.processDuringSpeech = processDuringSpeech
    }
}

public actor VoiceActivityControlledStreamingTranscriber {
    private let transcriber: StreamingWhisperTranscriber
    private let vad: VoiceActivityDetector
    private let options: VoiceActivityControlOptions

    private var stateStorage: VoiceActivityControlledStreamingState = .nonVoice
    private var absoluteSampleCursor: Int = 0
    private var trailingSilenceSamples: Int = 0
    private var preSpeechBuffer: [Float] = []
    private var confirmedSegmentsStorage: [TranscriptionSegment] = []
    private var unconfirmedSegmentsStorage: [TranscriptionSegment] = []

    public init(
        transcriber: StreamingWhisperTranscriber,
        vad: VoiceActivityDetector = EnergyVAD(),
        options: VoiceActivityControlOptions = VoiceActivityControlOptions()
    ) {
        self.transcriber = transcriber
        self.vad = vad
        self.options = options
    }

    public init(
        whisperKit: WhisperKit,
        streamingOptions: StreamingDecodingOptions = StreamingDecodingOptions(),
        vad: VoiceActivityDetector = EnergyVAD(),
        options: VoiceActivityControlOptions = VoiceActivityControlOptions()
    ) {
        self.transcriber = StreamingWhisperTranscriber(whisperKit: whisperKit, options: streamingOptions)
        self.vad = vad
        self.options = options
    }

    public var state: VoiceActivityControlledStreamingState {
        stateStorage
    }

    public var confirmedSegments: [TranscriptionSegment] {
        confirmedSegmentsStorage
    }

    public var unconfirmedSegments: [TranscriptionSegment] {
        unconfirmedSegmentsStorage
    }

    public func reset(offsetSeconds: Double = 0) async {
        await transcriber.reset(offsetSeconds: offsetSeconds)
        stateStorage = .nonVoice
        absoluteSampleCursor = max(0, Int((offsetSeconds * Double(WhisperKit.sampleRate)).rounded()))
        trailingSilenceSamples = 0
        preSpeechBuffer.removeAll(keepingCapacity: true)
        confirmedSegmentsStorage.removeAll(keepingCapacity: true)
        unconfirmedSegmentsStorage.removeAll(keepingCapacity: true)
    }

    public func processAudioChunk(
        _ samples: [Float],
        startTime: Double? = nil
    ) async throws -> [StreamingTranscriptionUpdate] {
        guard !samples.isEmpty else { return [] }

        let chunkStartSample = startTime.map {
            max(0, Int(($0 * Double(WhisperKit.sampleRate)).rounded()))
        } ?? absoluteSampleCursor
        absoluteSampleCursor = chunkStartSample + samples.count

        let activity = try await vad.voiceActivityAsync(in: samples)
        guard !activity.isEmpty else {
            rememberNonVoiceSamples(samples)
            return []
        }

        var updates: [StreamingTranscriptionUpdate] = []
        var sendStartIndex: Int? = stateStorage == .voice ? 0 : nil
        var nonVoicePaddingSamples: [Float] = []
        let frameLength = max(1, vad.frameLengthSamples)
        let silenceThresholdSamples = max(1, Int(Double(options.minSilenceSeconds) * Double(WhisperKit.sampleRate)))
        let speechPaddingSamples = max(0, Int(Double(options.speechPaddingSeconds) * Double(WhisperKit.sampleRate)))

        for (frameIndex, hasVoice) in activity.enumerated() {
            let frameStart = min(samples.count, frameIndex * frameLength)
            let frameEnd = min(samples.count, frameStart + frameLength)
            guard frameEnd > frameStart else { continue }

            if hasVoice {
                if stateStorage == .nonVoice {
                    let currentPadding = min(speechPaddingSamples, frameStart)
                    let priorPadding = min(speechPaddingSamples - currentPadding, preSpeechBuffer.count)
                    let startIndex = frameStart - currentPadding
                    let startSample = chunkStartSample + startIndex - priorPadding

                    await transcriber.reset(
                        offsetSeconds: Double(max(0, startSample)) / Double(WhisperKit.sampleRate)
                    )
                    if priorPadding > 0 {
                        await transcriber.insertAudioChunk(
                            Array(preSpeechBuffer.suffix(priorPadding)),
                            startTime: Double(max(0, startSample)) / Double(WhisperKit.sampleRate)
                        )
                    }

                    preSpeechBuffer.removeAll(keepingCapacity: true)
                    nonVoicePaddingSamples.removeAll(keepingCapacity: true)
                    sendStartIndex = startIndex
                    trailingSilenceSamples = 0
                    stateStorage = .voice
                } else {
                    trailingSilenceSamples = 0
                }
            } else if stateStorage == .voice {
                trailingSilenceSamples += frameEnd - frameStart
                if trailingSilenceSamples >= silenceThresholdSamples {
                    let endpoint = frameEnd
                    if let start = sendStartIndex, endpoint > start {
                        await transcriber.insertAudioChunk(
                            Array(samples[start..<endpoint]),
                            startTime: Double(chunkStartSample + start) / Double(WhisperKit.sampleRate)
                        )
                    }

                    let finished = try await transcriber.finish()
                    updates.append(mergeUpdate(finished))

                    sendStartIndex = nil
                    trailingSilenceSamples = 0
                    stateStorage = .nonVoice
                    if endpoint < samples.count {
                        nonVoicePaddingSamples.append(contentsOf: samples[endpoint..<frameEnd])
                    }
                }
            } else {
                nonVoicePaddingSamples.append(contentsOf: samples[frameStart..<frameEnd])
            }
        }

        if stateStorage == .voice, let start = sendStartIndex, samples.count > start {
            await transcriber.insertAudioChunk(
                Array(samples[start..<samples.count]),
                startTime: Double(chunkStartSample + start) / Double(WhisperKit.sampleRate)
            )
            if options.processDuringSpeech {
                let update = try await transcriber.processIter()
                updates.append(mergeUpdate(update))
            }
        } else if stateStorage == .nonVoice {
            rememberNonVoiceSamples(nonVoicePaddingSamples)
        }

        return updates
    }

    public func finish() async throws -> StreamingTranscriptionUpdate {
        let update: StreamingTranscriptionUpdate
        if stateStorage == .voice {
            update = try await transcriber.finish()
        } else {
            update = StreamingTranscriptionUpdate(
                confirmedSegments: confirmedSegmentsStorage,
                unconfirmedSegments: [],
                confirmedText: confirmedSegmentsStorage.map(\.text).joined(),
                isFinal: true
            )
        }

        stateStorage = .nonVoice
        trailingSilenceSamples = 0
        preSpeechBuffer.removeAll(keepingCapacity: true)
        return mergeUpdate(update)
    }

    private func rememberNonVoiceSamples(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        let maxPaddingSamples = max(0, Int(Double(options.speechPaddingSeconds) * Double(WhisperKit.sampleRate)))
        guard maxPaddingSamples > 0 else {
            preSpeechBuffer.removeAll(keepingCapacity: true)
            return
        }

        preSpeechBuffer.append(contentsOf: samples)
        if preSpeechBuffer.count > maxPaddingSamples {
            preSpeechBuffer.removeFirst(preSpeechBuffer.count - maxPaddingSamples)
        }
    }

    private func mergeUpdate(_ update: StreamingTranscriptionUpdate) -> StreamingTranscriptionUpdate {
        let firstNewSegmentID = confirmedSegmentsStorage.count
        let newSegments = update.newConfirmedSegments.enumerated().map { offset, segment in
            reindexedSegment(segment, id: firstNewSegmentID + offset)
        }
        confirmedSegmentsStorage.append(contentsOf: newSegments)
        unconfirmedSegmentsStorage = update.unconfirmedSegments.enumerated().map { offset, segment in
            reindexedSegment(segment, id: confirmedSegmentsStorage.count + offset)
        }

        return StreamingTranscriptionUpdate(
            newConfirmedSegments: newSegments,
            confirmedSegments: confirmedSegmentsStorage,
            unconfirmedSegments: unconfirmedSegmentsStorage,
            confirmedText: confirmedSegmentsStorage.map(\.text).joined(),
            unconfirmedText: unconfirmedSegmentsStorage.map(\.text).joined(),
            isFinal: update.isFinal
        )
    }

    private func reindexedSegment(_ segment: TranscriptionSegment, id: Int) -> TranscriptionSegment {
        var reindexed = segment
        reindexed.id = id
        return reindexed
    }
}

public extension WhisperKit {
    func makeVoiceActivityControlledStreamingTranscriber(
        streamingOptions: StreamingDecodingOptions = StreamingDecodingOptions(),
        vad: VoiceActivityDetector = EnergyVAD(),
        options: VoiceActivityControlOptions = VoiceActivityControlOptions()
    ) -> VoiceActivityControlledStreamingTranscriber {
        VoiceActivityControlledStreamingTranscriber(
            whisperKit: self,
            streamingOptions: streamingOptions,
            vad: vad,
            options: options
        )
    }
}
