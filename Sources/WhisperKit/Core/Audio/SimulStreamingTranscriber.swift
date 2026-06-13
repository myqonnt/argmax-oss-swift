//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation

/// Stateful SimulStreaming-style transcriber. Feed chunks and call `process()` serially from one task/queue.
public final class SimulStreamingTranscriber {
    private let audioEncoder: any AudioEncoding
    private let featureExtractor: any FeatureExtracting
    private let textDecoder: TextDecoder
    private let tokenizer: any WhisperTokenizer
    private let audioProcessor: any AudioProcessing
    private let baseDecodingOptions: DecodingOptions
    private let streamingOptions: SimulStreamingOptions

    private var audioBuffer: [Float] = []
    private var pendingSampleCount = 0
    private var audioBufferOffsetSeconds: Float = 0
    private var contextTokens: [Int] = []
    private var lastTimestamp: Float = -1
    private var lastAttendedFrame: Int?
    private var incompleteUnicodeTokens: [Int] = []

    public init(
        audioEncoder: any AudioEncoding,
        featureExtractor: any FeatureExtracting,
        textDecoder: TextDecoder,
        tokenizer: any WhisperTokenizer,
        audioProcessor: any AudioProcessing,
        decodingOptions: DecodingOptions,
        streamingOptions: SimulStreamingOptions = SimulStreamingOptions()
    ) {
        self.audioEncoder = audioEncoder
        self.featureExtractor = featureExtractor
        self.textDecoder = textDecoder
        self.tokenizer = tokenizer
        self.audioProcessor = audioProcessor
        self.baseDecodingOptions = decodingOptions
        self.streamingOptions = streamingOptions
    }

    public func reset(offsetSeconds: Float = 0) {
        audioBuffer = []
        pendingSampleCount = 0
        audioBufferOffsetSeconds = offsetSeconds
        contextTokens = []
        lastTimestamp = offsetSeconds - 0.001
        lastAttendedFrame = nil
        incompleteUnicodeTokens = []
    }

    public func insertAudioChunk(_ samples: [Float]) {
        guard !samples.isEmpty else {
            return
        }
        audioBuffer.append(contentsOf: samples)
        pendingSampleCount += samples.count
        trimAudioBufferIfNeeded()
    }

    public func process() async throws -> SimulStreamingUpdate {
        let pendingSeconds = Float(pendingSampleCount) / Float(WhisperKit.sampleRate)
        guard pendingSeconds >= streamingOptions.minChunkSeconds else {
            return SimulStreamingUpdate()
        }
        pendingSampleCount = 0
        return try await decode(isFinal: false)
    }

    public func finish() async throws -> SimulStreamingUpdate {
        pendingSampleCount = 0
        let nextOffsetSeconds = audioBufferOffsetSeconds + Float(audioBuffer.count) / Float(WhisperKit.sampleRate)
        let update = try await decode(isFinal: true)
        reset(offsetSeconds: nextOffsetSeconds)
        return update
    }

    private func decode(isFinal: Bool) async throws -> SimulStreamingUpdate {
        guard !audioBuffer.isEmpty else {
            return SimulStreamingUpdate(isFinal: isFinal)
        }

        let bufferedSeconds = Float(audioBuffer.count) / Float(WhisperKit.sampleRate)
        guard isFinal || bufferedSeconds >= streamingOptions.audioMinSeconds else {
            return SimulStreamingUpdate(isFinal: false)
        }

        let windowSamples = featureExtractor.windowSamples ?? Constants.defaultWindowSamples
        guard let audioSamples = audioProcessor.padOrTrim(fromArray: audioBuffer, startAt: 0, toLength: windowSamples) else {
            throw WhisperError.audioProcessingFailed("Audio samples are nil")
        }
        guard let melOutput = try await featureExtractor.logMelSpectrogram(fromAudio: audioSamples) else {
            throw WhisperError.transcriptionFailed("Mel output is nil")
        }
        guard let encoderOutput = try await audioEncoder.encodeFeatures(melOutput) else {
            throw WhisperError.transcriptionFailed("Encoder output is nil")
        }

        let options = Self.streamingDecodingOptions(
            from: baseDecodingOptions,
            contextTokens: contextTokens,
            maxContextTokens: streamingOptions.maxContextTokens
        )

        var decoderInputs = try textDecoder.prepareDecoderInputs(withPrompt: [tokenizer.specialTokens.startOfTranscriptToken])
        if options.usePrefillPrompt {
            decoderInputs = try await textDecoder.prefillDecoderInputs(decoderInputs, withOptions: options)
        }

        let contentFrameCount = Self.contentFrameCount(
            sampleCount: audioBuffer.count,
            maxFrameCount: textDecoder.windowSize ?? 1500
        )
        let frameThreshold = isFinal ? 4 : streamingOptions.frameThreshold
        let sampler = GreedyTokenSampler(
            temperature: FloatType(options.temperature),
            eotToken: tokenizer.specialTokens.endToken,
            decodingOptions: options
        )
        let result = try await textDecoder.decodeTextStreaming(
            from: encoderOutput,
            using: decoderInputs,
            sampler: sampler,
            options: options,
            contentFrameCount: contentFrameCount,
            frameThreshold: frameThreshold,
            lastAttendedFrame: lastAttendedFrame,
            rewindThreshold: streamingOptions.rewindThreshold
        )
        lastAttendedFrame = result.lastAttendedFrame

        var tokens = hideIncompleteUnicode(in: result.tokens)
        var tokenFrames = Array(result.tokenFrames.prefix(tokens.count))

        if streamingOptions.trimLastWordWhenUnfinished, !isFinal, result.stoppedNearAudioEnd {
            (tokens, tokenFrames) = trimLastWord(tokens: tokens, tokenFrames: tokenFrames)
        }

        guard !tokens.isEmpty else {
            return SimulStreamingUpdate(
                isFinal: isFinal && result.completed,
                noSpeechProb: result.noSpeechProb,
                stopReason: result.stopReason,
                lastSampledToken: result.lastSampledToken,
                lastSampledLogProb: result.lastSampledLogProb,
                lastSampledFrame: result.lastSampledFrame
            )
        }

        contextTokens = limitedContextTokens(contextTokens + tokens)
        let words = timestampedWords(tokens: tokens, tokenFrames: tokenFrames)
        let text = tokenizer.decode(tokens: tokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin })
        let start = words.first?.start
        let end = max(words.last?.end ?? start ?? audioBufferOffsetSeconds, (start ?? audioBufferOffsetSeconds) + 0.001)

        lastTimestamp = end

        return SimulStreamingUpdate(
            start: start,
            end: end,
            text: text,
            tokens: tokens,
            words: words,
            isFinal: isFinal,
            noSpeechProb: result.noSpeechProb,
            stopReason: result.stopReason,
            lastSampledToken: result.lastSampledToken,
            lastSampledLogProb: result.lastSampledLogProb,
            lastSampledFrame: result.lastSampledFrame
        )
    }

    private func trimAudioBufferIfNeeded() {
        let maxSamples = max(1, Int(streamingOptions.audioMaxSeconds * Float(WhisperKit.sampleRate)))
        guard audioBuffer.count > maxSamples else {
            return
        }

        let removeCount = audioBuffer.count - maxSamples
        audioBuffer.removeFirst(removeCount)
        audioBufferOffsetSeconds += Float(removeCount) / Float(WhisperKit.sampleRate)
        if let frame = lastAttendedFrame {
            let removedFrames = Int(
                round(Float(removeCount) / Float(WhisperKit.sampleRate) / WhisperKit.secondsPerTimeToken)
            )
            lastAttendedFrame = max(0, frame - removedFrames)
        }
    }

    private func limitedContextTokens(_ tokens: [Int]) -> [Int] {
        Self.limitedContextTokens(tokens, maxContextTokens: streamingOptions.maxContextTokens)
    }

    internal static func streamingDecodingOptions(
        from baseOptions: DecodingOptions,
        contextTokens: [Int],
        maxContextTokens: Int?
    ) -> DecodingOptions {
        var options = baseOptions
        options.temperatureFallbackCount = 0
        options.withoutTimestamps = true
        options.suppressBlank = true
        options.promptTokens = limitedContextTokens(contextTokens, maxContextTokens: maxContextTokens)
        options.prefixTokens = nil
        return options
    }

    internal static func limitedContextTokens(_ tokens: [Int], maxContextTokens: Int?) -> [Int] {
        let maxTokens = maxContextTokens ?? Constants.maxTokenContext / 2
        return Array(tokens.suffix(maxTokens))
    }

    internal static func contentFrameCount(sampleCount: Int, maxFrameCount: Int) -> Int {
        guard sampleCount > 0, maxFrameCount > 0 else {
            return 0
        }
        let seconds = Double(sampleCount) / Double(WhisperKit.sampleRate)
        let frames = Int(((seconds / Double(WhisperKit.secondsPerTimeToken)) - 1e-4).rounded(.up))
        return min(frames, maxFrameCount)
    }

    private func hideIncompleteUnicode(in tokens: [Int]) -> [Int] {
        var mergedTokens = incompleteUnicodeTokens + tokens
        incompleteUnicodeTokens = []

        guard !mergedTokens.isEmpty else {
            return mergedTokens
        }

        let decoded = tokenizer.decode(tokens: mergedTokens)
        if decoded.hasSuffix("\u{fffd}") {
            incompleteUnicodeTokens = [mergedTokens.removeLast()]
        }
        return mergedTokens
    }

    private func trimLastWord(tokens: [Int], tokenFrames: [Int]) -> ([Int], [Int]) {
        let split = tokenizer.splitToWordTokens(tokenIds: tokens)
        guard split.wordTokens.count > 1 else {
            return (tokens, tokenFrames)
        }

        let keepCount = split.wordTokens.dropLast().reduce(0) { $0 + $1.count }
        return (
            Array(tokens.prefix(keepCount)),
            Array(tokenFrames.prefix(keepCount))
        )
    }

    private func timestampedWords(tokens: [Int], tokenFrames: [Int]) -> [SimulStreamingWord] {
        let split = tokenizer.splitToWordTokens(tokenIds: tokens)
        var remainingFrames = tokenFrames
        var words: [SimulStreamingWord] = []

        for (word, wordTokens) in zip(split.words, split.wordTokens) {
            let count = wordTokens.count
            let frames = Array(remainingFrames.prefix(count))
            remainingFrames.removeFirst(min(count, remainingFrames.count))

            let startFrame = frames.first ?? 0
            let endFrame = frames.last ?? startFrame
            let start = max(audioBufferOffsetSeconds + Float(startFrame) * WhisperKit.secondsPerTimeToken, lastTimestamp + 0.001)
            let end = max(audioBufferOffsetSeconds + Float(endFrame) * WhisperKit.secondsPerTimeToken, start + 0.001)
            words.append(
                SimulStreamingWord(
                    start: start,
                    end: end,
                    text: word,
                    tokens: wordTokens
                )
            )
        }

        return words
    }
}
