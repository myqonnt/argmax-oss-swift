//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

extension WhisperKit: @unchecked Sendable {}

public enum StreamingConfirmationMode: Sendable {
    case localAgreement(rounds: Int = 2)
    case alignmentAttention
    case hybrid
}

public struct StreamingDecodingOptions: Sendable {
    public var minChunkSeconds: Float
    public var audioMinSeconds: Float
    public var audioMaxSeconds: Float
    public var holdBackWords: Int
    public var promptTokenLimit: Int
    public var confirmationMode: StreamingConfirmationMode
    public var decodingOptions: DecodingOptions

    public init(
        minChunkSeconds: Float = 0.5,
        audioMinSeconds: Float = 1.0,
        audioMaxSeconds: Float = 28.0,
        holdBackWords: Int = 1,
        promptTokenLimit: Int = 224,
        confirmationMode: StreamingConfirmationMode = .localAgreement(rounds: 2),
        decodingOptions: DecodingOptions = DecodingOptions(wordTimestamps: true)
    ) {
        self.minChunkSeconds = minChunkSeconds
        self.audioMinSeconds = audioMinSeconds
        self.audioMaxSeconds = audioMaxSeconds
        self.holdBackWords = holdBackWords
        self.promptTokenLimit = promptTokenLimit
        self.confirmationMode = confirmationMode
        self.decodingOptions = decodingOptions
        self.decodingOptions.wordTimestamps = true
    }
}

public struct StreamingTranscriptionUpdate: Sendable {
    public var newConfirmedSegments: [TranscriptionSegment]
    public var confirmedSegments: [TranscriptionSegment]
    public var unconfirmedSegments: [TranscriptionSegment]
    public var confirmedText: String
    public var unconfirmedText: String
    public var isFinal: Bool

    public init(
        newConfirmedSegments: [TranscriptionSegment] = [],
        confirmedSegments: [TranscriptionSegment] = [],
        unconfirmedSegments: [TranscriptionSegment] = [],
        confirmedText: String = "",
        unconfirmedText: String = "",
        isFinal: Bool = false
    ) {
        self.newConfirmedSegments = newConfirmedSegments
        self.confirmedSegments = confirmedSegments
        self.unconfirmedSegments = unconfirmedSegments
        self.confirmedText = confirmedText
        self.unconfirmedText = unconfirmedText
        self.isFinal = isFinal
    }
}

public actor StreamingWhisperTranscriber {
    private let whisperKit: WhisperKit
    private var options: StreamingDecodingOptions

    private var audioBuffer: [Float] = []
    private var audioOffsetSeconds: Double = 0
    private var pendingAudioSeconds: Double = 0
    private var hasReceivedAudio = false

    private var hypothesisBuffer = StreamingLocalAgreementBuffer()
    private var committedWords: [StreamingWord] = []
    private var confirmedSegmentsStorage: [TranscriptionSegment] = []
    private var unconfirmedSegmentsStorage: [TranscriptionSegment] = []
    private var detectedLanguage: String?

    public init(
        whisperKit: WhisperKit,
        options: StreamingDecodingOptions = StreamingDecodingOptions()
    ) {
        self.whisperKit = whisperKit
        self.options = options
    }

    public var confirmedSegments: [TranscriptionSegment] {
        confirmedSegmentsStorage
    }

    public var unconfirmedSegments: [TranscriptionSegment] {
        unconfirmedSegmentsStorage
    }

    public func reset(offsetSeconds: Double = 0) {
        audioBuffer.removeAll(keepingCapacity: true)
        audioOffsetSeconds = offsetSeconds
        pendingAudioSeconds = 0
        hasReceivedAudio = false
        hypothesisBuffer = StreamingLocalAgreementBuffer(offset: offsetSeconds)
        committedWords.removeAll(keepingCapacity: true)
        confirmedSegmentsStorage.removeAll(keepingCapacity: true)
        unconfirmedSegmentsStorage.removeAll(keepingCapacity: true)
        detectedLanguage = nil
    }

    public func insertAudioChunk(_ samples: [Float], startTime: Double? = nil) {
        guard !samples.isEmpty else { return }
        if !hasReceivedAudio {
            audioOffsetSeconds = startTime ?? audioOffsetSeconds
            hypothesisBuffer = StreamingLocalAgreementBuffer(offset: audioOffsetSeconds)
            hasReceivedAudio = true
        }

        audioBuffer.append(contentsOf: samples)
        pendingAudioSeconds += Double(samples.count) / Double(WhisperKit.sampleRate)
        trimAudioBufferIfNeeded()
    }

    public func processIter() async throws -> StreamingTranscriptionUpdate {
        guard !audioBuffer.isEmpty else {
            return makeUpdate(isFinal: false)
        }

        let audioSeconds = Double(audioBuffer.count) / Double(WhisperKit.sampleRate)
        guard pendingAudioSeconds >= Double(options.minChunkSeconds),
              audioSeconds >= Double(options.audioMinSeconds)
        else {
            return makeUpdate(isFinal: false)
        }
        pendingAudioSeconds = 0

        let words = try await transcribeCurrentBuffer()
        hypothesisBuffer.insert(words)
        let committed = hypothesisBuffer.flush(holdBackWords: options.holdBackWords)
        updateUnconfirmedSegments()

        guard !committed.isEmpty else {
            return makeUpdate(isFinal: false)
        }

        committedWords.append(contentsOf: committed)
        let newSegments = makeSegments(from: committed, isFinal: false)
        confirmedSegmentsStorage.append(contentsOf: newSegments)
        trimAudioBufferIfNeeded()
        return makeUpdate(newConfirmedSegments: newSegments, isFinal: false)
    }

    public func finish() async throws -> StreamingTranscriptionUpdate {
        if !audioBuffer.isEmpty {
            let words = try await transcribeCurrentBuffer()
            hypothesisBuffer.insert(words)
        }

        let remaining = hypothesisBuffer.complete()
        let newSegments: [TranscriptionSegment]
        if remaining.isEmpty {
            newSegments = []
        } else {
            committedWords.append(contentsOf: remaining)
            newSegments = makeSegments(from: remaining, isFinal: true)
            confirmedSegmentsStorage.append(contentsOf: newSegments)
        }

        unconfirmedSegmentsStorage.removeAll(keepingCapacity: true)
        audioBuffer.removeAll(keepingCapacity: true)
        pendingAudioSeconds = 0
        hasReceivedAudio = false
        audioOffsetSeconds = committedWords.last?.end ?? audioOffsetSeconds
        hypothesisBuffer = StreamingLocalAgreementBuffer(offset: audioOffsetSeconds)
        return makeUpdate(newConfirmedSegments: newSegments, isFinal: true)
    }

    private func transcribeCurrentBuffer() async throws -> [StreamingWord] {
        var decodingOptions = options.decodingOptions
        decodingOptions.wordTimestamps = true
        decodingOptions.promptTokens = makePromptTokens()

        let results = try await whisperKit.transcribe(
            audioArray: audioBuffer,
            decodeOptions: decodingOptions
        )
        if let language = results.first?.language, !language.isEmpty {
            detectedLanguage = language
        }

        let words: [StreamingWord] = results.flatMap(\.allWords).compactMap { word in
            let start = Double(word.start) + audioOffsetSeconds
            let end = Double(word.end) + audioOffsetSeconds
            guard end > start else { return nil }
            return StreamingWord(
                start: start,
                end: end,
                text: word.word,
                tokens: word.tokens,
                probability: word.probability
            )
        }
        return filterRepetitiveWords(words)
    }

    private func makePromptTokens() -> [Int]? {
        guard options.promptTokenLimit > 0 else { return nil }
        let oldTokens = committedWords
            .filter { $0.end <= audioOffsetSeconds }
            .flatMap(\.tokens)
            .filter { $0 < (whisperKit.tokenizer?.specialTokens.specialTokenBegin ?? Int.max) }
        guard !oldTokens.isEmpty else {
            return options.decodingOptions.promptTokens
        }

        let trimmed = Array(oldTokens.suffix(options.promptTokenLimit))
        if let promptTokens = options.decodingOptions.promptTokens, !promptTokens.isEmpty {
            return Array((promptTokens + trimmed).suffix(options.promptTokenLimit))
        }
        return trimmed
    }

    private func trimAudioBufferIfNeeded() {
        let maxSamples = max(1, Int(Double(options.audioMaxSeconds) * Double(WhisperKit.sampleRate)))
        guard audioBuffer.count > maxSamples else { return }

        let targetOffset: Double
        if let lastCommittedEnd = committedWords.last?.end,
           lastCommittedEnd > audioOffsetSeconds + 0.5
        {
            targetOffset = min(lastCommittedEnd, audioOffsetSeconds + Double(audioBuffer.count - maxSamples) / Double(WhisperKit.sampleRate))
        } else {
            targetOffset = audioOffsetSeconds + Double(audioBuffer.count - maxSamples) / Double(WhisperKit.sampleRate)
        }

        let samplesToRemove = min(
            audioBuffer.count - 1,
            max(0, Int((targetOffset - audioOffsetSeconds) * Double(WhisperKit.sampleRate)))
        )
        guard samplesToRemove > 0 else { return }

        audioBuffer.removeFirst(samplesToRemove)
        audioOffsetSeconds += Double(samplesToRemove) / Double(WhisperKit.sampleRate)
        hypothesisBuffer.popCommitted(until: audioOffsetSeconds)
    }

    private func updateUnconfirmedSegments() {
        let remaining = hypothesisBuffer.unconfirmed()
        unconfirmedSegmentsStorage = makeSegments(from: remaining, isFinal: false)
    }

    private func makeUpdate(
        newConfirmedSegments: [TranscriptionSegment] = [],
        isFinal: Bool
    ) -> StreamingTranscriptionUpdate {
        StreamingTranscriptionUpdate(
            newConfirmedSegments: newConfirmedSegments,
            confirmedSegments: confirmedSegmentsStorage,
            unconfirmedSegments: unconfirmedSegmentsStorage,
            confirmedText: confirmedSegmentsStorage.map(\.text).joined(),
            unconfirmedText: unconfirmedSegmentsStorage.map(\.text).joined(),
            isFinal: isFinal
        )
    }

    private func makeSegments(from words: [StreamingWord], isFinal: Bool) -> [TranscriptionSegment] {
        guard !words.isEmpty else { return [] }
        let text = words.map(\.text).joined().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }

        let tokens = words.flatMap(\.tokens)
        if let threshold = options.decodingOptions.compressionRatioThreshold,
           TextUtilities.compressionRatio(of: text) > threshold
        {
            return []
        }
        let segment = TranscriptionSegment(
            id: confirmedSegmentsStorage.count,
            seek: Int((words.first?.start ?? 0) * Double(WhisperKit.sampleRate)),
            start: Float(words.first?.start ?? 0),
            end: Float(words.last?.end ?? words.first?.start ?? 0),
            text: text,
            tokens: tokens,
            tokenLogProbs: tokens.map { [$0: Float(0)] },
            temperature: options.decodingOptions.temperature,
            words: words.map {
                WordTiming(
                    word: $0.text,
                    tokens: $0.tokens,
                    start: Float($0.start),
                    end: Float($0.end),
                    probability: $0.probability
                )
            }
        )
        return [segment]
    }

    private func filterRepetitiveWords(_ words: [StreamingWord]) -> [StreamingWord] {
        guard words.count >= 6 else { return words }
        let text = words.map(\.text).joined()
        if let threshold = options.decodingOptions.compressionRatioThreshold,
           TextUtilities.compressionRatio(of: text) <= threshold
        {
            return words
        }

        var filtered = words
        while filtered.count >= 6 {
            let tail = filtered.suffix(6).map { StreamingLocalAgreementBuffer.normalizedText($0.text) }
            let uniqueTail = Set(tail)
            if uniqueTail.count > 2 {
                break
            }
            filtered.removeLast()
        }

        return filtered
    }
}

private struct StreamingWord: Equatable, Sendable {
    var start: Double
    var end: Double
    var text: String
    var tokens: [Int]
    var probability: Float
}

private struct StreamingLocalAgreementBuffer: Sendable {
    private var committedInBuffer: [StreamingWord] = []
    private var previousHypothesis: [StreamingWord] = []
    private var currentHypothesis: [StreamingWord] = []
    private var lastCommittedTime: Double

    init(offset: Double = 0) {
        self.lastCommittedTime = offset
    }

    mutating func insert(_ words: [StreamingWord]) {
        currentHypothesis = words.filter { $0.start > lastCommittedTime - 0.1 }
        removeDuplicatePrefixNearCommitBoundary()
    }

    mutating func flush(holdBackWords: Int) -> [StreamingWord] {
        let common = Self.longestCommonPrefix(previousHypothesis, currentHypothesis)
        let committableCount = max(0, common.count - max(0, holdBackWords))
        let commit = Array(common.prefix(committableCount))

        for word in commit {
            lastCommittedTime = word.end
        }
        committedInBuffer.append(contentsOf: commit)

        previousHypothesis = Array(currentHypothesis.dropFirst(commit.count))
        currentHypothesis.removeAll(keepingCapacity: true)
        return commit
    }

    mutating func popCommitted(until time: Double) {
        committedInBuffer.removeAll { $0.end <= time }
    }

    func unconfirmed() -> [StreamingWord] {
        previousHypothesis
    }

    func complete() -> [StreamingWord] {
        currentHypothesis.isEmpty ? previousHypothesis : currentHypothesis
    }

    private mutating func removeDuplicatePrefixNearCommitBoundary() {
        guard let first = currentHypothesis.first,
              abs(first.start - lastCommittedTime) < 1,
              !committedInBuffer.isEmpty
        else { return }

        let maxNgram = min(min(committedInBuffer.count, currentHypothesis.count), 5)
        guard maxNgram > 0 else { return }

        for size in 1...maxNgram {
            let committedTail = committedInBuffer
                .suffix(size)
                .map { Self.normalizedText($0.text) }
                .joined(separator: " ")
            let currentHead = currentHypothesis
                .prefix(size)
                .map { Self.normalizedText($0.text) }
                .joined(separator: " ")
            if committedTail == currentHead {
                currentHypothesis.removeFirst(size)
                break
            }
        }
    }

    private static func longestCommonPrefix(_ lhs: [StreamingWord], _ rhs: [StreamingWord]) -> [StreamingWord] {
        var prefix: [StreamingWord] = []
        for (left, right) in zip(lhs, rhs) {
            guard normalizedText(left.text) == normalizedText(right.text) else {
                break
            }
            prefix.append(right)
        }
        return prefix
    }

    fileprivate static func normalizedText(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil)
    }
}

public extension WhisperKit {
    func makeStreamingTranscriber(
        options: StreamingDecodingOptions = StreamingDecodingOptions()
    ) -> StreamingWhisperTranscriber {
        StreamingWhisperTranscriber(whisperKit: self, options: options)
    }
}
