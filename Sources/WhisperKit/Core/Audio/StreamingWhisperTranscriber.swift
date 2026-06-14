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
    public var alignmentFrameMargin: Int
    public var alignmentRewindThreshold: Int
    public var boundaryFrameMargin: Int
    public var dropUnstableTrailingWord: Bool
    public var confirmationMode: StreamingConfirmationMode
    public var decodingOptions: DecodingOptions

    public init(
        minChunkSeconds: Float = 0.5,
        audioMinSeconds: Float = 1.0,
        audioMaxSeconds: Float = 28.0,
        holdBackWords: Int = 1,
        promptTokenLimit: Int = 224,
        alignmentFrameMargin: Int = 25,
        alignmentRewindThreshold: Int = 50,
        boundaryFrameMargin: Int? = nil,
        dropUnstableTrailingWord: Bool = true,
        confirmationMode: StreamingConfirmationMode = .localAgreement(rounds: 2),
        decodingOptions: DecodingOptions = DecodingOptions(wordTimestamps: true)
    ) {
        self.minChunkSeconds = minChunkSeconds
        self.audioMinSeconds = audioMinSeconds
        self.audioMaxSeconds = audioMaxSeconds
        self.holdBackWords = holdBackWords
        self.promptTokenLimit = promptTokenLimit
        self.alignmentFrameMargin = alignmentFrameMargin
        self.alignmentRewindThreshold = alignmentRewindThreshold
        self.boundaryFrameMargin = boundaryFrameMargin ?? alignmentFrameMargin
        self.dropUnstableTrailingWord = dropUnstableTrailingWord
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
    private var lastAttendedFrame: Int?

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
        lastAttendedFrame = nil
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
        guard !shouldRejectForAlignmentRewind(words) else {
            updateUnconfirmedSegments()
            return makeUpdate(isFinal: false)
        }
        updateLastAttendedFrame(with: words)

        let boundaryStableWords = dropUnstableTrailingWordIfNeeded(words)
        let committed = confirm(boundaryStableWords)
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
            if shouldRejectForAlignmentRewind(words) {
                Logging.debug("Rejecting final streaming hypothesis because alignment moved backward by more than \(options.alignmentRewindThreshold) frames")
            } else {
                updateLastAttendedFrame(with: words)
                hypothesisBuffer.insert(words)
            }
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
        lastAttendedFrame = nil
        return makeUpdate(newConfirmedSegments: newSegments, isFinal: true)
    }

    private func transcribeCurrentBuffer() async throws -> [StreamingWord] {
        var decodingOptions = options.decodingOptions
        applyDetectedLanguage(to: &decodingOptions)
        decodingOptions.wordTimestamps = supportsAlignmentAttention
        decodingOptions.promptTokens = makePromptTokens()
        decodingOptions.prefixTokens = makePrefixTokens()
        decodingOptions.alignmentEarlyStopping = usesAlignmentEarlyStopping
        decodingOptions.alignmentFrameMargin = options.alignmentFrameMargin

        let results = try await whisperKit.transcribe(
            audioArray: audioBuffer,
            decodeOptions: decodingOptions
        )
        if let language = results.first?.language, !language.isEmpty {
            detectedLanguage = language
        }

        let words = results.flatMap(\.segments).flatMap(streamingWords)

        if !words.isEmpty {
            return filterRepetitiveWords(words)
        }

        let segmentWords = results.flatMap(\.segments).compactMap { segment -> StreamingWord? in
            let start = Double(segment.start) + audioOffsetSeconds
            let end = Double(segment.end) + audioOffsetSeconds
            let text = segment.text
            guard end > start, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return nil
            }

            return StreamingWord(
                start: start,
                end: end,
                text: text,
                tokens: segment.tokens.filter { $0 < (whisperKit.tokenizer?.specialTokens.specialTokenBegin ?? Int.max) },
                alignmentFrames: segment.tokenAlignmentFrames.compactMap { $0 },
                probability: 0
            )
        }
        return filterRepetitiveWords(segmentWords)
    }

    private func streamingWords(from segment: TranscriptionSegment) -> [StreamingWord] {
        guard let wordTimings = segment.words, !wordTimings.isEmpty else { return [] }

        let specialTokenBegin = whisperKit.tokenizer?.specialTokens.specialTokenBegin ?? Int.max
        let segmentTextTokenFrames = zip(segment.tokens, normalizedAlignmentFrames(segment.tokenAlignmentFrames, count: segment.tokens.count))
            .filter { token, _ in token < specialTokenBegin }
            .map { _, frame in frame }
        var frameCursor = 0

        return wordTimings.compactMap { word -> StreamingWord? in
            let start = Double(word.start) + audioOffsetSeconds
            let end = Double(word.end) + audioOffsetSeconds
            guard end > start else { return nil }

            let wordTextTokenCount = word.tokens.filter { $0 < specialTokenBegin }.count
            let endCursor = min(segmentTextTokenFrames.count, frameCursor + wordTextTokenCount)
            let alignmentFrames = frameCursor < endCursor
                ? segmentTextTokenFrames[frameCursor..<endCursor].compactMap { $0 }
                : []
            frameCursor = endCursor

            return StreamingWord(
                start: start,
                end: end,
                text: word.word,
                tokens: word.tokens,
                alignmentFrames: alignmentFrames,
                probability: word.probability
            )
        }
    }

    private func normalizedAlignmentFrames(_ frames: [Int?], count: Int) -> [Int?] {
        if frames.count == count {
            return frames
        }
        if frames.count > count {
            return Array(frames.prefix(count))
        }
        return frames + Array(repeating: nil, count: count - frames.count)
    }

    private var supportsAlignmentAttention: Bool {
        whisperKit.textDecoder.supportsWordTimestamps
    }

    private var effectiveConfirmationMode: StreamingConfirmationMode {
        guard supportsAlignmentAttention else {
            return .localAgreement(rounds: 2)
        }
        return options.confirmationMode
    }

    private func applyDetectedLanguage(to decodingOptions: inout DecodingOptions) {
        guard decodingOptions.language == nil,
              let detectedLanguage,
              !detectedLanguage.isEmpty
        else { return }

        decodingOptions.language = detectedLanguage
        decodingOptions.detectLanguage = false
    }

    private var usesAlignmentEarlyStopping: Bool {
        switch effectiveConfirmationMode {
            case .localAgreement:
                return false
            case .alignmentAttention, .hybrid:
                return true
        }
    }

    private func confirm(_ words: [StreamingWord]) -> [StreamingWord] {
        switch effectiveConfirmationMode {
            case let .localAgreement(rounds):
                return confirmWithLocalAgreement(words, rounds: rounds)
            case .alignmentAttention:
                return confirmWithAlignmentAttention(words)
            case .hybrid:
                return confirmWithHybrid(words)
        }
    }

    private func confirmWithLocalAgreement(_ words: [StreamingWord], rounds: Int) -> [StreamingWord] {
        hypothesisBuffer.insert(words)
        return hypothesisBuffer.flush(
            holdBackWords: options.holdBackWords,
            agreementRounds: rounds
        )
    }

    private func confirmWithAlignmentAttention(_ words: [StreamingWord]) -> [StreamingWord] {
        hypothesisBuffer.insert(words)
        return hypothesisBuffer.flushStablePrefix(
            until: alignmentStableEndTime(),
            holdBackWords: options.holdBackWords
        )
    }

    private func confirmWithHybrid(_ words: [StreamingWord]) -> [StreamingWord] {
        let stableWords = words.filter { $0.end <= alignmentStableEndTime() }
        return confirmWithLocalAgreement(stableWords, rounds: 2)
    }

    private func alignmentStableEndTime() -> Double {
        let audioEnd = audioOffsetSeconds + Double(audioBuffer.count) / Double(WhisperKit.sampleRate)
        let margin = Double(max(0, options.alignmentFrameMargin)) * Double(WhisperKit.secondsPerTimeToken)
        return audioEnd - margin
    }

    private func dropUnstableTrailingWordIfNeeded(_ words: [StreamingWord]) -> [StreamingWord] {
        guard options.dropUnstableTrailingWord,
              words.count > 1,
              let lastWord = words.last,
              let lastWordFrame = lastWord.alignmentFrames.max()
        else {
            return words
        }

        let audioEndFrame = Int(
            (Double(audioBuffer.count) / Double(WhisperKit.sampleRate) / Double(WhisperKit.secondsPerTimeToken))
                .rounded(.up)
        )
        let margin = max(0, options.boundaryFrameMargin)
        guard audioEndFrame - lastWordFrame <= margin else {
            return words
        }

        Logging.debug("Dropping unstable trailing word \"\(lastWord.text)\" at frame \(lastWordFrame)/\(audioEndFrame)")
        return Array(words.dropLast())
    }

    private func shouldRejectForAlignmentRewind(_ words: [StreamingWord]) -> Bool {
        guard options.alignmentRewindThreshold > 0,
              let lastAttendedFrame,
              let currentFrame = estimatedLastAttendedFrame(from: words)
        else {
            return false
        }

        let rewind = lastAttendedFrame - currentFrame
        guard rewind > options.alignmentRewindThreshold else {
            return false
        }

        Logging.debug("Rejecting streaming hypothesis after alignment rewind of \(rewind) frames")
        return true
    }

    private func updateLastAttendedFrame(with words: [StreamingWord]) {
        guard let currentFrame = estimatedLastAttendedFrame(from: words) else { return }
        lastAttendedFrame = max(lastAttendedFrame ?? currentFrame, currentFrame)
    }

    private func estimatedLastAttendedFrame(from words: [StreamingWord]) -> Int? {
        let tracedFrames = words.flatMap(\.alignmentFrames)
        if let maxFrame = tracedFrames.max() {
            return maxFrame
        }

        guard let latestEnd = words.map(\.end).max() else { return nil }
        let relativeEnd = latestEnd - audioOffsetSeconds
        guard relativeEnd.isFinite, relativeEnd >= 0 else { return nil }

        return Int((relativeEnd / Double(WhisperKit.secondsPerTimeToken)).rounded())
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

    private func makePrefixTokens() -> [Int]? {
        guard options.promptTokenLimit > 0 else { return nil }

        let inWindowTokens = committedWords
            .filter { $0.end > audioOffsetSeconds }
            .flatMap(\.tokens)
            .filter { $0 < (whisperKit.tokenizer?.specialTokens.specialTokenBegin ?? Int.max) }

        let existingPrefix = options.decodingOptions.prefixTokens ?? []
        let combined = existingPrefix + inWindowTokens
        guard !combined.isEmpty else { return nil }

        return Array(combined.suffix(options.promptTokenLimit))
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
        trimLastAttendedFrame(removedSamples: samplesToRemove)
        audioOffsetSeconds += Double(samplesToRemove) / Double(WhisperKit.sampleRate)
        hypothesisBuffer.popCommitted(until: audioOffsetSeconds)
    }

    private func trimLastAttendedFrame(removedSamples: Int) {
        guard let lastAttendedFrame else { return }
        let removedSeconds = Double(removedSamples) / Double(WhisperKit.sampleRate)
        let removedFrames = Int((removedSeconds / Double(WhisperKit.secondsPerTimeToken)).rounded())
        self.lastAttendedFrame = max(0, lastAttendedFrame - removedFrames)
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
        let tokenAlignmentFrames = words.flatMap { word -> [Int?] in
            let frames = word.alignmentFrames.map(Optional.some)
            if frames.count == word.tokens.count {
                return frames
            }
            if frames.count > word.tokens.count {
                return Array(frames.prefix(word.tokens.count))
            }
            return frames + Array(repeating: nil, count: word.tokens.count - frames.count)
        }
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
            tokenAlignmentFrames: tokenAlignmentFrames,
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
    var alignmentFrames: [Int]
    var probability: Float
}

private struct StreamingLocalAgreementBuffer: Sendable {
    private var committedInBuffer: [StreamingWord] = []
    private var hypothesisHistory: [[StreamingWord]] = []
    private var lastCommittedTime: Double

    init(offset: Double = 0) {
        self.lastCommittedTime = offset
    }

    mutating func insert(_ words: [StreamingWord]) {
        hypothesisHistory.append(words.filter { $0.start > lastCommittedTime - 0.1 })
        removeDuplicatePrefixNearCommitBoundary()
    }

    mutating func flush(holdBackWords: Int, agreementRounds: Int = 2) -> [StreamingWord] {
        let requiredRounds = max(1, agreementRounds)
        let common: [StreamingWord]
        if requiredRounds == 1 {
            common = hypothesisHistory.last ?? []
        } else if hypothesisHistory.count >= requiredRounds {
            common = Self.longestCommonPrefix(Array(hypothesisHistory.suffix(requiredRounds)))
        } else {
            common = []
        }

        let committableCount = max(0, common.count - max(0, holdBackWords))
        let commit = Array(common.prefix(committableCount))

        for word in commit {
            lastCommittedTime = word.end
        }
        committedInBuffer.append(contentsOf: commit)

        hypothesisHistory = hypothesisHistory.map { Array($0.dropFirst(commit.count)) }

        let maxHistoryCount = max(1, requiredRounds - 1)
        if hypothesisHistory.count > maxHistoryCount {
            hypothesisHistory = Array(hypothesisHistory.suffix(maxHistoryCount))
        }
        return commit
    }

    mutating func flushStablePrefix(until stableEndTime: Double, holdBackWords: Int) -> [StreamingWord] {
        guard let latest = hypothesisHistory.last else { return [] }

        let stablePrefixCount = latest.prefix { $0.end <= stableEndTime }.count
        let committableCount = max(0, stablePrefixCount - max(0, holdBackWords))
        let commit = Array(latest.prefix(committableCount))

        for word in commit {
            lastCommittedTime = word.end
        }
        committedInBuffer.append(contentsOf: commit)

        hypothesisHistory = hypothesisHistory.map { Array($0.dropFirst(commit.count)) }
        if hypothesisHistory.count > 1 {
            hypothesisHistory = Array(hypothesisHistory.suffix(1))
        }
        return commit
    }

    mutating func popCommitted(until time: Double) {
        committedInBuffer.removeAll { $0.end <= time }
    }

    func unconfirmed() -> [StreamingWord] {
        hypothesisHistory.last ?? []
    }

    func complete() -> [StreamingWord] {
        hypothesisHistory.last ?? []
    }

    private mutating func removeDuplicatePrefixNearCommitBoundary() {
        guard let currentIndex = hypothesisHistory.indices.last,
              let first = hypothesisHistory[currentIndex].first,
              abs(first.start - lastCommittedTime) < 1,
              !committedInBuffer.isEmpty
        else { return }

        let maxNgram = min(min(committedInBuffer.count, hypothesisHistory[currentIndex].count), 5)
        guard maxNgram > 0 else { return }

        for size in 1...maxNgram {
            let committedTail = committedInBuffer
                .suffix(size)
                .map { Self.normalizedText($0.text) }
                .joined(separator: " ")
            let currentHead = hypothesisHistory[currentIndex]
                .prefix(size)
                .map { Self.normalizedText($0.text) }
                .joined(separator: " ")
            if committedTail == currentHead {
                hypothesisHistory[currentIndex].removeFirst(size)
                break
            }
        }
    }

    private static func longestCommonPrefix(_ hypotheses: [[StreamingWord]]) -> [StreamingWord] {
        guard var prefix = hypotheses.first else { return [] }
        for hypothesis in hypotheses.dropFirst() {
            prefix = longestCommonPrefix(prefix, hypothesis)
            if prefix.isEmpty {
                break
            }
        }
        return prefix
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
