//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

public struct SimulStreamingOptions: Sendable {
    public var minChunkSeconds: Float
    public var audioMinSeconds: Float
    public var audioMaxSeconds: Float
    public var frameThreshold: Int
    public var rewindThreshold: Int
    public var maxContextTokens: Int?
    public var trimLastWordWhenUnfinished: Bool

    public init(
        minChunkSeconds: Float = 1.0,
        audioMinSeconds: Float = 0.0,
        audioMaxSeconds: Float = 30.0,
        frameThreshold: Int = 25,
        rewindThreshold: Int = 200,
        maxContextTokens: Int? = nil,
        trimLastWordWhenUnfinished: Bool = true
    ) {
        self.minChunkSeconds = minChunkSeconds
        self.audioMinSeconds = audioMinSeconds
        self.audioMaxSeconds = audioMaxSeconds
        self.frameThreshold = frameThreshold
        self.rewindThreshold = rewindThreshold
        self.maxContextTokens = maxContextTokens
        self.trimLastWordWhenUnfinished = trimLastWordWhenUnfinished
    }
}

public struct SimulStreamingWord: Hashable, Codable, Sendable {
    public var start: Float
    public var end: Float
    public var text: String
    public var tokens: [Int]

    public init(start: Float, end: Float, text: String, tokens: [Int]) {
        self.start = start
        self.end = end
        self.text = text
        self.tokens = tokens
    }
}

public struct SimulStreamingUpdate: Sendable {
    public var start: Float?
    public var end: Float?
    public var text: String
    public var tokens: [Int]
    public var words: [SimulStreamingWord]
    public var isFinal: Bool
    public var noSpeechProb: Float?

    public init(
        start: Float? = nil,
        end: Float? = nil,
        text: String = "",
        tokens: [Int] = [],
        words: [SimulStreamingWord] = [],
        isFinal: Bool = false,
        noSpeechProb: Float? = nil
    ) {
        self.start = start
        self.end = end
        self.text = text
        self.tokens = tokens
        self.words = words
        self.isFinal = isFinal
        self.noSpeechProb = noSpeechProb
    }
}

public struct StreamingTokenPrediction: Sendable {
    public var token: Int
    public var logProb: Float
    public var mostAttendedFrame: Int?
    public var completed: Bool

    public init(token: Int, logProb: Float, mostAttendedFrame: Int?, completed: Bool) {
        self.token = token
        self.logProb = logProb
        self.mostAttendedFrame = mostAttendedFrame
        self.completed = completed
    }
}

public struct StreamingDecodingResult: Sendable {
    public var tokens: [Int]
    public var tokenLogProbs: [Float]
    public var tokenFrames: [Int]
    public var text: String
    public var completed: Bool
    public var stoppedNearAudioEnd: Bool
    public var lastAttendedFrame: Int?
    public var noSpeechProb: Float?

    public init(
        tokens: [Int],
        tokenLogProbs: [Float],
        tokenFrames: [Int],
        text: String,
        completed: Bool,
        stoppedNearAudioEnd: Bool,
        lastAttendedFrame: Int? = nil,
        noSpeechProb: Float? = nil
    ) {
        self.tokens = tokens
        self.tokenLogProbs = tokenLogProbs
        self.tokenFrames = tokenFrames
        self.text = text
        self.completed = completed
        self.stoppedNearAudioEnd = stoppedNearAudioEnd
        self.lastAttendedFrame = lastAttendedFrame
        self.noSpeechProb = noSpeechProb
    }
}
