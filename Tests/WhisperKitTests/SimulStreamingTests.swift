//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
@testable import WhisperKit
import XCTest

final class SimulStreamingTests: XCTestCase {
    func testNoSpeechProbabilityUsesSoftmax() throws {
        let logits = try MLMultiArray.logits([0.0, 1.0, 3.0])

        let probability = TextDecoder.probability(in: logits, token: 2)

        let expected = Float(exp(3.0) / (exp(0.0) + exp(1.0) + exp(3.0)))
        XCTAssertEqual(probability, expected, accuracy: 0.0001)
    }

    func testNoSpeechProbabilityReturnsZeroForInvalidToken() throws {
        let logits = try MLMultiArray.logits([0.0, 1.0, 3.0])

        XCTAssertEqual(TextDecoder.probability(in: logits, token: -1), 0)
        XCTAssertEqual(TextDecoder.probability(in: logits, token: logits.count), 0)
    }

    func testStreamingDecodingOptionsUsePromptTokensForHistory() {
        let baseOptions = DecodingOptions(
            temperatureFallbackCount: 4,
            withoutTimestamps: false,
            promptTokens: [1],
            prefixTokens: [2],
            suppressBlank: false
        )

        let options = SimulStreamingTranscriber.streamingDecodingOptions(
            from: baseOptions,
            contextTokens: [10, 11, 12, 13],
            maxContextTokens: 2
        )

        XCTAssertEqual(options.temperatureFallbackCount, 0)
        XCTAssertTrue(options.withoutTimestamps)
        XCTAssertTrue(options.suppressBlank)
        XCTAssertEqual(options.promptTokens, [12, 13])
        XCTAssertNil(options.prefixTokens)
    }

    func testContentFrameCountIsCappedAndRoundsUpToTimeTokenFrames() {
        XCTAssertEqual(
            SimulStreamingTranscriber.contentFrameCount(sampleCount: 0, maxFrameCount: 1500),
            0
        )
        XCTAssertEqual(
            SimulStreamingTranscriber.contentFrameCount(sampleCount: WhisperKit.sampleRate, maxFrameCount: 1500),
            50
        )
        XCTAssertEqual(
            SimulStreamingTranscriber.contentFrameCount(sampleCount: WhisperKit.sampleRate + 1, maxFrameCount: 1500),
            51
        )
        XCTAssertEqual(
            SimulStreamingTranscriber.contentFrameCount(sampleCount: Constants.defaultWindowSamples + 100, maxFrameCount: 1500),
            1500
        )
    }

    func testMostAttendedFrameIgnoresPaddedFrames() throws {
        let weights = try MLMultiArray.logits([0.1, 0.2, 0.7, 3.0, 4.0])

        XCTAssertEqual(TextDecoder.mostAttendedFrame(in: weights, maxFrameCount: 3), 2)
    }

    func testLimitedContextTokensUsesSuffix() {
        XCTAssertEqual(
            SimulStreamingTranscriber.limitedContextTokens([1, 2, 3, 4], maxContextTokens: 2),
            [3, 4]
        )
    }

    func testSimulStreamingUpdateCarriesNoSpeechProbability() {
        let update = SimulStreamingUpdate(isFinal: false, noSpeechProb: 0.92)

        XCTAssertEqual(update.noSpeechProb, 0.92)
    }
}
