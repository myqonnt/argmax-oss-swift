//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import XCTest
@testable import SpeakerKit

final class SpeakerDiarizationAudioSourceTests: XCTestCase {
    func testArrayAudioSourceClampsAndReadsRanges() throws {
        let samples = (0..<16).map(Float.init)
        let source = ArraySpeakerDiarizationAudioSource(samples: samples, sampleRate: 16_000)

        XCTAssertEqual(source.sampleRate, 16_000)
        XCTAssertEqual(source.sampleCount, samples.count)
        XCTAssertEqual(try source.readSamples(startSample: 4, endSample: 9), Array(samples[4..<9]))
        XCTAssertEqual(try source.readSamples(startSample: -4, endSample: 4), Array(samples[0..<4]))
        XCTAssertEqual(try source.readSamples(startSample: 8, endSample: 8), [])
    }
}
