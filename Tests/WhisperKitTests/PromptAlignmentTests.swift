import CoreML
@testable import WhisperKit
import XCTest

/// Exercises the real decode loop with synthetic alignment rows, without loading models.
final class PromptAlignmentTests: XCTestCase {
    func testRowsStartAtTranscriptForPromptAndPrefixCombinations() async throws {
        let cases: [(prompt: [Int]?, prefix: [Int]?)] = [
            (nil, nil), ([], nil), ([1000, 1001], nil),
            ([200, 201], nil), (Array(200..<220), nil),
            (nil, [300, 301]), ([200, 201], [300, 301])
        ]
        for item in cases {
            let (decoder, inputs, _, encoder, options) = try await MockTextDecoder.makePromptDecodingContext(
                promptTokens: item.prompt, prefixTokens: item.prefix
            )
            decoder.emitsAlignmentWeights = true
            try await assertRows(decoder: decoder, inputs: inputs, encoder: encoder, options: options)
        }
    }

    func testRowsRemainCorrectAfterFallbackOrWindowReset() async throws {
        let (decoder, inputs, _, encoder, options) = try await MockTextDecoder.makePromptDecodingContext(
            promptTokens: [200, 201], prefixTokens: [300]
        )
        decoder.emitsAlignmentWeights = true
        try await assertRows(decoder: decoder, inputs: inputs, encoder: encoder, options: options)
        // Fallback reuses the decoder buffers and the same prefill sequence.
        inputs.reset(maxTokenContext: 64)
        try await assertRows(decoder: decoder, inputs: inputs, encoder: encoder, options: options)
        // Another audio window may use a different effective prompt on the same buffers.
        inputs.reset(maxTokenContext: 64)
        var nextOptions = options
        nextOptions.promptTokens = nil
        nextOptions.prefixTokens = nil
        let nextInputs = try await decoder.prefillDecoderInputs(inputs, withOptions: nextOptions)
        try await assertRows(decoder: decoder, inputs: nextInputs, encoder: encoder, options: nextOptions)
    }

    private func assertRows(
        decoder: MockTextDecoder, inputs: any DecodingInputsType, encoder: MLMultiArray, options: DecodingOptions
    ) async throws {
        let tokens = try XCTUnwrap(decoder.tokenizer).specialTokens
        let start = try XCTUnwrap(inputs.initialPrompt.firstIndex(of: tokens.startOfTranscriptToken))
        let prefillSteps = inputs.initialPrompt.count - 1
        let baseCount = decoder.predictionCount
        // Earlier calls consumed script entries, so pad them before this pass's script.
        decoder.script = Array(repeating: .init(token: 100, confident: true), count: baseCount + prefillSteps)
            + [.init(token: 100, confident: true), .init(token: 101, confident: true),
               .init(token: tokens.endToken, confident: true)]
        let sampler = GreedyTokenSampler(temperature: 0, eotToken: tokens.endToken, decodingOptions: options)
        let result = try await decoder.decodeText(from: encoder, using: inputs, sampler: sampler, options: options)
        let weights = try XCTUnwrap(result.cache?.alignmentWeights)
        XCTAssertEqual(result.tokens.first, tokens.startOfTranscriptToken)
        XCTAssertEqual(Array(result.tokens.suffix(3)), [100, 101, tokens.endToken])
        XCTAssertEqual(weights[[0, 0]].intValue, 0, "Prompt-only steps must not populate transcript row zero")
        // Row zero has no preceding prediction. EOT terminates before its row is written.
        for row in 1..<(result.tokens.count - 1) {
            for column in [0, 750, 1499] {
                XCTAssertEqual(weights[[NSNumber(value: row), NSNumber(value: column)]].intValue,
                               baseCount + start + row, "Wrong alignment row for transcript token \(row)")
            }
        }
        // KV cache positions still refer to the full sequence, including conditioning prompt.
        XCTAssertEqual(inputs.cacheLength[0].intValue, inputs.initialPrompt.count + 1)
    }
}
