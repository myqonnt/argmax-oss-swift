//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import ArgmaxCore
import CoreML
import Foundation

public typealias StreamingTokenCallback = @Sendable (StreamingTokenPrediction) -> Bool?

public extension TextDecoder {
    func decodeTextStreaming(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        contentFrameCount: Int,
        frameThreshold: Int,
        lastAttendedFrame: Int? = nil,
        rewindThreshold: Int? = nil,
        callback: StreamingTokenCallback? = nil
    ) async throws -> StreamingDecodingResult {
        guard let tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }
        guard let encoderOutput = encoderOutput as? MLMultiArray else {
            throw WhisperError.prepareDecoderInputsFailed("Input must be MLMultiArray")
        }
        guard let decoderInputs = decoderInputs as? DecodingInputs else {
            throw WhisperError.prepareDecoderInputsFailed("DecodingInputsType must be DecodingInputs")
        }

        let prefilledIndex = decoderInputs.cacheLength[0].intValue
        let initialPromptIndex = decoderInputs.initialPrompt.count
        var currentTokens = decoderInputs.initialPrompt
        var nextToken = decoderInputs.initialPrompt.last!
        var logProbs = Array(repeating: Float(0), count: currentTokens.count)
        var logitsFilters = createLogitsFilters(
            options: options,
            prefilledIndex: prefilledIndex,
            initialPromptIndex: initialPromptIndex,
            tokenizer: tokenizer
        )
        logitsFilters.append(
            SuppressTokensFilter(
                suppressTokens: Array(
                    Set([
                        tokenizer.specialTokens.startOfTranscriptToken,
                        tokenizer.specialTokens.startOfPreviousToken,
                        tokenizer.specialTokens.transcribeToken,
                        tokenizer.specialTokens.translateToken,
                        tokenizer.specialTokens.noTimestampsToken,
                        tokenizer.specialTokens.noSpeechToken,
                    ]).union(tokenizer.allLanguageTokens)
                )
            )
        )

        var generatedTokens: [Int] = []
        var generatedLogProbs: [Float] = []
        var generatedFrames: [Int] = []
        var completed = false
        var stoppedNearAudioEnd = false
        var lastAcceptedFrame = lastAttendedFrame
        var noSpeechProb: Float?

        let loopCount = min(options.sampleLength, Constants.maxTokenContext - 1)
        for tokenIndex in prefilledIndex..<loopCount {
            let isPrefill = tokenIndex < initialPromptIndex - 1
            let isLastPrefillToken = tokenIndex == initialPromptIndex - 1

            if tokenIndex < initialPromptIndex {
                let isTimestampToken = currentTokens[tokenIndex] >= tokenizer.specialTokens.timeTokenBegin
                let modelPredictedTimestamp = nextToken >= tokenizer.specialTokens.timeTokenBegin
                if !(isLastPrefillToken && isTimestampToken && modelPredictedTimestamp) {
                    nextToken = currentTokens[tokenIndex]
                } else {
                    currentTokens[tokenIndex] = nextToken
                }
            }

            decoderInputs.inputIds[0] = NSNumber(value: nextToken)
            decoderInputs.cacheLength[0] = NSNumber(value: tokenIndex)

            guard let decoderOutput = try await predictLogits(
                TextDecoderMLMultiArrayInputType(
                    inputIds: decoderInputs.inputIds,
                    cacheLength: decoderInputs.cacheLength,
                    keyCache: decoderInputs.keyCache,
                    valueCache: decoderInputs.valueCache,
                    kvCacheUpdateMask: decoderInputs.kvCacheUpdateMask,
                    encoderOutputEmbeds: encoderOutput,
                    decoderKeyPaddingMask: decoderInputs.decoderKeyPaddingMask
                )
            ) as? TextDecoderMLMultiArrayOutputType else {
                throw WhisperError.decodingLogitsFailed("Unable to decode logits")
            }

            var logits = decoderOutput.logits!
            if tokenIndex == prefilledIndex, let threshold = options.noSpeechThreshold {
                noSpeechProb = Self.probability(in: logits, token: tokenizer.specialTokens.noSpeechToken)
                if let noSpeechProb, noSpeechProb > threshold {
                    return StreamingDecodingResult(
                        tokens: [],
                        tokenLogProbs: [],
                        tokenFrames: [],
                        text: "",
                        completed: true,
                        stoppedNearAudioEnd: false,
                        lastAttendedFrame: lastAcceptedFrame,
                        noSpeechProb: noSpeechProb
                    )
                }
            }

            for filter in logitsFilters {
                logits = filter.filterLogits(logits, withTokens: currentTokens)
            }

            let sampleResult = await tokenSampler.update(tokens: currentTokens, logits: logits, logProbs: logProbs)
            nextToken = sampleResult.tokens.last!
            let nextTokenLogProb = sampleResult.logProbs.last!
            completed = sampleResult.completed

            guard let decoderCache = decoderOutput.cache,
                  let newKeyCache = decoderCache.keyCache,
                  let newValueCache = decoderCache.valueCache
            else {
                throw WhisperError.decodingLogitsFailed("Invalid model output cache")
            }

            TextDecoder.updateKVCache(
                keyTensor: decoderInputs.keyCache,
                keySlice: newKeyCache,
                valueTensor: decoderInputs.valueCache,
                valueSlice: newValueCache,
                insertAtIndex: tokenIndex
            )

            decoderInputs.decoderKeyPaddingMask[tokenIndex + 1] = 0
            decoderInputs.kvCacheUpdateMask[tokenIndex] = 0
            decoderInputs.kvCacheUpdateMask[tokenIndex + 1] = 1

            guard let newAlignmentWeights = decoderOutput.cache?.alignmentWeights else {
                throw WhisperError.decodingFailed(
                    "Streaming decoding requires decoder models that output alignment_heads_weights."
                )
            }
            let mostAttendedFrame = Self.mostAttendedFrame(in: newAlignmentWeights)
            TextDecoder.updateAlignmentWeights(
                alignmentTensor: decoderInputs.alignmentWeights,
                alignmentSlice: newAlignmentWeights,
                insertAtIndex: tokenIndex
            )

            if completed {
                break
            }

            if !isPrefill {
                if let previousAcceptedFrame = lastAcceptedFrame,
                   let rewindThreshold,
                   previousAcceptedFrame - mostAttendedFrame > rewindThreshold
                {
                    if currentTokens.last ?? 0 >= tokenizer.specialTokens.specialTokenBegin {
                        lastAcceptedFrame = mostAttendedFrame
                    } else {
                        break
                    }
                }

                if contentFrameCount - mostAttendedFrame <= frameThreshold {
                    stoppedNearAudioEnd = true
                    break
                }

                currentTokens.append(nextToken)
                logProbs.append(nextTokenLogProb)
                generatedTokens.append(nextToken)
                generatedLogProbs.append(nextTokenLogProb)
                generatedFrames.append(mostAttendedFrame)
                lastAcceptedFrame = mostAttendedFrame

                let prediction = StreamingTokenPrediction(
                    token: nextToken,
                    logProb: nextTokenLogProb,
                    mostAttendedFrame: mostAttendedFrame,
                    completed: completed
                )
                if let shouldContinue = callback?(prediction), !shouldContinue {
                    break
                }
            }
        }

        let wordTokens = generatedTokens.filter { $0 < tokenizer.specialTokens.specialTokenBegin }
        return StreamingDecodingResult(
            tokens: generatedTokens,
            tokenLogProbs: generatedLogProbs,
            tokenFrames: generatedFrames,
            text: tokenizer.decode(tokens: wordTokens),
            completed: completed,
            stoppedNearAudioEnd: stoppedNearAudioEnd,
            lastAttendedFrame: lastAcceptedFrame,
            noSpeechProb: noSpeechProb
        )
    }

    internal static func probability(in logits: MLMultiArray, token: Int) -> Float {
        guard token >= 0, token < logits.count else {
            return 0
        }

        let pointer = logits.dataPointer.assumingMemoryBound(to: FloatType.self)
        var maxLogit = -Float.infinity
        for index in 0..<logits.count {
            maxLogit = max(maxLogit, Float(pointer[index]))
        }

        var denominator: Float = 0
        for index in 0..<logits.count {
            denominator += exp(Float(pointer[index]) - maxLogit)
        }

        guard denominator > 0 else {
            return 0
        }
        return exp(Float(pointer[token]) - maxLogit) / denominator
    }

    private static func mostAttendedFrame(in alignmentWeights: MLMultiArray) -> Int {
        guard alignmentWeights.count > 0 else {
            return 0
        }

        let pointer = alignmentWeights.dataPointer.assumingMemoryBound(to: FloatType.self)
        var bestIndex = 0
        var bestValue = Float(pointer[0])

        for index in 1..<alignmentWeights.count {
            let value = Float(pointer[index])
            if value > bestValue {
                bestValue = value
                bestIndex = index
            }
        }

        let shape = alignmentWeights.shape.map(\.intValue)
        guard let lastDimension = shape.last, lastDimension > 0 else {
            return bestIndex
        }
        return bestIndex % lastDimension
    }
}
