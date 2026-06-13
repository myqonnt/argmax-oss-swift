//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

public extension WhisperKit {
    func makeSimulStreamingTranscriber(
        decodingOptions: DecodingOptions = DecodingOptions(),
        streamingOptions: SimulStreamingOptions = SimulStreamingOptions()
    ) throws -> SimulStreamingTranscriber {
        guard let tokenizer else {
            throw WhisperError.tokenizerUnavailable()
        }
        guard let concreteTextDecoder = textDecoder as? TextDecoder else {
            throw WhisperError.decodingFailed(
                "SimulStreamingTranscriber requires WhisperKit's standard TextDecoder because it reads per-token alignment weights."
            )
        }

        return SimulStreamingTranscriber(
            audioEncoder: audioEncoder,
            featureExtractor: featureExtractor,
            textDecoder: concreteTextDecoder,
            tokenizer: tokenizer,
            audioProcessor: audioProcessor,
            decodingOptions: decodingOptions,
            streamingOptions: streamingOptions
        )
    }
}

