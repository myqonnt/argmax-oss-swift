//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.

import Foundation
import ArgmaxCore

/// Model manager specialised for diarization backends.
///
/// Backend-specific behaviour is provided by the ``ModelLoader`` passed at init,
/// while diarization is performed via a closure that handles the full pipeline internally.
///
/// Use the static factories to create instances:
/// - ``pyannote(config:segmenterModelInfo:embedderModelInfo:pldaModelInfo:downloader:)``
@available(macOS 13, iOS 16, watchOS 10, visionOS 1, *)
public class SpeakerKitDiarizer: ModelManager, Diarizer, @unchecked Sendable {
    /// Performs diarization on the provided audio using the loaded models.
    public let diarize: ([Float], (any DiarizationOptions)?, (@Sendable (Progress) -> Void)?) async throws -> DiarizationResult
    public let diarizeAudioSource: (any SpeakerDiarizationAudioSource, (any DiarizationOptions)?, (@Sendable (Progress) -> Void)?) async throws -> DiarizationResult

    /// - Parameters:
    ///   - loader: Backend-specific model loader.
    ///   - downloader: Downloader for model resolution.
    ///   - diarize: Performs diarization on audio and returns results.
    ///     When nil, auto-detects ``PyannoteModelLoader``.
    ///     Other implementations must provide this explicitly.
    public init(
        loader: ModelLoader,
        downloader: ModelDownloader,
        diarize: (([Float], (any DiarizationOptions)?, (@Sendable (Progress) -> Void)?) async throws -> DiarizationResult)? = nil,
        diarizeAudioSource: ((any SpeakerDiarizationAudioSource, (any DiarizationOptions)?, (@Sendable (Progress) -> Void)?) async throws -> DiarizationResult)? = nil
    ) {
        let resolvedDiarize: ([Float], (any DiarizationOptions)?, (@Sendable (Progress) -> Void)?) async throws -> DiarizationResult
        if let diarize {
            resolvedDiarize = diarize
        } else if let pyannoteLoader = loader as? PyannoteModelLoader {
            resolvedDiarize = { audioArray, options, progressCallback in
                let diarizer = try pyannoteLoader.makeDiarizer()
                return try await diarizer.diarize(audioArray: audioArray, options: options, progressCallback: progressCallback)
            }
        } else {
            preconditionFailure("diarize closure must be provided for non-Pyannote loaders")
        }
        self.diarize = resolvedDiarize

        if let diarizeAudioSource {
            self.diarizeAudioSource = diarizeAudioSource
        } else if let pyannoteLoader = loader as? PyannoteModelLoader {
            self.diarizeAudioSource = { audioSource, options, progressCallback in
                let diarizer = try pyannoteLoader.makeDiarizer()
                return try await diarizer.diarize(audioSource: audioSource, options: options, progressCallback: progressCallback)
            }
        } else {
            self.diarizeAudioSource = { audioSource, options, progressCallback in
                let samples = try audioSource.readSamples(startSample: 0, endSample: audioSource.sampleCount)
                return try await resolvedDiarize(samples, options, progressCallback)
            }
        }
        super.init(loader: loader, downloader: downloader)
    }

    public func downloadModels() async throws {
        try await super.downloadModels(progressCallback: nil)
    }

    public func downloadModels(progressCallback: (@Sendable (Progress) -> Void)?) async throws {
        try await super.downloadModels(progressCallback: progressCallback)
    }

    public func diarize(audioArray: [Float], options: (any DiarizationOptions)?, progressCallback: (@Sendable (Progress) -> Void)?) async throws -> DiarizationResult {
        return try await diarize(audioArray, options, progressCallback)
    }

    public func diarize(audioSource: any SpeakerDiarizationAudioSource, options: (any DiarizationOptions)?, progressCallback: (@Sendable (Progress) -> Void)?) async throws -> DiarizationResult {
        return try await diarizeAudioSource(audioSource, options, progressCallback)
    }
}
