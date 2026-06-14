//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

/// Configuration to initialize WhisperKit
open class WhisperKitConfig {
    /// Name for whisper model to use
    public var model: String?
    /// Base URL for downloading models
    public var downloadBase: URL?
    /// Repository for downloading models
    public var modelRepo: String?
    /// Token for downloading models from repo (if required)
    public var modelToken: String?
    /// HuggingFace Hub compatible endpoint URL
    public var modelEndpoint: String?
    /// Folder to store models
    public var modelFolder: String?
    /// Folder to store tokenizers
    public var tokenizerFolder: URL?

    /// Model compute options, see `ModelComputeOptions`
    public var computeOptions: ModelComputeOptions?
    /// Audio input config to define how to process audio input
    public var audioInputConfig: AudioInputConfig?
    /// Audio processor for the model
    public var audioProcessor: (any AudioProcessing)?
    public var featureExtractor: (any FeatureExtracting)?
    public var audioEncoder: (any AudioEncoding)?
    public var textDecoder: (any TextDecoding)?
    public var logitsFilters: [any LogitsFiltering]?
    public var segmentSeeker: (any SegmentSeeking)?
    public var voiceActivityDetector: VoiceActivityDetector?

    /// Enable extra verbosity for logging
    public var verbose: Bool
    /// Maximum log level
    public var logLevel: Logging.LogLevel

    /// Enable model prewarming
    /// 
    /// What does "prewarm" mean and when should it be enabled?
    /// 
    /// WhisperKit uses Apple Core ML models that are downloaded as device-agnostic
    /// model files (*.mlmodelc). These models need to be "specialized" to a user's
    /// device chip before it can be used. Core ML "specializes" a model automatically
    /// during the first time the models are being loaded. The resulting "specialized"
    /// model files are cached on-disk by Core ML (not by Argmax) outside the app bundle.
    /// This cache is maintained by Apple and is evicted after every OS update and if
    /// the models are not used for extended periods of time. Unfortunately, Apple does
    /// not yet provide a third-party API to check whether the cache will be hit or is
    /// evicted. Hence, Argmax built a defensive "prewarm" option to ensure that each
    /// model gets loaded sequentially and unloaded immediately to trigger specialization if necessary.
    /// 
    /// **Trade-offs**
    /// - **Pro** - The peak memory usage during compilation is reduced because
    ///   only one model is kept in memory at any given point. Otherwise, the
    ///   peak memory will bloat to all model weights combined plus the peak
    ///   compilation memory (higher than model weights). 
    /// - **Con** - The load time will be multiplied by 2 (usually <1s when cache is hit)
    ///   because of the load-unload-load pattern when the specialized model file cache is
    ///   actually hit and prewarm does not trigger specialization
    ///
    /// Enable `prewarm` when you want to minimize your peak memory impact throughout your app's lifecycle
    /// Disable `prewarm` if you can not take a 2x increase in load time 
    public var prewarm: Bool?
    /// Load models if available
    public var load: Bool?
    /// Download models if not available
    public var download: Bool
    /// Use background download session
    public var useBackgroundDownloadSession: Bool

    public init(model: String? = nil,
                downloadBase: URL? = nil,
                modelRepo: String? = nil,
                modelToken: String? = nil,
                modelEndpoint: String? = nil,
                modelFolder: String? = nil,
                tokenizerFolder: URL? = nil,
                computeOptions: ModelComputeOptions? = nil,
                audioInputConfig: AudioInputConfig? = nil,
                audioProcessor: (any AudioProcessing)? = nil,
                featureExtractor: (any FeatureExtracting)? = nil,
                audioEncoder: (any AudioEncoding)? = nil,
                textDecoder: (any TextDecoding)? = nil,
                logitsFilters: [any LogitsFiltering]? = nil,
                segmentSeeker: (any SegmentSeeking)? = nil,
                voiceActivityDetector: VoiceActivityDetector? = nil,
                verbose: Bool = true,
                logLevel: Logging.LogLevel = .info,
                prewarm: Bool? = nil,
                load: Bool? = nil,
                download: Bool = true,
                useBackgroundDownloadSession: Bool = false)
    {
        self.model = model
        self.downloadBase = downloadBase
        self.modelRepo = modelRepo
        self.modelToken = modelToken
        self.modelEndpoint = modelEndpoint
        self.modelFolder = modelFolder
        self.tokenizerFolder = tokenizerFolder
        self.computeOptions = computeOptions
        self.audioInputConfig = audioInputConfig
        self.audioProcessor = audioProcessor
        self.featureExtractor = featureExtractor
        self.audioEncoder = audioEncoder
        self.textDecoder = textDecoder
        self.logitsFilters = logitsFilters
        self.segmentSeeker = segmentSeeker
        self.voiceActivityDetector = voiceActivityDetector
        self.verbose = verbose
        self.logLevel = logLevel
        self.prewarm = prewarm
        self.load = load
        self.download = download
        self.useBackgroundDownloadSession = useBackgroundDownloadSession
    }
}

/// Options for how to transcribe an audio file using WhisperKit.
///
/// - Parameters:
///   - verbose: Whether to display the text being decoded to the console.
///              If true, displays all details; if false, displays minimal details;
///   - task: Whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')
///   - language: Language spoken in the audio
///   - temperature: Temperature to use for sampling.
///   - temperatureIncrementOnFallback: Increment which will be
///                  successively added to temperature upon failures according to either `compressionRatioThreshold`
///                  or `logProbThreshold`.
///   - temperatureFallbackCount: Number of times to increment temperature on fallback.
///   - sampleLength: The maximum number of tokens to sample.
///   - topK: Number of candidates when sampling with non-zero temperature.
///   - usePrefillPrompt: If true, the prefill tokens will be forced according to task and language settings.
///   - detectLanguage: Use this in conjuntion with `usePrefillPrompt: true` to detect the language of the input audio.
///   - skipSpecialTokens: Whether to skip special tokens in the output.
///   - withoutTimestamps: Whether to include timestamps in the transcription result.
///   - wordTimestamps: Whether to include word-level timestamps in the transcription result.
///   - maxInitialTimestamp: Maximal initial timestamp.
///   - maxWindowSeek: If provided, prevents the seek in samples from exceeding this value for each window
///   - clipTimestamps: Array of timestamps (in seconds) to split the audio into segments for transcription.
///   - windowClipTime: Time in seconds to clip from the end of an audio window to help prevent hallucinations
///   - promptTokens: Array of token IDs to use as the conditioning prompt for the decoder. These are prepended to the prefill tokens.
///   - prefixTokens: Array of token IDs to use as the initial prefix for the decoder. These are appended to the prefill tokens.
///   - suppressBlank: If true, blank tokens will be suppressed during decoding.
///   - suppressTokens: List of token IDs to suppress during decoding.
///   - compressionRatioThreshold: If the compression ratio of the transcription text is above this value, it is too repetitive and treated as failed.
///   - logProbThreshold: If the average log probability over sampled tokens is below this value, treat as failed.
///   - firstTokenLogProbThreshold: If the log probability over the first sampled token is below this value, treat as failed.
///   - noSpeechThreshold: If the no speech probability is higher than this value AND the average log
///                        probability over sampled tokens is below `logProbThreshold`, consider the segment as silent.
public struct DecodingOptions: Codable, Sendable {
    public var verbose: Bool
    public var task: DecodingTask
    public var language: String?
    public var temperature: Float
    public var temperatureIncrementOnFallback: Float
    public var temperatureFallbackCount: Int
    public var sampleLength: Int
    public var topK: Int
    public var usePrefillPrompt: Bool
    public var detectLanguage: Bool
    public var skipSpecialTokens: Bool
    public var withoutTimestamps: Bool
    public var wordTimestamps: Bool
    public var maxInitialTimestamp: Float?
    public var maxWindowSeek: Int?
    public var clipTimestamps: [Float]
    public var windowClipTime: Float
    public var promptTokens: [Int]?
    public var prefixTokens: [Int]?
    public var suppressBlank: Bool
    public var suppressTokens: [Int]
    public var compressionRatioThreshold: Float?
    public var logProbThreshold: Float?
    public var firstTokenLogProbThreshold: Float?
    public var noSpeechThreshold: Float?
    public var concurrentWorkerCount: Int
    public var chunkingStrategy: ChunkingStrategy?
    public var alignmentEarlyStopping: Bool
    public var alignmentFrameMargin: Int
    public var alignmentContentFrameCount: Int?

    private enum CodingKeys: String, CodingKey {
        case verbose
        case task
        case language
        case temperature
        case temperatureIncrementOnFallback
        case temperatureFallbackCount
        case sampleLength
        case topK
        case usePrefillPrompt
        case detectLanguage
        case skipSpecialTokens
        case withoutTimestamps
        case wordTimestamps
        case maxInitialTimestamp
        case maxWindowSeek
        case clipTimestamps
        case windowClipTime
        case promptTokens
        case prefixTokens
        case suppressBlank
        case suppressTokens
        case compressionRatioThreshold
        case logProbThreshold
        case firstTokenLogProbThreshold
        case noSpeechThreshold
        case concurrentWorkerCount
        case chunkingStrategy
        case alignmentEarlyStopping
        case alignmentFrameMargin
        case alignmentContentFrameCount
    }

    public init(
        verbose: Bool = false,
        task: DecodingTask = .transcribe,
        language: String? = nil,
        temperature: Float = 0.0,
        temperatureIncrementOnFallback: Float = 0.2,
        temperatureFallbackCount: Int = 5,
        sampleLength: Int = Constants.maxTokenContext,
        topK: Int = 5,
        usePrefillPrompt: Bool = true,
        detectLanguage: Bool? = nil,
        skipSpecialTokens: Bool = false,
        withoutTimestamps: Bool = false,
        wordTimestamps: Bool = false,
        maxInitialTimestamp: Float? = nil,
        maxWindowSeek: Int? = nil,
        clipTimestamps: [Float] = [],
        windowClipTime: Float = 1.0,
        promptTokens: [Int]? = nil,
        prefixTokens: [Int]? = nil,
        suppressBlank: Bool = false,
        suppressTokens: [Int]? = nil,
        compressionRatioThreshold: Float? = 2.4,
        logProbThreshold: Float? = -1.0,
        firstTokenLogProbThreshold: Float? = -1.5,
        noSpeechThreshold: Float? = 0.6,
        concurrentWorkerCount: Int? = nil,
        chunkingStrategy: ChunkingStrategy? = nil,
        alignmentEarlyStopping: Bool = false,
        alignmentFrameMargin: Int = 25,
        alignmentContentFrameCount: Int? = nil
    ) {
        self.verbose = verbose
        self.task = task
        self.language = language
        self.temperature = temperature
        self.temperatureIncrementOnFallback = temperatureIncrementOnFallback
        self.temperatureFallbackCount = temperatureFallbackCount
        self.sampleLength = sampleLength
        self.topK = topK
        self.usePrefillPrompt = usePrefillPrompt
        self.detectLanguage = detectLanguage ?? !usePrefillPrompt // If prefill is false, detect language by default
        self.skipSpecialTokens = skipSpecialTokens
        self.withoutTimestamps = withoutTimestamps
        self.wordTimestamps = wordTimestamps
        self.maxInitialTimestamp = maxInitialTimestamp
        self.maxWindowSeek = maxWindowSeek
        self.clipTimestamps = clipTimestamps
        self.windowClipTime = windowClipTime
        self.promptTokens = promptTokens
        self.prefixTokens = prefixTokens
        self.suppressBlank = suppressBlank
        self.suppressTokens = suppressTokens ?? [] // nonSpeechTokens() // TODO: implement these as default
        self.compressionRatioThreshold = compressionRatioThreshold
        self.logProbThreshold = logProbThreshold
        self.firstTokenLogProbThreshold = firstTokenLogProbThreshold
        self.noSpeechThreshold = noSpeechThreshold
        // Set platform-specific default worker count if not explicitly provided
        // Non-macOS devices have shown regressions with >4 workers, default to 4 for safety
        #if os(macOS)
        self.concurrentWorkerCount = concurrentWorkerCount ?? 16
        #else
        self.concurrentWorkerCount = concurrentWorkerCount ?? 4
        #endif
        self.chunkingStrategy = chunkingStrategy
        self.alignmentEarlyStopping = alignmentEarlyStopping
        self.alignmentFrameMargin = alignmentFrameMargin
        self.alignmentContentFrameCount = alignmentContentFrameCount
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let compressionRatioThreshold = try container.contains(.compressionRatioThreshold)
            ? container.decode(Float?.self, forKey: .compressionRatioThreshold)
            : Float(2.4)
        let logProbThreshold = try container.contains(.logProbThreshold)
            ? container.decode(Float?.self, forKey: .logProbThreshold)
            : Float(-1.0)
        let firstTokenLogProbThreshold = try container.contains(.firstTokenLogProbThreshold)
            ? container.decode(Float?.self, forKey: .firstTokenLogProbThreshold)
            : Float(-1.5)
        let noSpeechThreshold = try container.contains(.noSpeechThreshold)
            ? container.decode(Float?.self, forKey: .noSpeechThreshold)
            : Float(0.6)

        self.init(
            verbose: try container.decodeIfPresent(Bool.self, forKey: .verbose) ?? false,
            task: try container.decodeIfPresent(DecodingTask.self, forKey: .task) ?? .transcribe,
            language: try container.decodeIfPresent(String.self, forKey: .language),
            temperature: try container.decodeIfPresent(Float.self, forKey: .temperature) ?? 0.0,
            temperatureIncrementOnFallback: try container.decodeIfPresent(Float.self, forKey: .temperatureIncrementOnFallback) ?? 0.2,
            temperatureFallbackCount: try container.decodeIfPresent(Int.self, forKey: .temperatureFallbackCount) ?? 5,
            sampleLength: try container.decodeIfPresent(Int.self, forKey: .sampleLength) ?? Constants.maxTokenContext,
            topK: try container.decodeIfPresent(Int.self, forKey: .topK) ?? 5,
            usePrefillPrompt: try container.decodeIfPresent(Bool.self, forKey: .usePrefillPrompt) ?? true,
            detectLanguage: try container.decodeIfPresent(Bool.self, forKey: .detectLanguage),
            skipSpecialTokens: try container.decodeIfPresent(Bool.self, forKey: .skipSpecialTokens) ?? false,
            withoutTimestamps: try container.decodeIfPresent(Bool.self, forKey: .withoutTimestamps) ?? false,
            wordTimestamps: try container.decodeIfPresent(Bool.self, forKey: .wordTimestamps) ?? false,
            maxInitialTimestamp: try container.decodeIfPresent(Float.self, forKey: .maxInitialTimestamp),
            maxWindowSeek: try container.decodeIfPresent(Int.self, forKey: .maxWindowSeek),
            clipTimestamps: try container.decodeIfPresent([Float].self, forKey: .clipTimestamps) ?? [],
            windowClipTime: try container.decodeIfPresent(Float.self, forKey: .windowClipTime) ?? 1.0,
            promptTokens: try container.decodeIfPresent([Int].self, forKey: .promptTokens),
            prefixTokens: try container.decodeIfPresent([Int].self, forKey: .prefixTokens),
            suppressBlank: try container.decodeIfPresent(Bool.self, forKey: .suppressBlank) ?? false,
            suppressTokens: try container.decodeIfPresent([Int].self, forKey: .suppressTokens),
            compressionRatioThreshold: compressionRatioThreshold,
            logProbThreshold: logProbThreshold,
            firstTokenLogProbThreshold: firstTokenLogProbThreshold,
            noSpeechThreshold: noSpeechThreshold,
            concurrentWorkerCount: try container.decodeIfPresent(Int.self, forKey: .concurrentWorkerCount),
            chunkingStrategy: try container.decodeIfPresent(ChunkingStrategy.self, forKey: .chunkingStrategy),
            alignmentEarlyStopping: try container.decodeIfPresent(Bool.self, forKey: .alignmentEarlyStopping) ?? false,
            alignmentFrameMargin: try container.decodeIfPresent(Int.self, forKey: .alignmentFrameMargin) ?? 25,
            alignmentContentFrameCount: try container.decodeIfPresent(Int.self, forKey: .alignmentContentFrameCount)
        )
    }
}
