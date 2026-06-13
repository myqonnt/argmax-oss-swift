# SimulStreamingTranscriber 集成指南

`SimulStreamingTranscriber` 是 WhisperKit 中一条 SimulStreaming-style 的低延迟转录路径。它基于 Whisper decoder 的 `alignment_heads_weights` 做逐 token 判断：当新 token 的 cross-attention 已经接近当前音频末尾时，停止提交该 token，只输出更稳定的前缀。

这条路径适合需要实时字幕、会议听写、同传前置 ASR 等场景。它和 `AudioStreamTranscriber` 的区别是：`AudioStreamTranscriber` 主要用“重复转录当前 buffer + 确认前几个 segment”的策略；`SimulStreamingTranscriber` 直接在 decoder loop 里做 attention-guided early stop。

## 当前能力

- 支持增量音频输入：`insertAudioChunk(_:)`
- 支持按最小 chunk 间隔处理：`process()`
- 支持话轮结束时强制 flush：`finish()`
- 使用已输出 token 作为后续 prompt context，并通过 `<|startofprev|>` 保持 Whisper previous-context 语义
- 使用 Whisper `no_speech` 概率抑制静音/噪声幻觉
- 使用 attention frame 生成近似词级时间戳
- 在非 final 更新中可截掉末尾未稳定词
- 要求使用 WhisperKit 标准 `TextDecoder`，且 decoder 模型必须输出 `alignment_heads_weights`

当前第一版只实现 greedy decoding；beam search、CIF 词边界模型、Silero VAC 还未接入。

## SwiftPM 引入

在其它项目的 `Package.swift` 中添加依赖：

```swift
.package(url: "<argmax-oss-swift-repo-url>", branch: "<branch-or-tag>")
```

然后在 target 中依赖 `WhisperKit`：

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "WhisperKit", package: "argmax-oss-swift")
    ]
)
```

如果你是在本地开发当前仓库，也可以用本地路径：

```swift
.package(path: "/path/to/argmax-oss-swift")
```

## 初始化

先创建并加载 `WhisperKit`：

```swift
import WhisperKit

let whisperKit = try await WhisperKit(
    model: "openai_whisper-large-v3",
    download: true,
    load: true
)
```

创建 SimulStreaming transcriber：

```swift
let transcriber = try whisperKit.makeSimulStreamingTranscriber(
    decodingOptions: DecodingOptions(
        task: .transcribe,
        language: "en",
        temperature: 0,
        sampleLength: 224
    ),
    streamingOptions: SimulStreamingOptions(
        minChunkSeconds: 1.0,
        audioMinSeconds: 0.0,
        audioMaxSeconds: 30.0,
        frameThreshold: 25,
        rewindThreshold: 200,
        maxContextTokens: 112,
        trimLastWordWhenUnfinished: true
    )
)
```

`makeSimulStreamingTranscriber` 会检查 tokenizer 是否可用，并要求底层 decoder 是 WhisperKit 标准 `TextDecoder`。如果你给 `WhisperKitConfig` 注入了自定义 `TextDecoding` 实现，这个工厂会抛错，因为 SimulStreaming 需要读取每个 token 的 alignment weights。

## 从麦克风实时转录

`SimulStreamingTranscriber` 是有状态对象。请从同一个 task/queue 串行调用 `insertAudioChunk`、`process` 和 `finish`。

```swift
let audioProcessor = AudioProcessor()
let (stream, continuation) = audioProcessor.startStreamingRecordingLive()

for try await samples in stream {
    transcriber.insertAudioChunk(samples)

    let update = try await transcriber.process()
    if !update.text.isEmpty {
        print("[\(update.start ?? 0) - \(update.end ?? 0)] \(update.text)")
    }
}

let finalUpdate = try await transcriber.finish()
if !finalUpdate.text.isEmpty {
    print("[final] \(finalUpdate.text)")
}

continuation.finish()
audioProcessor.stopRecording()
```

## 从音频文件模拟流式转录

如果你已经有 16 kHz mono float samples，可以按固定 chunk 喂入：

```swift
let chunkSeconds: Float = 0.5
let chunkSize = Int(chunkSeconds * Float(WhisperKit.sampleRate))

var index = 0
while index < samples.count {
    let end = min(index + chunkSize, samples.count)
    transcriber.insertAudioChunk(Array(samples[index..<end]))

    let update = try await transcriber.process()
    if !update.text.isEmpty {
        print("\(update.start ?? 0) \(update.end ?? 0) \(update.text)")
    }

    index = end
}

let finalUpdate = try await transcriber.finish()
if !finalUpdate.text.isEmpty {
    print("\(finalUpdate.start ?? 0) \(finalUpdate.end ?? 0) \(finalUpdate.text)")
}
```

如果你的文件还没有转成 samples，可以用 WhisperKit 的 audio loading 工具先加载：

```swift
let results = await AudioProcessor.loadAudio(
    at: [audioPath],
    channelMode: .sumChannels(nil)
)

guard case let .success(samples) = results[0] else {
    throw WhisperError.loadAudioFailed("Unable to load audio")
}
```

## 输出格式

`process()` 和 `finish()` 返回 `SimulStreamingUpdate`：

```swift
public struct SimulStreamingUpdate {
    public var start: Float?
    public var end: Float?
    public var text: String
    public var tokens: [Int]
    public var words: [SimulStreamingWord]
    public var isFinal: Bool
    public var noSpeechProb: Float?
}
```

- `text`：本次新增文本，不是完整全文。
- `tokens`：本次新增 token。
- `words`：按 tokenizer 聚合后的词级信息。
- `start` / `end`：本次新增文本的近似音频时间戳。
- `isFinal`：通常只有 `finish()` 返回 true；可用于 UI 换行或话轮提交。
- `noSpeechProb`：Whisper 在当前窗口上预测 `<|nospeech|>` 的概率；高于 `DecodingOptions.noSpeechThreshold` 时会返回空文本。

词级输出：

```swift
public struct SimulStreamingWord {
    public var start: Float
    public var end: Float
    public var text: String
    public var tokens: [Int]
}
```

时间戳来自每个 token 的最关注 audio frame，精度适合实时 UI 和粗粒度对齐，不应当当作离线强制对齐结果。

## 参数建议

`minChunkSeconds`
: 新增音频达到这个时长后，`process()` 才会真正跑一次模型。低延迟可设为 `0.5` 到 `1.0`；更省电可设为 `1.0` 到 `1.5`。

`audioMinSeconds`
: 当前 audio buffer 短于该值时跳过解码。通常保持 `0.0` 即可。

`audioMaxSeconds`
: 保留的最大音频窗口。Whisper 默认窗口是 30 秒，因此默认 `30.0`。

`frameThreshold`
: AlignAtt 停止阈值。一个 frame 约等于 `0.02` 秒。默认 `25` 约等于 0.5 秒；值越大越保守、延迟越高但更稳。

`rewindThreshold`
: 如果 attention 相比上次大幅回退，停止本轮提交。默认 `200` 约等于 4 秒，用于防止异常 attention 跳变。

`maxContextTokens`
: 后续解码保留多少已输出 token 作为 previous-context prompt。WhisperKit 会在这些 token 前加 `<|startofprev|>`。默认使用 Whisper token context 的一半。

`trimLastWordWhenUnfinished`
: 非 final 更新中，如果本轮因为接近音频末尾而停止，且本轮至少识别出两个词，则截掉最后一个词，降低半词误识别概率。单词短 utterance 会保留，避免短命令一直不输出。

## 常见用法模式

实时 UI 可以把 `update.text` 追加到“已确认文本”区域：

```swift
var committedText = ""

let update = try await transcriber.process()
if !update.text.isEmpty {
    committedText += update.text
}
```

如果希望每个语音活动片段单独成句，可以在自己的 VAD 检测到语音结束时调用：

```swift
let finalUpdate = try await transcriber.finish()
```

之后 transcriber 会 reset，并从下一个音频 chunk 开始新的话轮。

## 限制与注意事项

- 必须串行调用；不要从多个 task 同时调用同一个 `SimulStreamingTranscriber` 实例。
- 当前仅实现 greedy decoding。
- 当前没有内置 Silero VAC；可以先用已有 VAD/能量检测在外部决定何时 `finish()`。
- 需要带 `alignment_heads_weights` 输出的 WhisperKit CoreML decoder 模型。
- `text` 是增量输出，不是完整 transcript。
- 非 final 输出为了稳定性可能不会吐出最后一个词。
- `no_speech` 检测使用 Whisper decoder logits，并不能替代外部 VAD；实时产品里建议两者结合，VAD 负责少跑模型，`no_speech` 负责抑制静音幻觉。
- 如果你需要高精度离线词级时间戳，仍应使用 WhisperKit 原有 `wordTimestamps` 路径。

## 测试与 Python 对齐

当前仓库为不依赖真实模型的 streaming helper 覆盖了单元测试：

```bash
swift test --filter SimulStreamingTests
```

这些测试覆盖：

- `no_speech` softmax 概率计算
- 历史 token 走 `promptTokens` 而不是 `prefixTokens`
- context token 限长策略
- `contentFrameCount` 的 20ms frame 计算和 30s 窗口上限
- `SimulStreamingUpdate.noSpeechProb` 输出字段

如果要做和 Python SimulStreaming 的端到端对齐，建议使用同一段 16kHz mono wav、相同 `minChunkSeconds`、相同语言和 greedy decoding，对比以下指标：

- 每次 update 的新增文本
- 每次 update 的 token ids
- 每个 token/word 的 most-attended frame
- near-end stop 是否在相同 chunk 触发
- 静音段是否返回空输出

Swift 和 Python 的 `contentFrameCount` 允许 1 frame 以内误差；这是 mel/frame 取整方式和浮点精度差异导致的 20ms 级偏差。

## 故障排查

`SimulStreamingTranscriber requires WhisperKit's standard TextDecoder`
: 你注入了自定义 `TextDecoding`。请使用 WhisperKit 标准 `TextDecoder`，或者在自定义 decoder 中实现等价的 per-token alignment 访问能力。

`Streaming decoding requires decoder models that output alignment_heads_weights`
: 当前 decoder 模型没有导出 alignment weights。请换用支持 word timestamps/alignment heads 的 WhisperKit 模型。

输出延迟偏高
: 降低 `minChunkSeconds` 或 `frameThreshold`，但过低会增加算力消耗和不稳定输出。

输出容易截断词尾
: 将 `trimLastWordWhenUnfinished` 设为 `false`，或降低 `frameThreshold`。

静音时仍然有幻觉
: 确认 `DecodingOptions.noSpeechThreshold` 没有设为 `nil` 或过高。默认值是 `0.6`。同时建议在外部加 VAD，在明显静音时跳过 `process()` 或直接等待更多音频。
