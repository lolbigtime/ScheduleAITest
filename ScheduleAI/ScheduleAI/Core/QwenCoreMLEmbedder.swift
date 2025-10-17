import Foundation
import CoreML
import Accelerate
import Tokenizers
import Folio

public enum QwenPooling: String {
    case mean
    case cls
    case last
}

public final class QwenCoreMLEmbedder: Embedder {
    private let model: MLModel
    private let tokenizer: any Tokenizer
    private let seqLen: Int
    private let storeDim: Int

    private let inputIdsName: String
    private let attentionMaskName: String
    private let outputName: String

    private let pooling: QwenPooling

    /// Create a Core ML embedder for Qwen embeddings.
    /// - Parameters:
    ///   - modelURL: Path to the .mlpackage
    ///   - tokenizerURL: Path to tokenizer.json compatible with the model export
    ///   - seqLen: Sequence length the model expects (e.g. 512)
    ///   - storeDim: Number of dimensions to keep (<= model output dim)
    ///   - inputIdsName: Optional override for the input ids feature name
    ///   - attentionMaskName: Optional override for the attention mask feature name
    ///   - outputName: Optional override for output feature name. If not provided, we try to detect it.
    ///   - pooling: If the model outputs per-token embeddings, how to pool to a single vector
    public init(
        modelURL: URL,
        tokenizerURL: URL,
        seqLen: Int = 512,
        storeDim: Int = 512,
        inputIdsName: String? = nil,
        attentionMaskName: String? = nil,
        outputName: String? = nil,
        pooling: QwenPooling = .mean
    ) async throws {
        self.model = try MLModel(contentsOf: modelURL)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerURL)
        
        
        self.seqLen = seqLen
        self.storeDim = storeDim
        self.pooling = pooling

        // Auto-detect IO names when not provided
        let io = QwenCoreMLEmbedder.detectIONames(from: self.model,
                                                  inputIdsOverride: inputIdsName,
                                                  attentionMaskOverride: attentionMaskName,
                                                  outputOverride: outputName)
        guard let io else {
            throw NSError(domain: "QwenEmbed", code: -10, userInfo: [NSLocalizedDescriptionKey: "Unable to detect Core ML IO feature names. Provide overrides."])
        }
        self.inputIdsName = io.inputIds
        self.attentionMaskName = io.attentionMask
        self.outputName = io.output

        // Warmup to reduce first-call latency
        _ = try? embed("warmup")
    }

    public func embed(_ text: String) throws -> [Float] {
        // 1) Tokenize with the provided tokenizer
        var ids = tokenizer.encode(text: text).map { Int32($0) }
        if ids.count > seqLen { ids = Array(ids.prefix(seqLen)) }
        var mask = [Int32](repeating: 1, count: ids.count)
        if ids.count < seqLen {
            ids += [Int32](repeating: 0, count: seqLen - ids.count)
            mask += [Int32](repeating: 0, count: seqLen - mask.count)
        }

        // 2) Build ML inputs (shape [1, seqLen])
        let idArr = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        let mkArr = try MLMultiArray(shape: [1, NSNumber(value: seqLen)], dataType: .int32)
        for i in 0..<seqLen {
            idArr[i] = NSNumber(value: ids[i])
            mkArr[i] = NSNumber(value: mask[i])
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            inputIdsName: idArr,
            attentionMaskName: mkArr
        ])

        // 3) Run inference
        let out = try model.prediction(from: provider)
        guard let ma = out.featureValue(for: outputName)?.multiArrayValue else {
            let keys = out.featureNames.joined(separator: ", ")
            throw NSError(domain: "QwenEmbed", code: -11,
                          userInfo: [NSLocalizedDescriptionKey: "Output '\(outputName)' not found. Available: [\(keys)]"])
        }

        // 4) Extract floats (float16 or float32)
        let full: [Float] = try QwenCoreMLEmbedder.multiArrayToFloat(ma)

        // 5) If output is [1, dim] -> use directly; if [1, seq, dim] -> pool
        let shape = ma.shape.map { Int(truncating: $0) }
        let vec: [Float]
        if shape.count == 2 {
            // [1, dim]
            vec = full
        } else if shape.count == 3 {
            // [1, seq, dim]
            let seq = shape[1]
            let dim = shape[2]
            vec = QwenCoreMLEmbedder.poolTokens(full: full, seq: seq, dim: dim, mask: mask, strategy: pooling)
        } else {
            throw NSError(domain: "QwenEmbed", code: -12,
                          userInfo: [NSLocalizedDescriptionKey: "Unsupported output rank \(shape) – expected [1, dim] or [1, seq, dim]"])
        }

        // 6) Keep first storeDim
        var cut = Array(vec.prefix(storeDim))

        // 7) L2-normalize
        var s: Float = 0; vDSP_svesq(cut, 1, &s, vDSP_Length(cut.count))
        let n = sqrtf(max(s, 1e-12))
        var outv = [Float](repeating: 0, count: cut.count); var nn = n
        vDSP_vsdiv(cut, 1, &nn, &outv, 1, vDSP_Length(cut.count))
        return outv
    }

    public func embedBatch(_ texts: [String]) throws -> [[Float]] {
        try texts.map { try embed($0) }
    }
}

// MARK: - Helpers

private extension QwenCoreMLEmbedder {
    static func detectIONames(from model: MLModel,
                              inputIdsOverride: String?,
                              attentionMaskOverride: String?,
                              outputOverride: String?) -> (inputIds: String, attentionMask: String, output: String)? {
        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName

        // Resolve input ids
        let inputIds: String
        if let override = inputIdsOverride { inputIds = override }
        else if inputs.keys.contains("input_ids") { inputIds = "input_ids" }
        else if let k = inputs.keys.first(where: { $0.localizedCaseInsensitiveContains("input") && $0.localizedCaseInsensitiveContains("id") }) { inputIds = k }
        else if let any = inputs.keys.sorted().first { inputIds = any } else { return nil }

        // Resolve attention mask
        let attn: String
        if let override = attentionMaskOverride { attn = override }
        else if inputs.keys.contains("attention_mask") { attn = "attention_mask" }
        else if let k = inputs.keys.first(where: { $0.localizedCaseInsensitiveContains("mask") }) { attn = k }
        else if let other = inputs.keys.first(where: { $0 != inputIds }) { attn = other } else { return nil }

        // Resolve output
        let outName: String
        if let override = outputOverride { outName = override }
        else if outputs.keys.contains("last_hidden_state") { outName = "last_hidden_state" }
        else if outputs.keys.contains("pooled_output") { outName = "pooled_output" }
        else if let k = outputs.keys.first(where: { $0.localizedCaseInsensitiveContains("output") }) { outName = k }
        else if let any = outputs.keys.sorted().first { outName = any } else { return nil }

        return (inputIds: inputIds, attentionMask: attn, output: outName)
    }

    static func multiArrayToFloat(_ ma: MLMultiArray) throws -> [Float] {
        switch ma.dataType {
        case .float32:
            let ptr = ma.dataPointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: ptr, count: ma.count))
        case .float16:
            let ptr = ma.dataPointer.assumingMemoryBound(to: UInt16.self)
            var f16 = Array(UnsafeBufferPointer(start: ptr, count: ma.count))
            var src = f16
            var out = [Float](repeating: 0, count: ma.count)
            out.withUnsafeMutableBufferPointer { dst in
                var srcBuf = vImage_Buffer(data: &src, height: 1, width: UInt(ma.count), rowBytes: ma.count * 2)
                var dstBuf = vImage_Buffer(data: dst.baseAddress, height: 1, width: UInt(ma.count), rowBytes: ma.count * 4)
                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
            return out
        default:
            throw NSError(domain: "QwenEmbed", code: -13,
                          userInfo: [NSLocalizedDescriptionKey: "Unsupported MLMultiArray dtype \(ma.dataType)"])
        }
    }

    /// Pool [seq, dim] token embeddings into a single [dim] vector.
    static func poolTokens(full: [Float], seq: Int, dim: Int, mask: [Int32], strategy: QwenPooling) -> [Float] {
        // full is in row-major order for [1, seq, dim] -> treat as [seq, dim]
        // Build masked mean / cls / last
        switch strategy {
        case .mean:
            var acc = [Float](repeating: 0, count: dim)
            var count: Float = 0
            for t in 0..<seq {
                if t < mask.count && mask[t] == 0 { continue }
                let base = t * dim
                vDSP_vadd(acc, 1, Array(full[base..<(base+dim)]), 1, &acc, 1, vDSP_Length(dim))
                count += 1
            }
            if count == 0 { return Array(full.prefix(dim)) }
            var c = count
            var out = [Float](repeating: 0, count: dim)
            vDSP_vsdiv(acc, 1, &c, &out, 1, vDSP_Length(dim))
            return out
        case .cls:
            // Take first non-masked token (usually position 0)
            let idx = (mask.firstIndex(where: { $0 != 0 }) ?? 0)
            let base = min(idx, seq - 1) * dim
            return Array(full[base..<(base+dim)])
        case .last:
            // Take last non-masked token, fallback to last token
            let lastIdx = (mask.lastIndex(where: { $0 != 0 }) ?? (seq - 1))
            let base = min(max(lastIdx, 0), seq - 1) * dim
            return Array(full[base..<(base+dim)])
        }
    }
}

