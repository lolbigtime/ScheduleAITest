//
//  Engine.swift
//  ScheduleAI
//
//  Created by Tai Wong on 10/1/25.
//

import Foundation
import Folio
import Combine
import CryptoKit
import OSLog
import Accelerate
import CoreML




public enum EngineError: Error, Equatable {
    case none
    case searchError
    case ingestError
}

public enum SearchMode: Codable, Sendable {
    case semantic
    case keyword
    case withContext(expand: Int)
    case hybrid(expand: Int, wBM25: Double)

    private enum CodingKeys: String, CodingKey {
        case type
        case expand
        case wBM25
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "semantic":
            self = .semantic
        case "keyword":
            self = .keyword
        case "withContext":
            let expand = try container.decodeIfPresent(Int.self, forKey: .expand) ?? 1
            self = .withContext(expand: expand)
        case "hybrid":
            let expand = try container.decodeIfPresent(Int.self, forKey: .expand) ?? 1
            let wBM25 = try container.decodeIfPresent(Double.self, forKey: .wBM25) ?? 0.5
            self = .hybrid(expand: expand, wBM25: wBM25)
        default:
            throw DecodingError.dataCorrupted(
                .init(codingPath: decoder.codingPath, debugDescription: "Unsupported search mode: \(type)")
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .semantic:
            try container.encode("semantic", forKey: .type)
        case .keyword:
            try container.encode("keyword", forKey: .type)
        case .withContext(let expand):
            try container.encode("withContext", forKey: .type)
            try container.encode(expand, forKey: .expand)
        case .hybrid(let expand, let wBM25):
            try container.encode("hybrid", forKey: .type)
            try container.encode(expand, forKey: .expand)
            try container.encode(wBM25, forKey: .wBM25)
        }
    }
}


public class Engine: ObservableObject {
    public let folio: FolioEngine
    private var ingestConfig: FolioConfig
    public var errorState: EngineError = .none
    public static let sharedTask: Task<Engine, Error> = Task { await Engine() }
    public static func shared() async throws -> Engine { try await sharedTask.value }
    
    @discardableResult
    public func importPDF(at url: URL) async throws -> (pages: Int, chunks: Int) {
        do {
            let sourceId = DocumentHasher.sha256Hex(of: try Data(contentsOf: url))
            return try await self.folio.ingestAsync(.pdf(url), sourceId: sourceId, config: self.ingestConfig)
        } catch {
            errorState = .ingestError
            throw error
        }
    }
    
    @discardableResult
    public func importText(text: String, name: String? = nil) async throws -> (chars: Int, chunks: Int) {
        do {
            let data = Data(text.utf8)
            let hash = DocumentHasher.sha256Hex(of: data)
            let docName = name ?? "text-\(hash.prefix(8))"
            let sourceId = "text:" + hash
            let result = try await self.folio.ingestAsync(.text(text, name: docName), sourceId: sourceId, config: self.ingestConfig)
            return (chars: text.utf8.count, chunks: result.chunks)
        } catch {
            errorState = .ingestError
            throw error
        }
    }
    
    @discardableResult
    public func importEmails(
        _ emails: [EmailMessage],
        sourceId: String,
        name: String? = nil
    ) async throws -> (items: Int, chunks: Int) {
        guard !emails.isEmpty else { return (items: 0, chunks: 0) }
        do {
            let iso = ISO8601DateFormatter()
            let blocks: [String] = emails.map { msg in
                var header = """
                Subject: \(msg.subject)
                From: \(msg.from)
                To: \(msg.to.joined(separator: ", "))
                Date: \(iso.string(from: msg.date))
                """.trimmingCharacters(in: .whitespacesAndNewlines)
                if !msg.cc.isEmpty {
                    header += "\nCC: " + msg.cc.joined(separator: ", ")
                }
                return header + "\n\n" + msg.body
            }
            let combined = blocks.joined(separator: "\n\n---\n\n")
            let docName = name ?? "Emails"
            let result = try await self.folio.ingestAsync(.text(combined, name: docName), sourceId: sourceId, config: self.ingestConfig)
            return (items: emails.count, chunks: result.chunks)
        } catch {
            errorState = .ingestError
            throw error
        }
    }

    @discardableResult
    public func importSchedule(
        _ events: [CalendarEvent],
        sourceId: String,
        name: String? = nil
    ) async throws -> (items: Int, chunks: Int) {
        guard !events.isEmpty else { return (items: 0, chunks: 0) }
        do {
            let iso = ISO8601DateFormatter()
            let blocks: [String] = events.map { ev in
                let when = "\(iso.string(from: ev.startDate)) – \(iso.string(from: ev.endDate))"
                let loc = ev.location ?? "N/A"
                let notes = ev.notes?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                return """
                Event: \(ev.title)
                When: \(when)
                Location: \(loc)
                Notes: \(notes)
                """.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            let combined = blocks.joined(separator: "\n\n---\n\n")
            let docName = name ?? "Schedule"
            let result = try await self.folio.ingestAsync(.text(combined, name: docName), sourceId: sourceId, config: self.ingestConfig)
            return (items: events.count, chunks: result.chunks)
        } catch {
            errorState = .ingestError
            throw error
        }
    }
    
    
    @discardableResult
    public func search(_ query: String, in source: String? = nil, topK: Int = 8, mode: SearchMode = .semantic) throws -> [RetrievedResult] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        let limit = max(1, min(topK, 50))

        do {
            switch mode {
            case .semantic:
                return try self.folio.searchHybrid(trimmed, in: source, limit: limit, expand: 1, wBM25: 0.0)
            case .keyword:
                return try mapPassagesToResults(self.folio.searchWithContext(trimmed, in: source, limit: limit, expand: 0))
            case .withContext(let expand):
                let safeExpand = max(0, expand)
                return try mapPassagesToResults(self.folio.searchWithContext(trimmed, in: source, limit: limit, expand: safeExpand))
            case .hybrid(let expand, let wBM25):
                let safeExpand = max(0, expand)
                let safeWeight = max(0.0, min(wBM25, 1.0))
                return try self.folio.searchHybrid(trimmed, in: source, limit: limit, expand: safeExpand, wBM25: safeWeight)
            }
        } catch {
            errorState = .searchError
            throw error
        }
    }
    
    

    public init() async {
        let fileManager = FileManager.default
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        
        let folioDir = appSupport.appendingPathComponent("Folio", isDirectory: true)
        let docsDir = appSupport.appendingPathComponent("Docs", isDirectory: true)
        
        try? fileManager.createDirectory(at: folioDir, withIntermediateDirectories: true)
        try? fileManager.createDirectory(at: docsDir, withIntermediateDirectories: true)
        
        let dbURL = folioDir.appendingPathComponent("folio.sqlite")
        
        
        let pdfLoader = PDFDocumentLoader()
        let textLoader = TextDocumentLoader()
        let chunker = UniversalChunker()
        
        
        
        guard let tokenizerURL = Bundle.main.url(forResource: "QwenTokenizer", withExtension: nil) else {
            fatalError("Missing tokenizer folder in app bundle")
        }
        
        let qwenURL: URL
        if let url = Bundle.main.url(forResource: "qwenEmbedding", withExtension: "mlmodelc") {
            qwenURL = url
        } else if let url = Bundle.main.url(forResource: "qwenEmbedding", withExtension: "mlpackage") {
            qwenURL = url
        } else {
            fatalError("Missing qwenEmbedding.mlmodelc in app bundle. Ensure the model is added to the app target and included in Copy Bundle Resources.")
        }
        
        let qwenOutputName = "var_3996"

        do {
            let qwen = try await QwenCoreMLEmbedder(
                modelURL: qwenURL,
                tokenizerURL: tokenizerURL,
                seqLen: 512,
                storeDim: 512,
                outputName: qwenOutputName
            )

            self.folio = try FolioEngine(databaseURL: dbURL, loaders: [pdfLoader, textLoader], chunker: chunker, embedder: qwen)
        } catch {
            let log = Logger(subsystem: "ScheduleAI", category: "Engine")
            log.error("Init failed: \(error.localizedDescription, privacy: .public)")
            // Also dump more detail:
            print("FULL ERROR:", error)
            assertionFailure("Init failed: \(error)")
            fatalError("Init failed")
        }
        
        var config = FolioConfig()
        
        config.indexing.useFoundationModelPrefixes()
        config.chunking.maxTokensPerChunk = 1000
        config.chunking.overlapTokens = 150
        
        self.ingestConfig = config
        
        self.errorState = .none
        
        
        
    }
}


/*
final class QwenEmbedder: Embedder {
    private let model: qwenEmbedding  // auto-generated by Core ML
    private let queue = DispatchQueue(label: "QwenEmbedder")

    init() throws {
        model = try qwenEmbedding(configuration: MLModelConfiguration())
    }

    func embed(_ text: String) throws -> [Float] {
        try queue.sync {
            let input = qwenEmbeddingInput(text: text)
            let output = try model.prediction(input: input)
            return Array(output.embeddingShapedArray.scalars.map(Float.init))
        }
    }

    func embedBatch(_ texts: [String]) throws -> [[Float]] {
        try queue.sync { try texts.map(embed(_:)) }
    }
}
*/

