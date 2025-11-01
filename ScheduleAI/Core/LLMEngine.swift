//
//  LLMEngine.swift
//  ScheduleAI
//
//  Created by Tai Wong on 10/30/25.
//

import Foundation
import FoundationModels
import Folio

public enum LLMEngineError: Error, Equatable {
    case none
    case engineError
}

public struct RagSearchTool: Tool {
    
    public let name = "search_rag"
    public let description: String = "Searches a local RAG database for saved documents"
    
    @Generable
    public struct Arguments {
        @Guide(
            description: "Natural‑language query describing what to retrieve from the local knowledge base (notes, docs, email, calendar, etc.). Keep it focused."
        )
        public var query: String

        @Guide(
            description: "Optional source identifier to narrow where to search. Pass nil to search all sources (notes, docs, email, calendar, etc.)."
        )
        public var in_: String? = nil

        @Guide(
            description: "Maximum number of results to return. Defaults to 5.", .range(1...25)
        )
        public var top_k: Int = 5

        @Guide(
            description: "Retrieval strategy to use. Common values: 'keyword', 'semantic', 'withContext', or 'hybrid'. Defaults to 'hybrid'."
        )
        public var mode: String = "hybrid"

        @Guide(
            description: "Context window expansion for 'withContext' or 'hybrid'. Non-negative. Defaults to 1.", .range(0...1000)
        )
        public var expand: Int = 1

        @Guide(
            description: "Weight for BM25 in hybrid fusion (0.0 to 1.0). Ignored unless mode is 'hybrid'. Defaults to 0.5.", .range(0.0...1.0)
        )
        public var wBM25: Double = 0.5
    }
    
    @Generable
    public struct Output {
        public var hits: [Hit]
    }

    @Generable
    public struct Hit {
        public var sourceId: String
        public var startPage: Int?
        public var excerpt: String
        public var text: String
        public var bm25: Double
        public var cosine: Double?
        public var score: Double
    }
    
    
    public func call(arguments: Arguments) async throws -> Output {
        let engine = try await Engine.shared()
        let sanitizedTopK = max(1, min(arguments.top_k, 25))
        let selectedMode: SearchMode = {
            let m = arguments.mode.lowercased()
            let clampedExpand = max(0, arguments.expand)
            let clampedWBM25 = max(0.0, min(arguments.wBM25, 1.0))
            switch m {
            case "semantic":
                return .semantic
            case "keyword":
                return .keyword
            case "withcontext", "context":
                return .withContext(expand: clampedExpand)
            case "hybrid":
                fallthrough
            default:
                return .hybrid(expand: clampedExpand, wBM25: clampedWBM25)
            }
        }()
        
        
        let results: [RetrievedResult] = try await engine.search(arguments.query, in: arguments.in_, topK: sanitizedTopK, mode: selectedMode)

        let hits = results.map { r in
            Hit(sourceId: r.sourceId, startPage: r.startPage, excerpt: r.excerpt, text: r.text, bm25: r.bm25, cosine: r.cosine, score: r.score)
        }
        return Output(hits: hits)
    }
}

public struct CurrentTimeTool: Tool {
    public let name = "now"
    public let description: String = "Fetches the current time for the device."
    
    @Generable
    public struct Arguments { }
    
    @Generable
    public struct Output {
        public var nowISO: String
        public var timezone: String
    }
    
    public func call(arguments: Arguments) async throws -> Output {
        let now = Date()
        let tz = TimeZone.current
        let iso = Self.isoFormatter.string(from: now)
        
        return Output(nowISO: iso, timezone: tz.identifier)
    }
    
    private static let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.timeZone = .current
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
}



public struct GetDocumentTool: Tool {
    
    public let name = "get_doc"
    public let description: String = "Fetch the canonical text for a document identified by sourceId from the local RAG database. Call this only with IDs returned by search_rag."
    
     
    @Generable
    public struct Arguments {
        @Guide(description: "The document's sourceId (as returned by search_rag).")
        public var sourceId: String

        @Guide(description: "Optional starting page (1-based) to anchor the fetch.", .range(1...1_000_000))
        public var startPage: Int? = nil

        @Guide(description: "Optional short anchor string (e.g., a snippet from a hit) to target a passage near this text.")
        public var anchor: String? = nil

        @Guide(description: "Neighbor expansion (±chunks) around the anchor/start. Non-negative.", .range(0...8))
        public var expand: Int = 2

        @Guide(description: "Maximum number of characters to return. Set null to disable. (200–100,000).", .range(200...100_000))
        public var maxChars: Int? = 8_000
    }

    @Generable
    public struct Output {
        public var sourceId: String
        public var displayName: String
        public var startPage: Int?
        public var endPage: Int?
        public var text: String
        public var chunkIds: [String]
    }
    
    
    public func call(arguments: Arguments) async throws -> Output {
        let engine = try await Engine.shared()

        let safeExpand = max(0, arguments.expand)

        let fetch = try await engine.getDocument(
            sourceId: arguments.sourceId,
            startPage: arguments.startPage,
            anchor: arguments.anchor,
            expand: safeExpand,
            maxChars: arguments.maxChars
        )

        return Output(
            sourceId: fetch.sourceId,
            displayName: fetch.displayName,
            startPage: fetch.startPage,
            endPage: fetch.endPage,
            text: fetch.text,
            chunkIds: fetch.chunkIds
        )
    }
}



public class LLMEngine {
    
    private static let systemPolicy: String = """
    You are ScheduleAI’s on-device planning and drafting assistant. Answer concisely and ground responses in the user’s local data via tools.

    Policy:
    - When a request involves notes, docs, emails, saved opportunities, or schedule content, FIRST call `search_rag` with a focused query.
    - If specific result IDs look promising, call `get_doc` to expand before drafting.
    - Prefer quoting exact snippets and include a brief “Sources:” list using [sourceId]/titles.
    - If time data is needed, call "now" to obtain time data.
    - If the query is ambiguous or context is insufficient, ask one short clarifying question rather than guessing.
    - For structured outputs (e.g., checklists/schedules), return valid JSON that conforms to the provided schema.
    - Keep answers brief, on-device, and do not invent facts beyond retrieved context.
    """
    
    private let ragEngine: Engine
    public let modelSession: LanguageModelSession

    
    private init(ragEngine: Engine, modelSession: LanguageModelSession) {
        self.ragEngine = ragEngine
        self.modelSession = modelSession
    }
    
    private static var _shared: LLMEngine?

    public static func shared() async throws -> LLMEngine {
        if let existing = _shared { return existing }
        
        do {
            let rag = try await Engine.shared()
            let session = LanguageModelSession(
                tools: [
                    RagSearchTool(),
                    GetDocumentTool(),
                    CurrentTimeTool()
                ],
                instructions: LLMEngine.systemPolicy
            )
            let engine = LLMEngine(ragEngine: rag, modelSession: session)
            _shared = engine
            return engine
        } catch {
            throw LLMEngineError.engineError
        }
    }
}

