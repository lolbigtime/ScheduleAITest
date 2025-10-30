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


public class LLMEngine {
    
    private static let systemPolicy: String = """
    You are ScheduleAI’s on-device planning and drafting assistant. Answer concisely and ground responses in the user’s local data via tools.

    Policy:
    - When a request involves notes, docs, emails, saved opportunities, or schedule content, FIRST call `search_rag` with a focused query.
    - If specific result IDs look promising, call `get_doc` to expand before drafting.
    - Prefer quoting exact snippets and include a brief “Sources:” list using [sourceId]/titles.
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
                    RagSearchTool()
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

