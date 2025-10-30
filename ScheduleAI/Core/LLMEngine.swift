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
            description: "Retrieval strategy to use. Common values: 'keyword', 'semantic', or 'hybrid'. Defaults to 'hybrid'."
        )
        public var mode: SearchMode = .hybrid(expand: 1, wBM25: 0.5)
    }
    
    public func call(arguments: Arguments) async throws -> [RetrievedResult] {
        let engine = try await Engine.shared()
        switch arguments.mode {
        case .hybrid(let expand, let wBM25):
            return try engine.searchHybrid(
                arguments.query,
                in: arguments.in_,
                limit: arguments.top_k,
                expand: expand,
                wBM25: wBM25
            )
        case .keyword:
            // Approximate pure keyword by using hybrid with full BM25 weight and no expansion
            return try engine.searchHybrid(
                arguments.query,
                in: arguments.in_,
                limit: arguments.top_k,
                expand: 0,
                wBM25: 1.0
            )
        case .semantic:
            // Approximate pure semantic by using hybrid with cosine-only weight
            return try engine.searchHybrid(
                arguments.query,
                in: arguments.in_,
                limit: arguments.top_k,
                expand: 1,
                wBM25: 0.0
            )
        }
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

