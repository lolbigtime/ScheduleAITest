import SwiftUI
import FoundationModels

struct LLMTestView: View {
    @State private var isRunning = false
    @State private var logs: [String] = []
    @State private var didPass: Bool?
    @State private var errorMessage: String?

    private let sampleNote = "LLM test note: The CS 101 midterm is on Oct 20 at 10am in Hall A."
    private let prompt = """
    You are verifying tool integration. Follow this procedure exactly:
    1. Call search_rag to locate when the CS 101 midterm happens.
    2. Call get_doc using one of the returned sourceIds to read the surrounding passage.
    3. Call now so you can report the current local time.
    4. Provide one concise sentence that mentions the exam schedule, its location, and the current time.
    """

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Run an automated test that ensures the on-device LLM is available and is able to call the local search tool.")
                .font(.body)

            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(logs.enumerated()), id: \.offset) { _, log in
                        Text(log)
                            .font(.system(.callout, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.secondary.opacity(0.2)))
            .frame(maxHeight: 320)

            if let didPass {
                Label(
                    didPass ? "LLM tool check completed successfully" : "LLM tool check failed",
                    systemImage: didPass ? "checkmark.seal.fill" : "xmark.octagon.fill"
                )
                .foregroundStyle(didPass ? Color.green : Color.red)
            }

            if let message = errorMessage {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(Color.red)
            }

            Button {
                runLLMTest()
            } label: {
                Label("Run LLM Tool Test", systemImage: "brain.head.profile")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning)

            if isRunning {
                ProgressView().progressViewStyle(.circular)
            }

            Spacer(minLength: 0)
        }
        .padding()
        .navigationTitle("LLM Tool Test")
    }

    private func runLLMTest() {
        isRunning = true
        logs = []
        didPass = nil
        errorMessage = nil

        Task {
            do {
                try await performLLMTest()
                await MainActor.run {
                    didPass = true
                    isRunning = false
                }
            } catch {
                await MainActor.run {
                    didPass = false
                    errorMessage = error.localizedDescription
                    isRunning = false
                }
            }
        }
    }

    private func performLLMTest() async throws {
        await appendLog("Preparing shared engine…")
        let engine = try await Engine.shared()
        await appendLog("Engine ready.")

        await appendLog("Ingesting sample note for retrieval…")
        let ingest = try await engine.importText(text: sampleNote, name: "LLM Tool Fixture")
        await appendLog("Sample note ingested. chars=\(ingest.chars) chunks=\(ingest.chunks)")

        await appendLog("Initializing LLM engine…")
        let llmEngine = try await LLMEngine.shared()
        await appendLog("LLM session ready.")

        await appendLog("Sending prompt to model…")
        let response = try await llmEngine.modelSession.respond(to: prompt)
        await appendLog("Model returned response text.")

        let toolCallBatches: [Transcript.ToolCalls] = response.transcriptEntries.compactMap { entry in
            if case .toolCalls(let calls) = entry {
                return calls
            }
            return nil
        }

        let allToolCalls = toolCallBatches.flatMap { $0 }

        guard !allToolCalls.isEmpty else {
            throw LLMTestFailure("Model produced no tool calls.")
        }

        guard let searchCall = allToolCalls.first(where: { $0.toolName == "search_rag" }) else {
            throw LLMTestFailure("Model did not invoke search_rag tool.")
        }

        await appendLog("Model invoked search_rag (id=\(searchCall.id)).")

        let outputs = response.transcriptEntries.compactMap { entry -> Transcript.ToolOutput? in
            if case .toolOutput(let output) = entry {
                return output
            }
            return nil
        }

        guard let searchOutput = outputs.first(where: { $0.toolName == "search_rag" }) else {
            throw LLMTestFailure("No tool output recorded for search_rag call.")
        }

        let outputSummary = searchOutput.segments.map(toolSegmentSummary).joined(separator: " ")
        await appendLog("search_rag output: \(outputSummary)")

        guard let getDocCall = allToolCalls.first(where: { $0.toolName == "get_doc" }) else {
            throw LLMTestFailure("Model did not invoke get_doc tool.")
        }

        await appendLog("Model invoked get_doc (id=\(getDocCall.id)).")

        guard let getDocOutput = outputs.first(where: { $0.toolName == "get_doc" }) else {
            throw LLMTestFailure("No tool output recorded for get_doc call.")
        }

        let getDocSummary = getDocOutput.segments.map(toolSegmentSummary).joined(separator: " ")
        await appendLog("get_doc output: \(getDocSummary)")

        guard let nowCall = allToolCalls.first(where: { $0.toolName == "now" }) else {
            throw LLMTestFailure("Model did not invoke now tool.")
        }

        await appendLog("Model invoked now (id=\(nowCall.id)).")

        guard let nowOutput = outputs.first(where: { $0.toolName == "now" }) else {
            throw LLMTestFailure("No tool output recorded for now call.")
        }

        let nowSummary = nowOutput.segments.map(toolSegmentSummary).joined(separator: " ")
        await appendLog("now output: \(nowSummary)")

        let answer = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !answer.isEmpty else {
            throw LLMTestFailure("Model returned empty answer.")
        }

        await appendLog("Model final answer: \(answer)")

        let normalizedAnswer = answer.lowercased()
        guard normalizedAnswer.contains("oct") else {
            throw LLMTestFailure("Model answer did not reference the expected date.")
        }

        guard normalizedAnswer.contains("hall") else {
            throw LLMTestFailure("Model answer did not reference the expected location.")
        }

        guard normalizedAnswer.contains(":") else {
            throw LLMTestFailure("Model answer did not include the current time.")
        }

        await appendLog("LLM tool integration test complete.")
    }

    private func toolSegmentSummary(_ segment: Transcript.Segment) -> String {
        switch segment {
        case .text(let textSegment):
            return textSegment.content
        @unknown default:
            return String(describing: segment)
        }
    }

    @MainActor
    private func appendLog(_ message: String) {
        logs.append(message)
    }
}

private struct LLMTestFailure: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
