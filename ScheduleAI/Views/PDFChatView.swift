import SwiftUI
import UniformTypeIdentifiers

struct PDFChatView: View {
    private enum MessageRole {
        case user
        case assistant
    }

    private struct ChatMessage: Identifiable {
        let id = UUID()
        let role: MessageRole
        let text: String
    }

    @State private var engine: Engine?
    @State private var llmEngine: LLMEngine?
    @State private var messages: [ChatMessage] = []
    @State private var inputText: String = ""
    @State private var statusText: String = "Loading engines…"
    @State private var isReady: Bool = false
    @State private var isSending: Bool = false
    @State private var showImporter: Bool = false
    @State private var importStatus: String?
    @State private var showImportProgress: Bool = false

    var body: some View {
        VStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Import PDFs")
                    .font(.headline)
                Text("Bring in a PDF first so the on-device retrieval engine has content to reference before you ask questions.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                HStack {
                    Button {
                        showImporter = true
                    } label: {
                        Label("Import PDF", systemImage: "tray.and.arrow.down")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!isReady || showImportProgress)

                    if showImportProgress {
                        ProgressView()
                            .progressViewStyle(.circular)
                    }
                }

                if let importStatus {
                    Text(importStatus)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(RoundedRectangle(cornerRadius: 12).fill(Color(.secondarySystemBackground)))

            Divider()

            VStack(spacing: 12) {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            HStack {
                                if message.role == .assistant { Spacer() }
                                Text(message.text)
                                    .padding(10)
                                    .background(bubbleColor(for: message.role))
                                    .clipShape(RoundedRectangle(cornerRadius: 12))
                                    .foregroundStyle(message.role == .assistant ? .white : .primary)
                                if message.role == .user { Spacer() }
                            }
                        }
                        if isSending {
                            HStack {
                                Spacer()
                                ProgressView()
                                    .progressViewStyle(.circular)
                                Spacer()
                            }
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
                .background(RoundedRectangle(cornerRadius: 12).stroke(Color.secondary.opacity(0.2)))

                HStack(alignment: .bottom, spacing: 12) {
                    TextEditor(text: $inputText)
                        .frame(minHeight: 60, maxHeight: 100)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.3)))
                        .disabled(!isReady || isSending)

                    Button {
                        sendMessage()
                    } label: {
                        Image(systemName: "paperplane.fill")
                            .font(.title3)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!isReady || isSending || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }

            if !statusText.isEmpty {
                Text(statusText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .navigationTitle("PDF Chat")
        .fileImporter(isPresented: $showImporter, allowedContentTypes: [.pdf]) { result in
            handleImport(result: result)
        }
        .task {
            await prepareEngines()
        }
    }

    private func bubbleColor(for role: MessageRole) -> Color {
        switch role {
        case .user:
            return Color(.systemGray6)
        case .assistant:
            return Color.accentColor
        }
    }

    @MainActor
    private func handleImport(result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            guard let engine else {
                statusText = "Engine not ready yet."
                return
            }
            importStatus = "Ingesting \(url.lastPathComponent)…"
            showImportProgress = true
            Task {
                let didAccess = url.startAccessingSecurityScopedResource()
                defer {
                    if didAccess {
                        url.stopAccessingSecurityScopedResource()
                    }
                }
                do {
                    let outcome = try await engine.importPDF(at: url)
                    await MainActor.run {
                        importStatus = "Imported \(url.lastPathComponent). pages: \(outcome.pages) chunks: \(outcome.chunks)"
                        statusText = "Ready to chat over your PDFs."
                        showImportProgress = false
                    }
                } catch {
                    await MainActor.run {
                        importStatus = "Import failed: \(error.localizedDescription)"
                        showImportProgress = false
                    }
                }
            }
        case .failure(let error):
            statusText = "Import canceled: \(error.localizedDescription)"
        }
    }

    private func prepareEngines() async {
        if engine != nil, llmEngine != nil {
            return
        }

        do {
            let ragEngine = try await Engine.shared()
            let llm = try await LLMEngine.shared()
            await MainActor.run {
                engine = ragEngine
                llmEngine = llm
                statusText = "Ready. Import a PDF and start chatting."
                isReady = true
            }
        } catch {
            await MainActor.run {
                statusText = "Failed to start engines: \(error.localizedDescription)"
                isReady = false
            }
        }
    }

    private func sendMessage() {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let llmEngine else { return }

        let userMessage = ChatMessage(role: .user, text: trimmed)
        messages.append(userMessage)
        inputText = ""
        isSending = true
        statusText = "Thinking…"

        let history = messages

        Task {
            do {
                let answer = try await generateResponse(using: llmEngine, history: history)
                let assistantMessage = ChatMessage(role: .assistant, text: answer)
                await MainActor.run {
                    messages.append(assistantMessage)
                    isSending = false
                    statusText = "Ready."
                }
            } catch {
                await MainActor.run {
                    isSending = false
                    statusText = "Response failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func generateResponse(using engine: LLMEngine, history: [ChatMessage]) async throws -> String {
        let prompt = buildPrompt(from: history)
        let response = try await engine.modelSession.respond(to: prompt)
        let answer = response.content.trimmingCharacters(in: .whitespacesAndNewlines)
        if answer.isEmpty {
            throw LLMEngineError.engineError
        }
        return answer
    }

    private func buildPrompt(from history: [ChatMessage]) -> String {
        if history.isEmpty { return "" }
        let segments = history.map { message -> String in
            switch message.role {
            case .user:
                return "User: \(message.text)"
            case .assistant:
                return "Assistant: \(message.text)"
            }
        }
        return segments.joined(separator: "\n\n") + "\n\nAssistant:"
    }
}
