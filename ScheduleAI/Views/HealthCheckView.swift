import SwiftUI
import Folio
import GRDB

struct HealthCheckView: View {
    @State private var isRunning = false
    @State private var logs: [String] = []
    @State private var didPass: Bool?
    @State private var errorMessage: String?

    private let sampleNote = "This is a sample note about CS 101 midterm on Oct 20."

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Run an automated smoke test across engine bring-up, ingestion, database persistence, embeddings, and hybrid retrieval.")
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
                Label(didPass ? "Health check completed successfully" : "Health check failed", systemImage: didPass ? "checkmark.seal.fill" : "xmark.octagon.fill")
                    .foregroundStyle(didPass ? Color.green : Color.red)
            }

            if let message = errorMessage {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(Color.red)
            }

            Button {
                runHealthCheck()
            } label: {
                Label("Run Health Check", systemImage: "stethoscope")
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
        .navigationTitle("RAG Health Check")
    }

    private func runHealthCheck() {
        isRunning = true
        logs = []
        didPass = nil
        errorMessage = nil

        Task {
            do {
                try await performHealthCheck()
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

    private func performHealthCheck() async throws {
        await appendLog("Initializing engine…")
        let engine = try await Engine()
        await appendLog("Engine ready.")

        let textName = "Health Check Note"
        let emailSourceId = "email:health-check"
        let scheduleSourceId = "schedule:health-check"

        await appendLog("Ingesting sample note…")
        let textResult = try await engine.importText(text: sampleNote, name: textName)
        await appendLog("Text ingestion completed. chars=\(textResult.chars) chunks=\(textResult.chunks)")

        let now = Date()
        let emails: [EmailMessage] = [
            EmailMessage(subject: "Lab opportunity in NLP", body: "Hi Alex, we have openings for undergrads this fall.", from: "prof@university.edu", to: ["alex@school.edu"], cc: [], date: now.addingTimeInterval(-86400 * 7), messageID: "health-1"),
            EmailMessage(subject: "CS 101 Midterm Details", body: "Midterm is on Oct 20, covers chapters 1-5.", from: "ta@school.edu", to: ["alex@school.edu"], cc: [], date: now.addingTimeInterval(-86400 * 3), messageID: "health-2"),
            EmailMessage(subject: "Internship application", body: "Please submit your resume and cover letter by Nov 1.", from: "hr@company.com", to: ["alex@school.edu"], cc: [], date: now, messageID: "health-3")
        ]

        await appendLog("Ingesting sample emails…")
        let emailResult = try await engine.importEmails(emails, sourceId: emailSourceId, name: "Gmail (health check)")
        await appendLog("Email ingestion completed. items=\(emailResult.items) chunks=\(emailResult.chunks)")

        let events: [CalendarEvent] = [
            CalendarEvent(title: "CS 101 Lecture", notes: "Room 12", startDate: now.addingTimeInterval(3600), endDate: now.addingTimeInterval(7200), location: "Hall A", uid: "health-ev-1"),
            CalendarEvent(title: "Study Session", notes: "Midterm review", startDate: now.addingTimeInterval(86400 * 2), endDate: now.addingTimeInterval(86400 * 2 + 5400), location: nil, uid: "health-ev-2"),
            CalendarEvent(title: "NLP Lab Info Session", notes: "With Prof. Smith", startDate: now.addingTimeInterval(86400 * 5), endDate: now.addingTimeInterval(86400 * 5 + 3600), location: "Lab 3", uid: "health-ev-3")
        ]

        await appendLog("Ingesting sample schedule…")
        let scheduleResult = try await engine.importSchedule(events, sourceId: scheduleSourceId, name: "Schedule (health check)")
        await appendLog("Schedule ingestion completed. items=\(scheduleResult.items) chunks=\(scheduleResult.chunks)")

        let totalChunks = textResult.chunks + emailResult.chunks + scheduleResult.chunks
        try await verifyDatabase(expectedChunks: totalChunks, textSourceId: sampleTextSourceId(), emailSourceId: emailSourceId, scheduleSourceId: scheduleSourceId)

        try await verifyEmbeddings()

        try await verifyRetrieval(using: engine)

        await appendLog("Health check complete.")
    }

    private func sampleTextSourceId() -> String {
        let data = Data(sampleNote.utf8)
        let hash = DocumentHasher.sha256Hex(of: data)
        return "text:" + hash
    }

    private func verifyDatabase(expectedChunks: Int, textSourceId: String, emailSourceId: String, scheduleSourceId: String) async throws {
        await appendLog("Verifying Folio database…")
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dbURL = support.appendingPathComponent("Folio/folio.sqlite")
        let dbQueue = try DatabaseQueue(path: dbURL.path)

        let arguments: StatementArguments = [textSourceId, emailSourceId, scheduleSourceId]

        let chunkCount: Int = try await dbQueue.read { db in
            try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM doc_chunks WHERE source_id IN (?, ?, ?)", arguments: arguments) ?? 0
        }

        guard chunkCount == expectedChunks else {
            throw HealthCheckFailure("Chunk count mismatch. Expected \(expectedChunks), found \(chunkCount)")
        }

        let ftsCount: Int = try await dbQueue.read { db in
            try Int.fetchOne(
                db,
                sql: "SELECT COUNT(*) FROM doc_chunks_fts WHERE rowid IN (SELECT rowid FROM doc_chunks WHERE source_id IN (?, ?, ?))",
                arguments: arguments
            ) ?? 0
        }

        guard ftsCount == expectedChunks else {
            throw HealthCheckFailure("FTS row count mismatch. Expected \(expectedChunks), found \(ftsCount)")
        }

        let prefixedChunks: Int = try await dbQueue.read { db in
            try Int.fetchOne(db, sql: "SELECT COUNT(*) FROM doc_chunks WHERE source_id IN (?, ?, ?) AND section_title <> ''", arguments: arguments) ?? 0
        }

        guard prefixedChunks > 0 else {
            throw HealthCheckFailure("No foundation-model prefixes found for recent ingests.")
        }

        await appendLog("Database verification passed.")
    }

    private func verifyEmbeddings() async throws {
        await appendLog("Verifying embedding model…")
        guard let tokenizerURL = Bundle.main.url(forResource: "QwenTokenizer", withExtension: nil) else {
            throw HealthCheckFailure("Tokenizer resources missing in bundle.")
        }
        guard let modelURL = Bundle.main.url(forResource: "qwenEmbedding", withExtension: "mlmodelc") ?? Bundle.main.url(forResource: "qwenEmbedding", withExtension: "mlpackage") else {
            throw HealthCheckFailure("Embedding model missing in bundle.")
        }

        let embedder = try await QwenCoreMLEmbedder(modelURL: modelURL, tokenizerURL: tokenizerURL, seqLen: 512, storeDim: 512, outputName: "var_3996")
        let vector = try embedder.embed("diagnostic text")

        guard vector.count == 512 else {
            throw HealthCheckFailure("Unexpected embedding dimensionality: \(vector.count)")
        }

        await appendLog("Embedding model verification passed.")
    }

    private func verifyRetrieval(using engine: Engine) async throws {
        await appendLog("Running retrieval checks…")
        let contextHits = try engine.search("CS 101 midterm", in: nil, topK: 5, mode: .withContext(expand: 300))
        let hybridHits = try engine.search("CS 101 midterm", in: nil, topK: 5, mode: .hybrid(expand: 0, wBM25: 0.3))

        guard !contextHits.isEmpty else {
            throw HealthCheckFailure("Context search returned no results.")
        }

        guard !hybridHits.isEmpty else {
            throw HealthCheckFailure("Hybrid search returned no results.")
        }

        guard contextHits.map(\.score) != hybridHits.map(\.score) else {
            throw HealthCheckFailure("Hybrid scores match context-only scores; embeddings may be inactive.")
        }

        await appendLog("Retrieval verification passed.")
    }

    @MainActor
    private func appendLog(_ message: String) {
        logs.append(message)
    }
}

private struct HealthCheckFailure: LocalizedError {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? { message }
}
