import SwiftUI
import Folio

struct ContentView: View {
    @State private var engine: Engine?

    // Text ingestion
    @State private var textInput: String = "This is a sample note about CS 101 midterm on Oct 20."
    @State private var textName: String = "Sample Note"

    // Search
    @State private var query: String = "CS 101 midterm"
    @State private var results: [RetrievedResult] = []
    @State private var status: String = "Ready"

    // Email/Schedule sample state
    @State private var lastEmailIngest: (items: Int, chunks: Int)?
    @State private var lastScheduleIngest: (items: Int, chunks: Int)?

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Group {
                        Text("Ingest Text").font(.headline)
                        TextField("Name", text: $textName)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                        TextEditor(text: $textInput)
                            .frame(minHeight: 100)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.3)))
                        HStack {
                            Button("Ingest Text") { ingestText() }
                                .buttonStyle(.borderedProminent)
                                .disabled(engine == nil)
                            Spacer()
                        }
                    }

                    Divider()

                    Group {
                        Text("Ingest Sample Emails").font(.headline)
                        Button("Ingest 3 Emails (Gmail)") { ingestSampleEmails() }
                            .buttonStyle(.bordered)
                            .disabled(engine == nil)
                        if let e = lastEmailIngest {
                            Text("Emails ingested: \(e.items), chunks: \(e.chunks)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Divider()

                    Group {
                        Text("Ingest Sample Schedule").font(.headline)
                        Button("Ingest 3 Events (Fall 2025)") { ingestSampleSchedule() }
                            .buttonStyle(.bordered)
                            .disabled(engine == nil)
                        if let s = lastScheduleIngest {
                            Text("Events ingested: \(s.items), chunks: \(s.chunks)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Divider()

                    Group {
                        Text("Search").font(.headline)
                        TextField("Query", text: $query)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                        HStack {
                            Button("Hybrid Search") { runSearch(hybrid: true) }
                                .buttonStyle(.borderedProminent)
                                .disabled(engine == nil)
                            Button("Context Search") { runSearch(hybrid: false) }
                                .buttonStyle(.bordered)
                                .disabled(engine == nil)
                        }
                        Text(status)
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        ForEach(Array(results.enumerated()), id: \.offset) { idx, r in
                            VStack(alignment: .leading, spacing: 6) {
                                Text("#\(idx + 1) • score: \(String(format: "%.3f", r.score))")
                                    .font(.subheadline).bold()
                                Text(r.excerpt.isEmpty ? r.text : r.excerpt)
                                    .font(.callout)
                                    .lineLimit(6)
                                Text("source: \(r.sourceId) page: \(r.startPage.map(String.init) ?? "-")")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(8)
                            .background(RoundedRectangle(cornerRadius: 8).fill(Color(.secondarySystemBackground)))
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("ScheduleAI Test")
            .task {
                status = "Initializing engine…"
                do {
                    engine = try await Engine()
                    status = "Ready"
                } catch {
                    status = "Engine init failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Actions

    private func ingestText() {
        guard let engine = engine else {
            status = "Engine not ready"
            return
        }
        status = "Ingesting text…"
        Task {
            do {
                let res = try await engine.importText(text: textInput, name: textName)
                status = "Text ingested. chars: \(res.chars) chunks: \(res.chunks)"
            } catch {
                status = "Ingest failed: \(error.localizedDescription)"
            }
        }
    }

    private func ingestSampleEmails() {
        guard let engine = engine else {
            status = "Engine not ready"
            return
        }
        status = "Ingesting emails…"
        let now = Date()
        let emails: [EmailMessage] = [
            EmailMessage(subject: "Lab opportunity in NLP", body: "Hi Alex, we have openings for undergrads this fall.", from: "prof@university.edu", to: ["alex@school.edu"], cc: [], date: now.addingTimeInterval(-86400 * 7), messageID: "msg-1"),
            EmailMessage(subject: "CS 101 Midterm Details", body: "Midterm is on Oct 20, covers chapters 1-5.", from: "ta@school.edu", to: ["alex@school.edu"], cc: [], date: now.addingTimeInterval(-86400 * 3), messageID: "msg-2"),
            EmailMessage(subject: "Internship application", body: "Please submit your resume and cover letter by Nov 1.", from: "hr@company.com", to: ["alex@school.edu"], cc: [], date: now, messageID: "msg-3")
        ]
        Task {
            do {
                let res = try await engine.importEmails(emails, sourceId: "email:gmail:demo", name: "Gmail (demo)")
                lastEmailIngest = res
                status = "Emails ingested. items: \(res.items) chunks: \(res.chunks)"
            } catch {
                status = "Email ingest failed: \(error.localizedDescription)"
            }
        }
    }

    private func ingestSampleSchedule() {
        guard let engine = engine else {
            status = "Engine not ready"
            return
        }
        status = "Ingesting schedule…"
        let now = Date()
        let events: [CalendarEvent] = [
            CalendarEvent(title: "CS 101 Lecture", notes: "Room 12", startDate: now.addingTimeInterval(3600), endDate: now.addingTimeInterval(7200), location: "Hall A", uid: "ev-1"),
            CalendarEvent(title: "Study Session", notes: "Midterm review", startDate: now.addingTimeInterval(86400 * 2), endDate: now.addingTimeInterval(86400 * 2 + 5400), location: nil, uid: "ev-2"),
            CalendarEvent(title: "NLP Lab Info Session", notes: "With Prof. Smith", startDate: now.addingTimeInterval(86400 * 5), endDate: now.addingTimeInterval(86400 * 5 + 3600), location: "Lab 3", uid: "ev-3")
        ]
        Task {
            do {
                let res = try await engine.importSchedule(events, sourceId: "schedule:fall-2025", name: "Fall 2025 Schedule (demo)")
                lastScheduleIngest = res
                status = "Schedule ingested. items: \(res.items) chunks: \(res.chunks)"
            } catch {
                status = "Schedule ingest failed: \(error.localizedDescription)"
            }
        }
    }

    private func runSearch(hybrid: Bool) {
        guard let engine = engine else {
            status = "Engine not ready"
            return
        }
        status = "Searching…"
        do {
            let mode: SearchMode = hybrid ? .hybrid(expand: 0, wBM25: 0.3) : .withContext(expand: 300)
            let hits = try engine.search(query, in: nil, topK: 8, mode: mode)
            results = hits
            status = "Found \(hits.count) results"
        } catch {
            status = "Search failed: \(error.localizedDescription)"
        }
    }
}

#Preview {
    ContentView()
}
