//
//  FolioDocument.swift
//  ScheduleAI
//
//  Created by Tai Wong on 10/16/25.
//

import Foundation
import CryptoKit

public enum DocumentHasher {
    public static func sha256Hex(of data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

public enum DocumentKind: String, Codable, CaseIterable {
    case pdf
    case text
    case markdown
    case html
}

public enum DocumentStatus: Equatable, Codable {
    case idle
    case queued
    case extracting
    case ocr
    case chunking
    case writing
    case completed
    case failed(String) // store last error
}

public struct DocumentSummary: Identifiable, Codable, Equatable {
    public let id: String
    public var title: String
    public var kind: DocumentKind
    public var fileURL: URL?
    public var fileSize: Int64
    public var updatedAt: Date
    public var createdAt: Date
    public var chunkCount: Int
    public var pageCount: Int?
    public var status: DocumentStatus

    // Organization & scoping
    public var tags: [String]
    public var course: String?
    public var term: String?

    // Versioning & provenance
    public var checksum: String?
    public var contentVersion: Int
    public var embeddingVersion: Int

    // Extensibility
    public var metadata: [String: String]

    public init(
        id: String,
        title: String,
        kind: DocumentKind,
        fileURL: URL?,
        fileSize: Int64,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        chunkCount: Int = 0,
        pageCount: Int? = nil,
        status: DocumentStatus = .idle,
        tags: [String] = [],
        course: String? = nil,
        term: String? = nil,
        checksum: String? = nil,
        contentVersion: Int = 1,
        embeddingVersion: Int = 1,
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.title = title
        self.kind = kind
        self.fileURL = fileURL
        self.fileSize = fileSize
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.chunkCount = chunkCount
        self.pageCount = pageCount
        self.status = status
        self.tags = tags
        self.course = course
        self.term = term
        self.checksum = checksum
        self.contentVersion = contentVersion
        self.embeddingVersion = embeddingVersion
        self.metadata = metadata
    }
}

public struct EmailMessage: Sendable {
    public var subject: String
    public var body: String
    public var from: String
    public var to: [String]
    public var cc: [String] = []
    public var date: Date
    public var messageID: String?
}

public struct CalendarEvent: Sendable {
    public var title: String
    public var notes: String?
    public var startDate: Date
    public var endDate: Date
    public var location: String?
    public var uid: String?
}
