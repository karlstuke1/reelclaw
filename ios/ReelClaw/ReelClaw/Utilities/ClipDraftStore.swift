import Foundation

enum ClipDraftStore {
    struct Draft: Codable {
        var version: Int = 1
        var clips: [DraftClip] = []
    }

    struct DraftClip: Codable, Hashable {
        let id: UUID
        let relativePath: String
        let displayName: String
        let durationSeconds: Double
    }

    struct ImportSession: Sendable {
        let id: UUID
        let tmpClipsDir: URL
    }

    static func loadDraft() -> Draft? {
        do {
            let url = try draftFileURL()
            guard FileManager.default.fileExists(atPath: url.path) else { return nil }
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            return try decoder.decode(Draft.self, from: data)
        } catch {
            return nil
        }
    }

    static func fileURL(for clip: DraftClip) -> URL? {
        do {
            let root = try rootDir()
            let url = root.appendingPathComponent(clip.relativePath)
            return FileManager.default.fileExists(atPath: url.path) ? url : nil
        } catch {
            return nil
        }
    }

    static func beginImportSession() throws -> ImportSession {
        let root = try rootDir()
        let tmpRoot = root.appendingPathComponent("tmp", isDirectory: true)
        let sessionId = UUID()
        let sessionDir = tmpRoot.appendingPathComponent(sessionId.uuidString, isDirectory: true)
        let clipsDir = sessionDir.appendingPathComponent("clips", isDirectory: true)

        try FileManager.default.createDirectory(at: clipsDir, withIntermediateDirectories: true)
        return ImportSession(id: sessionId, tmpClipsDir: clipsDir)
    }

    static func discardImportSession(_ session: ImportSession) {
        do {
            let root = try rootDir()
            let sessionDir = root
                .appendingPathComponent("tmp", isDirectory: true)
                .appendingPathComponent(session.id.uuidString, isDirectory: true)
            if FileManager.default.fileExists(atPath: sessionDir.path) {
                try FileManager.default.removeItem(at: sessionDir)
            }
        } catch {
            // Ignore cleanup errors.
        }
    }

    static func commitImportSession(_ session: ImportSession, clips: [DraftClip]) throws {
        let root = try rootDir()

        let clipsDir = root.appendingPathComponent("clips", isDirectory: true)
        if FileManager.default.fileExists(atPath: clipsDir.path) {
            try FileManager.default.removeItem(at: clipsDir)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        // Move the fully-imported tmp clips into place.
        try FileManager.default.moveItem(at: session.tmpClipsDir, to: clipsDir)

        // Remove the now-empty session dir.
        let sessionDir = root
            .appendingPathComponent("tmp", isDirectory: true)
            .appendingPathComponent(session.id.uuidString, isDirectory: true)
        if FileManager.default.fileExists(atPath: sessionDir.path) {
            try? FileManager.default.removeItem(at: sessionDir)
        }

        let draft = Draft(version: 1, clips: clips)
        try saveDraft(draft)
    }

    static func clear() {
        do {
            let root = try rootDir()
            if FileManager.default.fileExists(atPath: root.path) {
                try FileManager.default.removeItem(at: root)
            }
        } catch {
            // Ignore cleanup errors.
        }
    }

    static func moveImportedFile(from src: URL, to dst: URL) throws {
        if FileManager.default.fileExists(atPath: dst.path) {
            try FileManager.default.removeItem(at: dst)
        }
        do {
            try FileManager.default.moveItem(at: src, to: dst)
        } catch {
            // Fallback in case move fails across volumes.
            try FileManager.default.copyItem(at: src, to: dst)
            try? FileManager.default.removeItem(at: src)
        }
    }

    static func makeStoredFilename(id: UUID, sourceExtension: String) -> String {
        let ext = sourceExtension.isEmpty ? "mov" : sourceExtension
        return "clip-\(id.uuidString).\(ext)"
    }

	    private static func rootDir() throws -> URL {
	        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
	        guard let base else { throw CocoaError(.fileNoSuchFile) }
	        var root = base.appendingPathComponent("ReelClaw", isDirectory: true).appendingPathComponent("draft_clips", isDirectory: true)
	        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
	        var resourceValues = URLResourceValues()
	        resourceValues.isExcludedFromBackup = true
	        try? root.setResourceValues(resourceValues)
	        return root
	    }

    private static func draftFileURL() throws -> URL {
        let root = try rootDir()
        return root.appendingPathComponent("draft.json", isDirectory: false)
    }

    private static func saveDraft(_ draft: Draft) throws {
        let url = try draftFileURL()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(draft)
        try data.write(to: url, options: [.atomic])
    }
}
