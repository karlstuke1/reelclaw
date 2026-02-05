import Foundation
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

enum VideoImportError: LocalizedError {
    case notFound
    case copyFailed

    var errorDescription: String? {
        switch self {
        case .notFound:
            return "Couldn’t load that video."
        case .copyFailed:
            return "Couldn’t import that video."
        }
    }
}

private func _safeFilenameComponent(_ s: String) -> String {
    let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return "" }
    var out: [Character] = []
    out.reserveCapacity(trimmed.count)
    for ch in trimmed {
        if ch.isLetter || ch.isNumber || ch == "-" || ch == "_" || ch == " " {
            out.append(ch)
        } else {
            out.append("_")
        }
        if out.count >= 80 { break }
    }
    return String(out).trimmingCharacters(in: .whitespacesAndNewlines)
}

private struct ImportedVideo: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let baseName = _safeFilenameComponent(received.file.deletingPathExtension().lastPathComponent)
            let stem = baseName.isEmpty ? UUID().uuidString : baseName
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent(stem)
                .appendingPathExtension(ext)

            var dst = tmp
            if FileManager.default.fileExists(atPath: dst.path) {
                dst = FileManager.default.temporaryDirectory
                    .appendingPathComponent(stem + "-" + UUID().uuidString)
                    .appendingPathExtension(ext)
            }
            do {
                // Fast path: moving within the same volume is usually O(1).
                try FileManager.default.moveItem(at: received.file, to: dst)
            } catch {
                // Fallback: Photos may provide a file that can't be moved (permissions / different volume).
                try FileManager.default.copyItem(at: received.file, to: dst)
            }
            return ImportedVideo(url: dst)
        }
    }
}

extension PhotosPickerItem {
    func loadVideoToTemporaryURL() async throws -> URL {
        guard let imported = try await loadTransferable(type: ImportedVideo.self) else {
            throw VideoImportError.notFound
        }
        return imported.url
    }
}
