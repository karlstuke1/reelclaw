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

private struct ImportedVideo: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let ext = received.file.pathExtension.isEmpty ? "mov" : received.file.pathExtension
            let tmp = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString)
                .appendingPathExtension(ext)

            if FileManager.default.fileExists(atPath: tmp.path) {
                try FileManager.default.removeItem(at: tmp)
            }
            try FileManager.default.copyItem(at: received.file, to: tmp)
            return ImportedVideo(url: tmp)
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
