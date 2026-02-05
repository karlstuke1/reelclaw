import AVFoundation
import Foundation

enum VideoCompressionError: LocalizedError {
    case exportFailed
    case unsupported

    var errorDescription: String? {
        switch self {
        case .exportFailed:
            return "Video compression failed."
        case .unsupported:
            return "This video can’t be compressed on this device."
        }
    }
}

enum VideoCompressor {
    static func compressIfNeeded(fileURL: URL, enabled: Bool) async throws -> URL {
        guard enabled else { return fileURL }
        return try await compress(fileURL: fileURL)
    }

    static func compress(fileURL: URL) async throws -> URL {
        let asset = AVURLAsset(url: fileURL)
        guard let exportSession = AVAssetExportSession(asset: asset, presetName: AVAssetExportPreset1280x720) else {
            throw VideoCompressionError.unsupported
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("reelclaw-compressed-\(UUID().uuidString)")
            .appendingPathExtension("mp4")

        if FileManager.default.fileExists(atPath: outputURL.path) {
            try? FileManager.default.removeItem(at: outputURL)
        }

        exportSession.outputURL = outputURL
        exportSession.outputFileType = .mp4
        exportSession.shouldOptimizeForNetworkUse = true

        return try await withCheckedThrowingContinuation { continuation in
            exportSession.exportAsynchronously {
                switch exportSession.status {
                case .completed:
                    continuation.resume(returning: outputURL)
                case .failed, .cancelled:
                    continuation.resume(throwing: exportSession.error ?? VideoCompressionError.exportFailed)
                default:
                    continuation.resume(throwing: VideoCompressionError.exportFailed)
                }
            }
        }
    }
}

