import AVFoundation
import Foundation

struct PickedClip: Identifiable, Hashable {
    let id: UUID
    let url: URL
    let filename: String
    let durationSeconds: Double

    var durationLabel: String {
        let total = Int(durationSeconds.rounded(.down))
        let minutes = total / 60
        let seconds = total % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    static func from(fileURL: URL) async throws -> PickedClip {
        try await from(fileURL: fileURL, id: UUID(), filename: fileURL.lastPathComponent)
    }

    static func from(fileURL: URL, id: UUID, filename: String) async throws -> PickedClip {
        let asset = AVURLAsset(url: fileURL)
        let duration = try await asset.load(.duration)
        let seconds = max(0.0, CMTimeGetSeconds(duration))

        return PickedClip(
            id: id,
            url: fileURL,
            filename: filename,
            durationSeconds: seconds.isFinite ? seconds : 0.0
        )
    }
}
