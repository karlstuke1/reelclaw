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
        let asset = AVURLAsset(url: fileURL)
        let duration = try await asset.load(.duration)
        let seconds = max(0.0, CMTimeGetSeconds(duration))

        return PickedClip(
            id: UUID(),
            url: fileURL,
            filename: fileURL.lastPathComponent,
            durationSeconds: seconds.isFinite ? seconds : 0.0
        )
    }
}
