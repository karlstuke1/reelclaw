import AVFoundation
import UIKit

actor VideoThumbnailService {
    static let shared = VideoThumbnailService()

    private let cache = NSCache<NSURL, UIImage>()
    private var inFlight: [NSURL: Task<UIImage, Error>] = [:]

    init() {
        cache.countLimit = 250
    }

    func thumbnail(for url: URL, maximumSize: CGSize = CGSize(width: 320, height: 320)) async throws -> UIImage {
        let key = url as NSURL
        if let cached = cache.object(forKey: key) {
            return cached
        }

        if let existing = inFlight[key] {
            return try await existing.value
        }

        let task = Task.detached(priority: .utility) { () throws -> UIImage in
            try Task.checkCancellation()

            let asset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = maximumSize

            // Pick a frame slightly in to avoid black first frames.
            let duration = try? await asset.load(.duration)
            let seconds = duration.map { CMTimeGetSeconds($0) } ?? 0
            let t = seconds.isFinite && seconds > 0 ? min(0.5, seconds * 0.25) : 0
            let time = CMTime(seconds: max(0.0, t), preferredTimescale: 600)
            return try await VideoThumbnailService.thumbnailImage(generator: generator, time: time)
        }

        inFlight[key] = task
        do {
            let img = try await task.value
            cache.setObject(img, forKey: key)
            inFlight[key] = nil
            return img
        } catch {
            inFlight[key] = nil
            throw error
        }
    }

    func clear() {
        cache.removeAllObjects()
        inFlight.removeAll()
    }

    private static func thumbnailImage(generator: AVAssetImageGenerator, time: CMTime) async throws -> UIImage {
        try await withCheckedThrowingContinuation { cont in
            var didResume = false
            let times = [NSValue(time: time)]

            generator.generateCGImagesAsynchronously(forTimes: times) { _, cgImage, _, result, error in
                if didResume { return }
                didResume = true

                if let error {
                    cont.resume(throwing: error)
                    return
                }
                guard let cgImage, result == .succeeded else {
                    cont.resume(throwing: CocoaError(.fileReadUnknown))
                    return
                }
                cont.resume(returning: UIImage(cgImage: cgImage))
            }
        }
    }
}
