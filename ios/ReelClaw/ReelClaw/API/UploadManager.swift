import Foundation

enum UploadError: LocalizedError {
    case invalidResponse
    case http(Int)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Upload failed: invalid response."
        case .http(let code):
            return "Upload failed (HTTP \(code))."
        }
    }
}

/// Background uploader for large video files (S3 presigned PUT or local API PUT).
final class UploadManager: NSObject {
    static let shared = UploadManager()

    private let sessionIdentifier = "com.reelclaw.uploads"

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.background(withIdentifier: sessionIdentifier)
        config.waitsForConnectivity = true
        config.allowsExpensiveNetworkAccess = true
        config.allowsConstrainedNetworkAccess = true
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    private var continuations: [Int: CheckedContinuation<Void, Error>] = [:]
    private var backgroundCompletionHandler: (() -> Void)?

    func setBackgroundCompletionHandler(_ handler: @escaping () -> Void) {
        backgroundCompletionHandler = handler
    }

    func upload(
        fileURL: URL,
        to uploadURL: URL,
        method: String = "PUT",
        headers: [String: String]
    ) async throws {
        var req = URLRequest(url: uploadURL)
        req.httpMethod = method
        for (k, v) in headers {
            req.setValue(v, forHTTPHeaderField: k)
        }

        let task = session.uploadTask(with: req, fromFile: fileURL)
        try await withCheckedThrowingContinuation { continuation in
            continuations[task.taskIdentifier] = continuation
            task.resume()
        }
    }
}

extension UploadManager: URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let cont = continuations.removeValue(forKey: task.taskIdentifier) else {
            return
        }

        if let error {
            cont.resume(throwing: error)
            return
        }

        guard let http = task.response as? HTTPURLResponse else {
            cont.resume(throwing: UploadError.invalidResponse)
            return
        }

        guard (200...299).contains(http.statusCode) else {
            cont.resume(throwing: UploadError.http(http.statusCode))
            return
        }

        cont.resume()
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        backgroundCompletionHandler?()
        backgroundCompletionHandler = nil
    }
}

