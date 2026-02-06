import CryptoKit
import Foundation

enum FileHasher {
    enum HashError: LocalizedError {
        case readFailed

        var errorDescription: String? {
            switch self {
            case .readFailed:
                return "Couldn’t read file to hash."
            }
        }
    }

    static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1024 * 1024) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }

        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

