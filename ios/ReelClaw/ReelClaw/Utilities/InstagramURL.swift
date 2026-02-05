import Foundation

enum InstagramURL {
    static func normalize(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }
        if trimmed.lowercased().hasPrefix("http://") || trimmed.lowercased().hasPrefix("https://") {
            return trimmed
        }
        return "https://" + trimmed
    }

    static func isProbablyInstagramURL(_ raw: String) -> Bool {
        guard let url = URL(string: normalize(raw)),
              let host = url.host?.lowercased()
        else { return false }

        guard host == "instagram.com" || host.hasSuffix(".instagram.com") else {
            return false
        }
        let path = url.path.lowercased()
        return path.contains("/reel/") || path.contains("/p/") || path.contains("/tv/")
    }
}

