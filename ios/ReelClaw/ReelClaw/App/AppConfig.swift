import Foundation

enum AppConfig {
    static let defaultAPIBaseURL = URL(string: "https://y8uexiqu08.execute-api.us-east-1.amazonaws.com")!

    static var apiBaseURL: URL {
        if let raw = UserDefaults.standard.string(forKey: UserDefaultsKeys.apiBaseURL),
           let url = URL(string: raw.trimmingCharacters(in: .whitespacesAndNewlines)),
           url.scheme != nil,
           url.host != nil
        {
            return url
        }
        return defaultAPIBaseURL
    }
}

enum UserDefaultsKeys {
    static let apiBaseURL = "apiBaseURL"
    static let recentJobs = "recentJobs"
}
