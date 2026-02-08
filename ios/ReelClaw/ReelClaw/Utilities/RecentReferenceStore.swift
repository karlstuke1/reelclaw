import Foundation

struct RecentReference: Codable, Identifiable, Hashable {
    let id: String
    let url: String
    let lastUsedAt: Date

    var host: String {
        URL(string: url)?.host ?? url
    }
}

@MainActor
final class RecentReferenceStore: ObservableObject {
    @Published private(set) var items: [RecentReference] = []

    private let key = UserDefaultsKeys.recentReferences
    private let maxItems = 20

    init() {
        load()
    }

    func record(url: String) {
        let trimmed = url.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        guard let normalized = Self.normalizeURLString(trimmed) else { return }

        let now = Date()
        var next = items

        // De-dupe by normalized id.
        next.removeAll(where: { $0.id == normalized })
        next.insert(RecentReference(id: normalized, url: normalized, lastUsedAt: now), at: 0)

        if next.count > maxItems {
            next = Array(next.prefix(maxItems))
        }

        items = next
        save()
    }

    func remove(id: String) {
        items.removeAll(where: { $0.id == id })
        save()
    }

    func clear() {
        items = []
        UserDefaults.standard.removeObject(forKey: key)
    }

    private func load() {
        do {
            guard let data = UserDefaults.standard.data(forKey: key) else {
                items = []
                return
            }
            let decoded = try JSONDecoder().decode([RecentReference].self, from: data)
            items = decoded.sorted(by: { $0.lastUsedAt > $1.lastUsedAt })
        } catch {
            items = []
        }
    }

    private func save() {
        do {
            let data = try JSONEncoder().encode(items)
            UserDefaults.standard.setValue(data, forKey: key)
        } catch {
            // Best-effort.
        }
    }

    private static func normalizeURLString(_ raw: String) -> String? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return nil }

        var withScheme = trimmed
        if !withScheme.lowercased().hasPrefix("http://") && !withScheme.lowercased().hasPrefix("https://") {
            withScheme = "https://" + withScheme
        }

        guard let url = URL(string: withScheme),
              let scheme = url.scheme,
              (scheme == "http" || scheme == "https"),
              url.host != nil
        else {
            return nil
        }

        // Drop URL fragments, trim trailing slash for stable IDs.
        var comps = URLComponents(url: url, resolvingAgainstBaseURL: false)
        comps?.fragment = nil
        var normalized = (comps?.url ?? url).absoluteString
        if normalized.count > 1, normalized.hasSuffix("/") {
            normalized.removeLast()
        }
        return normalized
    }
}

