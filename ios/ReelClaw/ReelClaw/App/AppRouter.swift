import Foundation

enum AppTab: String, Hashable {
    case create
    case jobs
    case settings
}

@MainActor
final class AppRouter: ObservableObject {
    @Published var selectedTab: AppTab = AppRouter.restoreSelectedTab() {
        didSet {
            AppRouter.persistSelectedTab(selectedTab)
        }
    }
    @Published var pendingJobIdToOpen: String?

    init() {
        NotificationCenter.default.addObserver(
            forName: .reelclawOpenJob,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let self else { return }
            guard let jobId = note.object as? String, !jobId.isEmpty else { return }
            self.selectedTab = .jobs
            self.pendingJobIdToOpen = jobId
        }
    }

    private static func restoreSelectedTab() -> AppTab {
        let raw = UserDefaults.standard.string(forKey: UserDefaultsKeys.selectedTab) ?? ""
        return AppTab(rawValue: raw) ?? .create
    }

    private static func persistSelectedTab(_ tab: AppTab) {
        UserDefaults.standard.setValue(tab.rawValue, forKey: UserDefaultsKeys.selectedTab)
    }

    func consumePendingJobIdToOpen() -> String? {
        let id = pendingJobIdToOpen
        pendingJobIdToOpen = nil
        return id
    }
}
