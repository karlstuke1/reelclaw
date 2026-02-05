import Foundation

enum AppTab: Hashable {
    case create
    case jobs
    case settings
}

@MainActor
final class AppRouter: ObservableObject {
    @Published var selectedTab: AppTab = .create
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

    func consumePendingJobIdToOpen() -> String? {
        let id = pendingJobIdToOpen
        pendingJobIdToOpen = nil
        return id
    }
}

