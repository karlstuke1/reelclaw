import Foundation
import Photos

@MainActor
final class SaveAllManager: ObservableObject {
    enum ItemState: Hashable {
        case pending
        case downloading
        case saving
        case done
        case failed(String)
    }

    struct Item: Identifiable, Hashable {
        let id: String
        let title: String
        let remoteURL: URL
        var state: ItemState
    }

    @Published private(set) var items: [Item] = []
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var permissionError: String?

    private var task: Task<Void, Never>?
    private var activeRunId: UUID?

    var totalCount: Int { items.count }

    var doneCount: Int {
        items.reduce(0) { partialResult, item in
            if case .done = item.state {
                return partialResult + 1
            }
            return partialResult
        }
    }

    var failedCount: Int {
        items.reduce(0) { partialResult, item in
            if case .failed = item.state {
                return partialResult + 1
            }
            return partialResult
        }
    }

    var processedCount: Int { doneCount + failedCount }

    var hasFinished: Bool {
        totalCount > 0 && !isRunning && processedCount == totalCount
    }

    func start(variants: [VariantsResponse.Variant]) {
        guard !isRunning else { return }

        permissionError = nil
        items = variants.enumerated().map { idx, v in
            let baseTitle = (v.title ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let title = baseTitle.isEmpty ? "Variation \(idx + 1)" : baseTitle
            return Item(
                id: v.id,
                title: title,
                remoteURL: v.videoUrl,
                state: .pending
            )
        }

        beginRun()
    }

    func retryFailed() {
        guard !isRunning else { return }
        permissionError = nil

        var hadFailure = false
        for idx in items.indices {
            if case .failed = items[idx].state {
                items[idx].state = .pending
                hadFailure = true
            }
        }
        guard hadFailure else { return }
        beginRun()
    }

    func cancel() {
        activeRunId = nil
        task?.cancel()
        task = nil
        isRunning = false
    }

    func dismiss() {
        cancel()
        permissionError = nil
        items = []
    }

    private func beginRun() {
        guard !items.isEmpty else { return }

        isRunning = true
        let runId = UUID()
        activeRunId = runId
        task?.cancel()
        task = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            await self.runBatch(runId: runId, maxConcurrent: 3)
        }
    }

    nonisolated private func runBatch(runId: UUID, maxConcurrent: Int) async {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        let isStillActive = await MainActor.run { self.activeRunId == runId }
        guard isStillActive else { return }

        guard status == .authorized || status == .limited else {
            await MainActor.run {
                guard self.activeRunId == runId else { return }
                self.permissionError = "Enable Photos access to save videos."
                self.isRunning = false
            }
            return
        }

        let snapshot: [Item] = await MainActor.run {
            guard self.activeRunId == runId else { return [] }
            return self.items.filter { item in
                if case .pending = item.state { return true }
                return false
            }
        }
        guard !snapshot.isEmpty else {
            await MainActor.run {
                guard self.activeRunId == runId else { return }
                self.isRunning = false
            }
            return
        }

        let limit = max(1, maxConcurrent)
        await withTaskGroup(of: Void.self) { group in
            var nextIndex = 0

            func addNext() {
                guard nextIndex < snapshot.count else { return }
                let item = snapshot[nextIndex]
                nextIndex += 1
                group.addTask { [weak self] in
                    guard let self else { return }
                    await self.process(item, runId: runId)
                }
            }

            for _ in 0..<min(limit, snapshot.count) {
                addNext()
            }
            while await group.next() != nil {
                addNext()
            }
        }

        await MainActor.run {
            guard self.activeRunId == runId else { return }
            self.isRunning = false
        }
    }

    nonisolated private func process(_ item: Item, runId: UUID) async {
        do {
            try Task.checkCancellation()
            await MainActor.run { self.setState(id: item.id, to: .downloading, runId: runId) }

            let localURL = try await Self.downloadVideo(from: item.remoteURL, id: item.id)
            defer { try? FileManager.default.removeItem(at: localURL) }

            try Task.checkCancellation()
            await MainActor.run { self.setState(id: item.id, to: .saving, runId: runId) }

            try await Self.saveVideoToPhotos(fileURL: localURL)
            await MainActor.run { self.setState(id: item.id, to: .done, runId: runId) }
        } catch is CancellationError {
            await MainActor.run { self.setState(id: item.id, to: .pending, runId: runId) }
        } catch {
            let message = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            await MainActor.run { self.setState(id: item.id, to: .failed(message), runId: runId) }
        }
    }

    private func setState(id: String, to state: ItemState, runId: UUID) {
        guard activeRunId == runId else { return }
        guard let idx = items.firstIndex(where: { $0.id == id }) else { return }
        items[idx].state = state
    }

    nonisolated private static func downloadVideo(from remoteURL: URL, id: String) async throws -> URL {
        let (tempURL, _) = try await URLSession.shared.download(from: remoteURL)
        let ext = remoteURL.pathExtension.isEmpty ? "mp4" : remoteURL.pathExtension
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("reelclaw-\(id)")
            .appendingPathExtension(ext)

        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.moveItem(at: tempURL, to: dest)
        return dest
    }

    nonisolated private static func saveVideoToPhotos(fileURL: URL) async throws {
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: fileURL)
        }
    }
}
