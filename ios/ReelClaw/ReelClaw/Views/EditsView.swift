import SwiftUI

struct EditsView: View {
    private enum Route: Hashable {
        case job(String)
        case variants(String)
    }

    private let uploadStuckThresholdSeconds: TimeInterval = 2 * 60 * 60

    @EnvironmentObject private var session: SessionStore
    @EnvironmentObject private var router: AppRouter

    @State private var path: [Route] = []
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?
    @State private var jobs: [JobSummaryResponse] = []

    @State private var deleteCandidate: JobSummaryResponse?
    @State private var isDeleting: Bool = false
    @State private var showDeleteStuckUploadsDialog: Bool = false
    @State private var isCleaningStuckUploads: Bool = false

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                if isLoading && jobs.isEmpty {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Loading edits…")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let errorMessage, jobs.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "xmark.octagon.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(.red)
                        Text("Couldn’t load edits")
                            .font(.headline)
                        Text(errorMessage)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .textSelection(.enabled)
                        Button("Try again") {
                            Task { await load() }
                        }
                        .buttonStyle(.borderedProminent)
                        .padding(.top, 8)
                    }
                    .padding(24)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 18) {
                            if let errorMessage {
                                Text(errorMessage)
                                    .font(.footnote)
                                    .foregroundStyle(.red)
                                    .textSelection(.enabled)
                            }

                            if !stuckUploadingJobs.isEmpty {
                                sectionHeader("Stuck Uploads")
                                stuckUploadsCard
                            }

                            if !inProgressJobs.isEmpty {
                                sectionHeader("In Progress")
                                VStack(spacing: 12) {
                                    ForEach(inProgressJobs) { job in
                                        Button {
                                            path.append(.job(job.jobId))
                                        } label: {
                                            EditJobCard(job: job)
                                        }
                                        .buttonStyle(.plain)
                                        .contextMenu {
                                            if job.status == .uploading {
                                                Button(role: .destructive) {
                                                    deleteCandidate = job
                                                } label: {
                                                    Label("Delete upload…", systemImage: "trash")
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if !failedJobs.isEmpty {
                                sectionHeader("Needs Attention")
                                VStack(spacing: 12) {
                                    ForEach(failedJobs) { job in
                                        VStack(spacing: 10) {
                                            Button {
                                                path.append(.job(job.jobId))
                                            } label: {
                                                EditJobCard(job: job, style: .failed)
                                            }
                                            .buttonStyle(.plain)
                                            .contextMenu {
                                                Button(role: .destructive) {
                                                    deleteCandidate = job
                                                } label: {
                                                    Label("Delete edit…", systemImage: "trash")
                                                }
                                            }

                                            HStack(spacing: 10) {
                                                Button {
                                                    Task { await retry(jobId: job.jobId) }
                                                } label: {
                                                    Label("Retry", systemImage: "arrow.clockwise")
                                                        .frame(maxWidth: .infinity)
                                                }
                                                .buttonStyle(.borderedProminent)
                                                .disabled(isDeleting || isCleaningStuckUploads)

                                                Button {
                                                    path.append(.job(job.jobId))
                                                } label: {
                                                    Label("Details", systemImage: "info.circle")
                                                        .frame(maxWidth: .infinity)
                                                }
                                                .buttonStyle(.bordered)
                                                .disabled(isDeleting || isCleaningStuckUploads)
                                            }
                                        }
                                    }
                                }
                            }

                            sectionHeader("Completed")
                            if completedJobs.isEmpty {
                                emptyCompletedState
                            } else {
                                LazyVGrid(columns: gridColumns, spacing: 12) {
                                    ForEach(completedJobs) { job in
                                        Button {
                                            path.append(.variants(job.jobId))
                                        } label: {
                                            CompletedTile(job: job)
                                        }
                                        .buttonStyle(.plain)
                                        .contextMenu {
                                            Button(role: .destructive) {
                                                deleteCandidate = job
                                            } label: {
                                                Label("Delete edit…", systemImage: "trash")
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        .padding(12)
                    }
                    .refreshable {
                        await load()
                    }
                }
            }
            .navigationTitle("My Edits")
            .navigationDestination(for: Route.self) { route in
                switch route {
                case .job(let jobId):
                    JobProgressView(jobId: jobId)
                case .variants(let jobId):
                    VariantsView(jobId: jobId)
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await load() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(isLoading || isDeleting || isCleaningStuckUploads)
                }
            }
            .confirmationDialog(
                "Delete edit?",
                isPresented: Binding(
                    get: { deleteCandidate != nil },
                    set: { if !$0 { deleteCandidate = nil } }
                ),
                presenting: deleteCandidate
            ) { job in
                Button("Delete", role: .destructive) {
                    Task { await delete(jobId: job.jobId) }
                }
                Button("Cancel", role: .cancel) {}
            } message: { job in
                switch job.status {
                case .succeeded:
                    Text("This deletes generated videos. Uploaded clips may be retained up to 90 days to speed up re-edits.")
                case .uploading:
                    Text("This removes the stuck upload. Uploaded clips may be retained up to 90 days to speed up re-edits.")
                case .failed:
                    Text("This deletes the job and any generated outputs.")
                default:
                    Text("This deletes the job and any generated outputs.")
                }
            }
            .confirmationDialog("Delete stuck uploads?", isPresented: $showDeleteStuckUploadsDialog) {
                Button("Delete \(stuckUploadingJobs.count) uploads", role: .destructive) {
                    Task { await deleteStuckUploads() }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("These were created more than 2 hours ago and never finished uploading. They won’t process until recreated.")
            }
            .task {
                await load()
                openPendingJobIfAny()
            }
            .onChange(of: router.pendingJobIdToOpen) {
                openPendingJobIfAny()
            }
        }
    }

    private var gridColumns: [GridItem] {
        [
            GridItem(.flexible(minimum: 150), spacing: 12),
            GridItem(.flexible(minimum: 150), spacing: 12),
        ]
    }

    private var inProgressJobs: [JobSummaryResponse] {
        jobs
            .filter { $0.status == .queued || $0.status == .running }
            .sorted(by: newestFirst)
    }

    private var stuckUploadingJobs: [JobSummaryResponse] {
        jobs
            .filter { isUploadStuck($0) }
            .sorted(by: newestFirst)
    }

    private var failedJobs: [JobSummaryResponse] {
        jobs
            .filter { $0.status == .failed }
            .sorted(by: newestFirst)
    }

    private var completedJobs: [JobSummaryResponse] {
        jobs
            .filter { $0.status == .succeeded }
            .sorted(by: newestFirst)
    }

    private func newestFirst(_ a: JobSummaryResponse, _ b: JobSummaryResponse) -> Bool {
        (a.createdAt ?? .distantPast) > (b.createdAt ?? .distantPast)
    }

    private func isUploadStuck(_ job: JobSummaryResponse) -> Bool {
        guard job.status == .uploading else { return false }
        guard let t = job.updatedAt ?? job.createdAt else { return false }
        return Date().timeIntervalSince(t) > uploadStuckThresholdSeconds
    }

    @ViewBuilder
    private func sectionHeader(_ title: String) -> some View {
        HStack {
            Text(title)
                .font(.headline)
            Spacer()
        }
        .padding(.top, 2)
    }

    @ViewBuilder
    private var emptyCompletedState: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("No completed edits yet.")
                .font(.headline)
            Text("When your first edit finishes, it will show up here with a thumbnail.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var stuckUploadsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("\(stuckUploadingJobs.count) upload\(stuckUploadingJobs.count == 1 ? "" : "s") got stuck.")
                .font(.headline)

            Text("These edits were created but never finished uploading, so they won’t process. Delete them to clean up My Edits.")
                .font(.footnote)
                .foregroundStyle(.secondary)

            Button(role: .destructive) {
                showDeleteStuckUploadsDialog = true
            } label: {
                Label("Delete stuck uploads…", systemImage: "trash")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(isDeleting || isCleaningStuckUploads)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color(.separator).opacity(0.15), lineWidth: 1)
        )
    }

    private func openPendingJobIfAny() {
        guard let jobId = router.consumePendingJobIdToOpen() else { return }
        if !jobId.isEmpty {
            path = [.job(jobId)]
        }
    }

    private func load() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })
        do {
            let resp = try await api.listJobs()
            jobs = resp.jobs
        } catch {
            if let apiError = error as? APIError,
               case let .server(code, _) = apiError,
               code == 401
            {
                session.signOut(reason: "Session expired. Please sign in again.")
                return
            }
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    private func deleteStuckUploads() async {
        guard !isCleaningStuckUploads else { return }
        isCleaningStuckUploads = true
        defer { isCleaningStuckUploads = false }

        errorMessage = nil
        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })
        let ids = stuckUploadingJobs.map(\.jobId)
        var failedCount = 0

        for id in ids {
            do {
                try await api.deleteJob(jobId: id)
            } catch {
                if let apiError = error as? APIError,
                   case let .server(code, _) = apiError,
                   code == 401
                {
                    session.signOut(reason: "Session expired. Please sign in again.")
                    return
                }
                failedCount += 1
            }
        }

        await load()
        if failedCount > 0 {
            errorMessage = "Couldn’t delete \(failedCount) stuck upload\(failedCount == 1 ? "" : "s"). Pull to refresh and try again."
        }
    }

    private func retry(jobId: String) async {
        errorMessage = nil
        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })
        do {
            try await api.startJob(jobId: jobId)
            path.append(.job(jobId))
            await load()
        } catch {
            if let apiError = error as? APIError,
               case let .server(code, _) = apiError,
               code == 401
            {
                session.signOut(reason: "Session expired. Please sign in again.")
                return
            }
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    private func delete(jobId: String) async {
        guard !isDeleting else { return }
        isDeleting = true
        defer { isDeleting = false }

        errorMessage = nil
        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })
        do {
            try await api.deleteJob(jobId: jobId)
            deleteCandidate = nil
            await load()
        } catch {
            if let apiError = error as? APIError,
               case let .server(code, _) = apiError,
               code == 401
            {
                session.signOut(reason: "Session expired. Please sign in again.")
                return
            }
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }
}

private enum EditJobCardStyle {
    case normal
    case failed
}

private struct EditJobCard: View {
    let job: JobSummaryResponse
    var style: EditJobCardStyle = .normal

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                statusBadge
                Spacer()
                if let createdAt = job.createdAt {
                    Text(createdAt, style: .relative)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            if let stage = job.stage, !stage.isEmpty {
                Text(stage)
                    .font(.headline)
                    .foregroundStyle(.primary)
            }

            if let msg = job.message, !msg.isEmpty {
                Text(msg)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            if let cur = job.progressCurrent, let total = job.progressTotal, total > 0 {
                ProgressView(value: Double(max(0, cur)), total: Double(max(1, total)))
                    .progressViewStyle(.linear)
            }

            Text(etaLine)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(backgroundColor)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(borderColor, lineWidth: 1)
        )
    }

    private var backgroundColor: Color {
        switch style {
        case .normal:
            return Color(.secondarySystemGroupedBackground)
        case .failed:
            return Color(.secondarySystemGroupedBackground)
        }
    }

    private var borderColor: Color {
        switch style {
        case .normal:
            return Color(.separator).opacity(0.15)
        case .failed:
            return Color.red.opacity(0.25)
        }
    }

    private var statusBadge: some View {
        Text(statusText)
            .font(.caption.bold())
            .foregroundStyle(statusTextColor)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(statusBackground)
            .clipShape(Capsule())
    }

    private var statusText: String {
        switch job.status {
        case .queued:
            return "Queued"
        case .uploading:
            return "Uploading"
        case .running:
            return "Processing"
        case .succeeded:
            return "Done"
        case .failed:
            return "Failed"
        }
    }

    private var statusBackground: Color {
        switch job.status {
        case .queued:
            return Color.orange.opacity(0.15)
        case .uploading:
            return Color.blue.opacity(0.15)
        case .running:
            return Color.purple.opacity(0.12)
        case .succeeded:
            return Color.green.opacity(0.15)
        case .failed:
            return Color.red.opacity(0.15)
        }
    }

    private var statusTextColor: Color {
        switch job.status {
        case .queued:
            return .orange
        case .uploading:
            return .blue
        case .running:
            return .purple
        case .succeeded:
            return .green
        case .failed:
            return .red
        }
    }

    private var etaLine: String {
        switch job.status {
        case .running, .queued:
            if let s = job.etaSeconds {
                return "About \(etaString(seconds: s)) left"
            }
            return "Estimating time…"
        case .uploading:
            return "Uploading…"
        case .failed:
            return "Tap for details"
        case .succeeded:
            return "Ready"
        }
    }

    private func etaString(seconds: Int) -> String {
        let s = max(0, seconds)
        if s < 60 { return "1 min" }
        if s < 3600 { return "\(max(1, Int(round(Double(s) / 60.0)))) min" }
        let h = Int(round(Double(s) / 3600.0))
        return h == 1 ? "1 hr" : "\(h) hr"
    }
}

private struct CompletedTile: View {
    let job: JobSummaryResponse

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))

                if let thumb = job.previewThumbnailUrl {
                    AsyncImage(url: thumb) { phase in
                        switch phase {
                        case .empty:
                            ProgressView()
                        case .success(let image):
                            image
                                .resizable()
                                .scaledToFill()
                        case .failure:
                            placeholder
                        @unknown default:
                            placeholder
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                } else {
                    placeholder
                }
            }
            .frame(height: 170)
            .clipped()

            HStack(alignment: .firstTextBaseline) {
                Text("Edit")
                    .font(.headline)
                    .foregroundStyle(.primary)
                Spacer()
                if let createdAt = job.createdAt {
                    Text(createdAt, style: .relative)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            if let count = job.variantsCount, count > 0 {
                Text(count == 1 ? "1 variation" : "\(count) variations")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(Color(.systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color(.separator).opacity(0.25), lineWidth: 1)
        )
    }

    private var placeholder: some View {
        Image(systemName: "play.rectangle")
            .font(.system(size: 28))
            .foregroundStyle(.secondary)
    }
}

struct EditsView_Previews: PreviewProvider {
    static var previews: some View {
        EditsView()
            .environmentObject(SessionStore())
            .environmentObject(AppRouter())
    }
}
