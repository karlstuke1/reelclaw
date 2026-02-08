import SwiftUI

struct JobsView: View {
    @EnvironmentObject private var session: SessionStore
    @EnvironmentObject private var router: AppRouter

    @State private var path: [String] = []
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?
    @State private var jobs: [JobSummaryResponse] = []

    var body: some View {
        NavigationStack(path: $path) {
            Group {
                if isLoading && jobs.isEmpty {
                    VStack(spacing: 12) {
                        ProgressView()
                        Text("Loading jobs…")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let errorMessage, jobs.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "xmark.octagon.fill")
                            .font(.system(size: 44))
                            .foregroundStyle(.red)
                        Text("Couldn’t load jobs")
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
                    List {
                        if let errorMessage {
                            Text(errorMessage)
                                .font(.footnote)
                                .foregroundStyle(.red)
                                .textSelection(.enabled)
                        }

                        ForEach(jobs) { job in
                            Button {
                                path.append(job.jobId)
                            } label: {
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text(job.status.rawValue)
                                            .font(.headline)
                                        Spacer()
                                        if let createdAt = job.createdAt {
                                            Text(createdAt, style: .relative)
                                                .font(.footnote)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    if let stage = job.stage, !stage.isEmpty {
                                        Text(stage)
                                            .font(.footnote)
                                            .foregroundStyle(.secondary)
                                    }
                                    if let message = job.message, !message.isEmpty {
                                        Text(message)
                                            .font(.footnote)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(2)
                                    }
                                    Text(job.jobId)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                                .padding(.vertical, 4)
                            }
                        }
                    }
                    .listStyle(.insetGrouped)
                    .refreshable {
                        await load()
                    }
                }
            }
            .navigationTitle("Jobs")
            .navigationDestination(for: String.self) { jobId in
                JobProgressView(jobId: jobId)
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await load() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(isLoading)
                }
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

    private func openPendingJobIfAny() {
        guard let jobId = router.consumePendingJobIdToOpen() else { return }
        if !jobId.isEmpty {
            path = [jobId]
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
}

struct JobsView_Previews: PreviewProvider {
    static var previews: some View {
        JobsView()
            .environmentObject(SessionStore())
            .environmentObject(AppRouter())
    }
}
