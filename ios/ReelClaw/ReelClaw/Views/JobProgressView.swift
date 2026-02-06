import SwiftUI

struct JobProgressView: View {
    let jobId: String
    @EnvironmentObject private var session: SessionStore

    @State private var status: JobStatusResponse?
    @State private var errorMessage: String?
    @State private var errorDetail: String?
    @State private var showVariants: Bool = false
    @State private var showErrorDetails: Bool = false

    var body: some View {
        VStack(spacing: 14) {
            if let errorMessage {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 44))
                    .foregroundStyle(.orange)
                Text("Something went wrong")
                    .font(.headline)
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .textSelection(.enabled)
                if let errorDetail, !errorDetail.isEmpty {
                    Button("Show details") { showErrorDetails = true }
                        .buttonStyle(.bordered)
                        .padding(.top, 4)
                }
                if status?.status == .failed {
                    Button("Retry") {
                        Task { await retry() }
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 8)

                    Button("Refresh status") {
                        Task { await poll() }
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button("Try again") {
                        Task { await poll() }
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 8)
                }
            } else {
                ProgressView(value: progressValue, total: progressTotal)
                    .progressViewStyle(.linear)
                    .frame(maxWidth: 360)
                    .padding(.bottom, 4)

                Text(titleText)
                    .font(.headline)
                if let message = status?.message, !message.isEmpty {
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 480)
                }

                if let etaLine {
                    Text(etaLine)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }

                Text(jobId)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.top, 8)

                if status?.status == .succeeded {
                    Button(variantsButtonTitle) {
                        showVariants = true
                    }
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 10)
                }
            }
        }
        .padding(24)
        .navigationTitle("Generating")
        .navigationDestination(isPresented: $showVariants) {
            VariantsView(jobId: jobId)
        }
        .sheet(isPresented: $showErrorDetails) {
            NavigationStack {
                ScrollView {
                    Text(errorDetail ?? "")
                        .font(.system(.footnote, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .navigationTitle("Details")
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Done") { showErrorDetails = false }
                    }
                }
            }
        }
        .task {
            await poll()
        }
    }

    private var titleText: String {
        if let stage = status?.stage, !stage.isEmpty {
            return stage
        }
        if let s = status?.status {
            switch s {
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
        return "Starting"
    }

    private var progressTotal: Double {
        guard let total = status?.progressTotal, total > 0 else { return 1 }
        return Double(total)
    }

    private var progressValue: Double {
        guard let cur = status?.progressCurrent, cur >= 0 else { return 0 }
        return Double(cur)
    }

    private var variantsButtonTitle: String {
        let total = status?.progressTotal ?? 0
        let n = max(1, total)
        return n == 1 ? "View variation" : "View \(n) variations"
    }

    private var etaLine: String? {
        guard let s = status?.status else { return nil }
        if s != .queued && s != .running {
            return nil
        }
        if let seconds = status?.etaSeconds {
            return "About \(etaString(seconds: seconds)) remaining"
        }
        return "Estimating time…"
    }

    private func etaString(seconds: Int) -> String {
        let s = max(0, seconds)
        if s < 60 { return "1 min" }
        if s < 3600 { return "\(max(1, Int(round(Double(s) / 60.0)))) min" }
        let h = Int(round(Double(s) / 3600.0))
        return h == 1 ? "1 hr" : "\(h) hr"
    }

    private func poll() async {
        errorMessage = nil
        errorDetail = nil
        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })

        while !Task.isCancelled {
            do {
                let next = try await api.getJob(jobId: jobId)
                status = next

                if next.status == .succeeded {
                    showVariants = true
                    return
                }
                if next.status == .failed {
                    errorMessage = next.message ?? "Job failed."
                    errorDetail = next.errorDetail
                    return
                }
            } catch {
                if let apiError = error as? APIError,
                   case let .server(code, _) = apiError,
                   code == 401
                {
                    session.signOut(reason: "Session expired. Please sign in again.")
                    return
                }
                errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
                return
            }

            try? await Task.sleep(nanoseconds: 2_000_000_000)
        }
    }

    private func retry() async {
        errorMessage = nil
        errorDetail = nil
        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })
        do {
            try await api.startJob(jobId: jobId)
        } catch {
            if let apiError = error as? APIError,
               case let .server(code, _) = apiError,
               code == 401
            {
                session.signOut(reason: "Session expired. Please sign in again.")
                return
            }
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            return
        }
        await poll()
    }
}

struct JobProgressView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            JobProgressView(jobId: "job_123")
        }
        .environmentObject(SessionStore())
    }
}
