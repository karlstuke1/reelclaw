import SwiftUI

struct VariantsView: View {
    let jobId: String

    @EnvironmentObject private var session: SessionStore

    @State private var isLoading: Bool = true
    @State private var errorMessage: String?
    @State private var variants: [VariantsResponse.Variant] = []
    @StateObject private var saveAllManager = SaveAllManager()

    private let columns: [GridItem] = [
        GridItem(.adaptive(minimum: 170), spacing: 12, alignment: .top),
    ]

    var body: some View {
        Group {
            if isLoading {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Loading variations…")
                        .font(.headline)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let errorMessage {
                VStack(spacing: 12) {
                    Image(systemName: "xmark.octagon.fill")
                        .font(.system(size: 44))
                        .foregroundStyle(.red)
                    Text("Couldn’t load variations")
                        .font(.headline)
                    Text(errorMessage)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
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
                    LazyVGrid(columns: columns, spacing: 12) {
                        ForEach(variants) { variant in
                            NavigationLink {
                                VariantDetailView(variant: variant)
                            } label: {
                                VariantCardView(variant: variant)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(12)
                }
            }
        }
        .navigationTitle("Variations")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    saveAllManager.start(variants: variants)
                } label: {
                    Label("Save All", systemImage: "square.and.arrow.down.on.square")
                }
                .disabled(isLoading || errorMessage != nil || variants.isEmpty || saveAllManager.isRunning)
            }
        }
        .safeAreaInset(edge: .bottom) {
            SaveAllStatusBar(manager: saveAllManager)
        }
        .task {
            await load()
        }
    }

    private func load() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }

        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })
        do {
            let resp = try await api.getVariants(jobId: jobId)
            variants = resp.variants
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

private struct SaveAllStatusBar: View {
    @ObservedObject var manager: SaveAllManager

    var body: some View {
        if shouldShowBar {
            VStack(alignment: .leading, spacing: 8) {
                HStack(alignment: .firstTextBaseline) {
                    Text(titleText)
                        .font(.headline)
                    Spacer()
                    trailingButtons
                }

                if let permissionError = manager.permissionError, !permissionError.isEmpty {
                    Text(permissionError)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else {
                    ProgressView(value: Double(manager.processedCount), total: Double(max(1, manager.totalCount)))
                        .progressViewStyle(.linear)

                    Text(detailText)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .background(.ultraThinMaterial)
            .overlay(
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .stroke(Color(.separator).opacity(0.18), lineWidth: 1)
            )
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
            .padding(.horizontal, 12)
            .padding(.bottom, 8)
        }
    }

    private var shouldShowBar: Bool {
        manager.isRunning || manager.hasFinished || manager.permissionError != nil
    }

    private var titleText: String {
        if manager.permissionError != nil {
            return "Save All"
        }
        if manager.isRunning {
            return "Saving to Photos…"
        }
        if manager.failedCount > 0 {
            return "Saved with Issues"
        }
        return "Saved"
    }

    private var detailText: String {
        if manager.isRunning {
            return "\(manager.processedCount)/\(manager.totalCount)"
        }
        if manager.failedCount > 0 {
            return "\(manager.doneCount) saved • \(manager.failedCount) failed"
        }
        return "\(manager.doneCount) saved"
    }

    @ViewBuilder
    private var trailingButtons: some View {
        if manager.isRunning {
            Button("Cancel") {
                manager.cancel()
            }
            .buttonStyle(.bordered)
        } else {
            HStack(spacing: 10) {
                if manager.failedCount > 0 {
                    Button("Retry") {
                        manager.retryFailed()
                    }
                    .buttonStyle(.borderedProminent)
                }
                Button("Dismiss") {
                    manager.dismiss()
                }
                .buttonStyle(.bordered)
            }
        }
    }
}

private struct VariantCardView: View {
    let variant: VariantsResponse.Variant

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(Color(.secondarySystemGroupedBackground))

                if let thumb = variant.thumbnailUrl {
                    AsyncImage(url: thumb) { phase in
                        switch phase {
                        case .empty:
                            ProgressView()
                        case .success(let image):
                            image
                                .resizable()
                                .scaledToFill()
                        case .failure:
                            Image(systemName: "video")
                                .font(.system(size: 28))
                                .foregroundStyle(.secondary)
                        @unknown default:
                            EmptyView()
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                } else {
                    Image(systemName: "video")
                        .font(.system(size: 28))
                        .foregroundStyle(.secondary)
                }
            }
            .frame(height: 180)
            .clipped()

            VStack(alignment: .leading, spacing: 2) {
                Text(variant.title?.isEmpty == false ? variant.title! : "Variation")
                    .font(.headline)
                    .foregroundStyle(.primary)

                if let score = variant.score {
                    Text(String(format: "Score: %.1f", score))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
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
}

struct VariantsView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            VariantsView(jobId: "job_123")
                .environmentObject(SessionStore())
        }
    }
}
