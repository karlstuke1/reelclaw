import SwiftUI

struct VariantsView: View {
    let jobId: String

    @EnvironmentObject private var session: SessionStore

    @State private var isLoading: Bool = true
    @State private var errorMessage: String?
    @State private var variants: [VariantsResponse.Variant] = []

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
                            NavigationLink(value: variant) {
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
        .navigationDestination(for: VariantsResponse.Variant.self) { variant in
            VariantDetailView(variant: variant)
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
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
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

                if let thumb = variant.thumbnailURL {
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
