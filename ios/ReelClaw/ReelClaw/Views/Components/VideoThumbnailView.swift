import SwiftUI
import UIKit

struct VideoThumbnailView: View {
    let url: URL
    var cornerRadius: CGFloat = 12

    @State private var image: UIImage?
    @State private var isLoading: Bool = false

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                .fill(Color(.secondarySystemGroupedBackground))

            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .transition(.opacity)
            } else if isLoading {
                ProgressView()
            } else {
                Image(systemName: "video")
                    .font(.system(size: 22))
                    .foregroundStyle(.secondary)
            }
        }
        .clipped()
        .task(id: url) {
            await load()
        }
    }

    private func load() async {
        if image != nil { return }
        isLoading = true
        defer { isLoading = false }

        do {
            let img = try await VideoThumbnailService.shared.thumbnail(for: url)
            await MainActor.run {
                self.image = img
            }
        } catch {
            // Keep placeholder.
        }
    }
}
