import AVKit
import Photos
import SwiftUI

struct VariantDetailView: View {
    let variant: VariantsResponse.Variant

    @State private var player: AVPlayer?
    @State private var isDownloading: Bool = false
    @State private var localFileURL: URL?
    @State private var errorMessage: String?
    @State private var isSharePresented: Bool = false
    @State private var isAlertPresented: Bool = false
    @State private var alertTitle: String = ""
    @State private var alertMessage: String = ""

    var body: some View {
        VStack(spacing: 12) {
            VideoPlayer(player: player)
                .frame(height: 420)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .padding(.horizontal, 12)

            if let errorMessage {
                Text(errorMessage)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 12)
            }

            HStack(spacing: 10) {
                Button {
                    Task { await prepareLocalFile() }
                } label: {
                    if isDownloading {
                        ProgressView()
                            .frame(maxWidth: .infinity)
                    } else {
                        Label(localFileURL == nil ? "Download" : "Downloaded", systemImage: "arrow.down.circle")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(isDownloading)

                Button {
                    Task {
                        await prepareLocalFile()
                        if localFileURL != nil {
                            isSharePresented = true
                        }
                    }
                } label: {
                    Label("Share", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isDownloading)
            }
            .padding(.horizontal, 12)

            Button {
                Task {
                    await prepareLocalFile()
                    await saveToPhotos()
                }
            } label: {
                if isDownloading {
                    ProgressView()
                } else {
                    Label("Save to Photos", systemImage: "square.and.arrow.down")
                }
            }
            .buttonStyle(.bordered)
            .disabled(isDownloading)

            Spacer()
        }
        .navigationTitle(variant.title?.isEmpty == false ? variant.title! : "Variation")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $isSharePresented) {
            if let localFileURL {
                ShareSheet(items: [localFileURL])
            }
        }
        .alert(alertTitle, isPresented: $isAlertPresented) {
            Button("OK") {}
        } message: {
            Text(alertMessage)
        }
        .task {
            player = AVPlayer(url: variant.videoURL)
        }
        .onDisappear {
            player?.pause()
        }
    }

    private func prepareLocalFile() async {
        errorMessage = nil
        if localFileURL != nil { return }

        isDownloading = true
        defer { isDownloading = false }

        do {
            let (tempURL, _) = try await URLSession.shared.download(from: variant.videoURL)
            let ext = variant.videoURL.pathExtension.isEmpty ? "mp4" : variant.videoURL.pathExtension
            let dest = FileManager.default.temporaryDirectory
                .appendingPathComponent("reelclaw-\(variant.id)")
                .appendingPathExtension(ext)

            if FileManager.default.fileExists(atPath: dest.path) {
                try FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.moveItem(at: tempURL, to: dest)
            localFileURL = dest
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func saveToPhotos() async {
        guard let localFileURL else { return }

        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            alertTitle = "Photos Access Needed"
            alertMessage = "Enable Photos access to save videos."
            isAlertPresented = true
            return
        }

        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: localFileURL)
            }
            alertTitle = "Saved"
            alertMessage = "Video saved to Photos."
            isAlertPresented = true
        } catch {
            alertTitle = "Save Failed"
            alertMessage = error.localizedDescription
            isAlertPresented = true
        }
    }
}

struct VariantDetailView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationStack {
            VariantDetailView(
                variant: .init(
                    id: "v1",
                    title: "Variation 1",
                    score: 6.0,
                    videoURL: URL(string: "https://example.com/video.mp4")!,
                    thumbnailURL: nil
                )
            )
        }
    }
}
