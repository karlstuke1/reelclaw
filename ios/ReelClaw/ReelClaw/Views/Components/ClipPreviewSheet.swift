import AVKit
import SwiftUI

struct ClipPreviewSheet: View {
    let clip: PickedClip

    @Environment(\.dismiss) private var dismiss
    @State private var player: AVPlayer?

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                VideoPlayer(player: player)
                    .frame(height: 420)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .padding(.horizontal, 12)

                VStack(alignment: .leading, spacing: 4) {
                    Text(clip.filename)
                        .font(.headline)
                        .lineLimit(2)
                    Text(clip.durationLabel)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)

                Spacer()
            }
            .navigationTitle("Preview")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        player?.pause()
                        player = nil
                        dismiss()
                    }
                }
            }
            .task(id: clip.url) {
                player = AVPlayer(url: clip.url)
            }
            .onDisappear {
                player?.pause()
                player = nil
            }
        }
    }
}
