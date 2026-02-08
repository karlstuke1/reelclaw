import SwiftUI

struct SelectedClipTileView: View {
    let clip: PickedClip
    let index: Int?

    var body: some View {
        ZStack {
            VideoThumbnailView(url: clip.url, cornerRadius: 12)

            VStack {
                HStack {
                    if let index {
                        Text("\(index + 1)")
                            .font(.caption2.bold())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.55))
                            .clipShape(Capsule())
                    }
                    Spacer()
                }
                Spacer()
                HStack {
                    Spacer()
                    Text(clip.durationLabel)
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.black.opacity(0.65))
                        .clipShape(Capsule())
                }
            }
            .padding(6)
        }
        .aspectRatio(1, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

