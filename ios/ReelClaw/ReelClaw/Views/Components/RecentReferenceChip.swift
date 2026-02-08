import SwiftUI

struct RecentReferenceChip: View {
    let reference: RecentReference

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(reference.host)
                .font(.footnote.weight(.semibold))
                .lineLimit(1)
            Text(reference.lastUsedAt, style: .relative)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color(.separator).opacity(0.18), lineWidth: 1)
        )
    }
}

