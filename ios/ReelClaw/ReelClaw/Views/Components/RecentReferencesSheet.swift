import SwiftUI
import UIKit

struct RecentReferencesSheet: View {
    let onSelect: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var store: RecentReferenceStore

    var body: some View {
        NavigationStack {
            Group {
                if store.items.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "clock")
                            .font(.system(size: 34))
                            .foregroundStyle(.secondary)
                        Text("No recent references yet")
                            .font(.headline)
                        Text("When you generate an edit from a link reference, it will show up here for quick reuse.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 320)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(24)
                } else {
                    List {
                        ForEach(store.items) { ref in
                            Button {
                                store.record(url: ref.url)
                                onSelect(ref.url)
                                dismiss()
                            } label: {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(ref.host)
                                        .font(.headline)
                                        .lineLimit(1)
                                    Text(ref.url)
                                        .font(.footnote)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    Text(ref.lastUsedAt, style: .relative)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .contextMenu {
                                Button {
                                    UIPasteboard.general.string = ref.url
                                } label: {
                                    Label("Copy URL", systemImage: "doc.on.doc")
                                }

                                Button(role: .destructive) {
                                    store.remove(id: ref.id)
                                } label: {
                                    Label("Remove", systemImage: "trash")
                                }
                            }
                            .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                Button(role: .destructive) {
                                    store.remove(id: ref.id)
                                } label: {
                                    Label("Remove", systemImage: "trash")
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Recent References")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    if !store.items.isEmpty {
                        Button(role: .destructive) {
                            store.clear()
                        } label: {
                            Text("Clear")
                        }
                    }
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

