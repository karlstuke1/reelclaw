//
//  ContentView.swift
//  ReelClaw
//

import PhotosUI
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var session: SessionStore

    @State private var path: [String] = []

    private enum ReferenceMode: String, CaseIterable, Identifiable, Hashable {
        case link = "Link"
        case upload = "Upload"

        var id: String { rawValue }
    }

    @State private var referenceMode: ReferenceMode = .link
    @State private var referenceLink: String = ""
    @State private var referencePickerItem: PhotosPickerItem?
    @State private var referenceClip: PickedClip?

    @State private var burnOverlays: Bool = false
    @State private var compressBeforeUpload: Bool = true
    @State private var variations: Int = 3

    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var clipSlots: [PickedClip?] = []
    @State private var hasUserPickedClips: Bool = false
    @State private var didRestoreDraftClips: Bool = false

    @State private var isImportingClips: Bool = false
    @State private var importedClipCount: Int = 0
    @State private var importTotal: Int = 0

    @State private var isSubmitting: Bool = false
    @State private var errorMessage: String?

    private var readyClips: [PickedClip] {
        clipSlots.compactMap { $0 }
    }

    var body: some View {
        NavigationStack(path: $path) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    referenceSection
                    clipsSection
                    optionsSection
                    errorSection
                    submitSection
                }
                .padding()
            }
            .navigationTitle("Create")
            .navigationDestination(for: String.self) { jobId in
                JobProgressView(jobId: jobId)
            }
        }
        .task(id: pickerItems) {
            await loadPickerItems()
        }
        .task(id: referencePickerItem) {
            await loadReferenceItem()
        }
        .task {
            await restoreDraftClipsIfNeeded()
        }
    }

    private var referenceSection: some View {
        GroupBox("Reference") {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Reference", selection: $referenceMode) {
                    ForEach(ReferenceMode.allCases) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                if referenceMode == .link {
                    TextField("Paste reference URL (Instagram, YouTube, etc.)", text: $referenceLink, axis: .vertical)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                } else {
                    PhotosPicker(selection: $referencePickerItem, matching: .videos) {
                        Label(referenceClip == nil ? "Select reference video" : "Change reference video", systemImage: "film")
                    }

                    if let selectedReference = referenceClip {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(selectedReference.filename)
                                    .lineLimit(1)
                                Text(selectedReference.durationLabel)
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Clear", role: .destructive) {
                                referencePickerItem = nil
                                referenceClip = nil
                            }
                        }
                    } else {
                        Text("No reference selected yet.")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var clipsSection: some View {
        GroupBox("Your Clips") {
            VStack(alignment: .leading, spacing: 12) {
                PhotosPicker(
                    selection: $pickerItems,
                    maxSelectionCount: 30,
                    matching: .videos
                ) {
                    Label("Select clips", systemImage: "photo.on.rectangle.angled")
                }

                if isImportingClips {
                    HStack(spacing: 10) {
                        ProgressView()
                        Text("Importing \(min(importedClipCount, importTotal))/\(importTotal)…")
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                }

                if clipSlots.isEmpty {
                    Text("No clips selected yet.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(clipSlots.indices, id: \.self) { idx in
                        HStack {
                            if let clip = clipSlots[idx] {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(clip.filename)
                                        .lineLimit(1)
                                    Text(clip.durationLabel)
                                        .font(.footnote)
                                        .foregroundStyle(.secondary)
                                }
                            } else {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Importing clip \(idx + 1)…")
                                        .foregroundStyle(.secondary)
                                    Text("This can take a bit for large videos.")
                                        .font(.footnote)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                        }
                    }
                    Button("Clear clips", role: .destructive) {
                        ClipDraftStore.clear()
                        clipSlots = []
                        pickerItems = []
                        hasUserPickedClips = false
                        isImportingClips = false
                        importedClipCount = 0
                        importTotal = 0
                    }
                }
            }
        }
    }

    private var optionsSection: some View {
        GroupBox("Options") {
            VStack(alignment: .leading, spacing: 12) {
                Toggle("Burn captions (experimental)", isOn: $burnOverlays)
                Toggle("Compress videos before upload", isOn: $compressBeforeUpload)

                Stepper("Variations: \(variations)", value: $variations, in: 1...10, step: 1)
                    .disabled(isSubmitting)
                    .accessibilityIdentifier("variations_stepper")

                Text("Tip: keep variations low while testing to save time and cost.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var errorSection: some View {
        if let errorMessage {
            Text(errorMessage)
                .foregroundStyle(.red)
        }
    }

    private var submitSection: some View {
        Button {
            Task { await submit() }
        } label: {
            if isSubmitting {
                HStack {
                    Spacer()
                    ProgressView()
                    Spacer()
                }
            } else {
                Text("Generate \(variations) variation\(variations == 1 ? "" : "s")")
            }
        }
        .buttonStyle(.borderedProminent)
        .disabled(isSubmitting || isImportingClips || readyClips.isEmpty)
    }

	    private func loadPickerItems() async {
	        let items = pickerItems
	        if items.isEmpty {
            isImportingClips = false
            importedClipCount = 0
            importTotal = 0

            // If the user explicitly cleared via the picker UI (not just initial launch),
            // treat that as a "clear selection" event and wipe the persisted draft.
            if hasUserPickedClips {
                ClipDraftStore.clear()
                clipSlots = []
                hasUserPickedClips = false
            }
            return
        }

        hasUserPickedClips = true
        errorMessage = nil
        isImportingClips = true
        importedClipCount = 0
        importTotal = items.count
	        clipSlots = Array(repeating: nil, count: items.count)

	        var importSession: ClipDraftStore.ImportSession?
	        var didCommitImportSession = false
	        do {
	            let newImportSession = try ClipDraftStore.beginImportSession()
	            importSession = newImportSession
	            var records: [ClipDraftStore.DraftClip?] = Array(repeating: nil, count: items.count)

	            let maxConcurrent = 3
	            struct Imported: Sendable {
                let index: Int
                let clip: PickedClip
                let record: ClipDraftStore.DraftClip
            }
            try await withThrowingTaskGroup(of: Imported.self) { group in
                var nextIndex = 0

                func addNext() {
                    guard nextIndex < items.count else { return }
                    let idx = nextIndex
                    let item = items[idx]
                    nextIndex += 1
	                    group.addTask {
	                        try Task.checkCancellation()
	                        let url = try await item.loadVideoToTemporaryURL()
	                        try Task.checkCancellation()

	                        let displayName = url.lastPathComponent
	                        let clipId = UUID()
	                        let storedFilename = ClipDraftStore.makeStoredFilename(id: clipId, sourceExtension: url.pathExtension)
	                        let tmpDst = newImportSession.tmpClipsDir.appendingPathComponent(storedFilename)
	                        try ClipDraftStore.moveImportedFile(from: url, to: tmpDst)

	                        let clip = try await PickedClip.from(fileURL: tmpDst, id: clipId, filename: displayName)
	                        let record = ClipDraftStore.DraftClip(
	                            id: clipId,
	                            relativePath: "clips/\(storedFilename)",
                            displayName: displayName,
                            durationSeconds: clip.durationSeconds
                        )
                        return Imported(index: idx, clip: clip, record: record)
                    }
                }

                for _ in 0..<min(maxConcurrent, items.count) {
                    addNext()
                }

                while let imported = try await group.next() {
                    clipSlots[imported.index] = imported.clip
                    records[imported.index] = imported.record
                    importedClipCount += 1
                    addNext()
	                }
	            }

	            let committed = records.compactMap { $0 }
	            if committed.count != items.count {
	                throw VideoImportError.copyFailed
	            }

	            try ClipDraftStore.commitImportSession(newImportSession, clips: committed)
	            didCommitImportSession = true

	            // Swap the UI to the committed (stable) file URLs.
	            var restored: [PickedClip] = []
	            restored.reserveCapacity(committed.count)
            for rec in committed {
                guard let url = ClipDraftStore.fileURL(for: rec) else { continue }
                restored.append(
                    PickedClip(
                        id: rec.id,
                        url: url,
                        filename: rec.displayName,
                        durationSeconds: rec.durationSeconds
                    )
                )
            }
	            clipSlots = restored.map { Optional.some($0) }
	            isImportingClips = false
	        } catch is CancellationError {
	            if let importSession, !didCommitImportSession {
	                ClipDraftStore.discardImportSession(importSession)
	            }
	            // Selection changed; SwiftUI cancels the previous task.
	            isImportingClips = false
	        } catch {
	            if let importSession, !didCommitImportSession {
	                ClipDraftStore.discardImportSession(importSession)
	            }
	            isImportingClips = false
	            errorMessage = error.localizedDescription
	        }
	    }

    private func restoreDraftClipsIfNeeded() async {
        guard !didRestoreDraftClips else { return }
        didRestoreDraftClips = true

        guard !hasUserPickedClips else { return }
        guard pickerItems.isEmpty else { return }
        guard clipSlots.isEmpty else { return }

        guard let draft = ClipDraftStore.loadDraft(), !draft.clips.isEmpty else { return }
        let restored: [PickedClip] = draft.clips.compactMap { rec in
            guard let url = ClipDraftStore.fileURL(for: rec) else { return nil }
            return PickedClip(id: rec.id, url: url, filename: rec.displayName, durationSeconds: rec.durationSeconds)
        }
        if !restored.isEmpty {
            clipSlots = restored.map { Optional.some($0) }
        }
    }

    private func loadReferenceItem() async {
        guard let item = referencePickerItem else { return }
        do {
            let url = try await item.loadVideoToTemporaryURL()
            referenceClip = try await PickedClip.from(fileURL: url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func submit() async {
        errorMessage = nil

        guard !isImportingClips else {
            errorMessage = "Still importing clips. Please wait a moment."
            return
        }
        guard !readyClips.isEmpty else {
            errorMessage = "Please select at least 1 clip."
            return
        }
        if referenceMode == .upload, referenceClip == nil {
            errorMessage = "Please select a reference video (or switch to Link)."
            return
        }
        if (session.accessToken ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            errorMessage = "Sign in required. Please sign in again."
            return
        }

        isSubmitting = true
        defer { isSubmitting = false }

        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })

        do {
            let preparedClips = try await prepareUploads(readyClips.map(\.url))

            var referenceSpec: ReferenceSpec
            var referenceFileURL: URL?
            switch referenceMode {
            case .link:
                var raw = referenceLink.trimmingCharacters(in: .whitespacesAndNewlines)
                if !raw.lowercased().hasPrefix("http://") && !raw.lowercased().hasPrefix("https://") {
                    raw = "https://" + raw
                }
                guard let url = URL(string: raw), url.scheme != nil, url.host != nil else {
                    throw APIError.network("Please paste a valid http(s) URL (or switch to Upload).")
                }
                referenceSpec = ReferenceSpec(type: .url, url: url.absoluteString, filename: nil, contentType: nil, bytes: nil)
            case .upload:
                guard let clip = referenceClip else { throw APIError.network("Missing reference video.") }
                let refURL = try await VideoCompressor.compressIfNeeded(fileURL: clip.url, enabled: compressBeforeUpload)
                let refBytes = try fileSize(refURL)
                let refContentType = mimeType(for: refURL)
                referenceSpec = ReferenceSpec(
                    type: .upload,
                    url: nil,
                    filename: refURL.lastPathComponent,
                    contentType: refContentType,
                    bytes: refBytes
                )
                referenceFileURL = refURL
            }

            let job = try await api.createJob(
                CreateJobRequest(
                    reference: referenceSpec,
                    variations: variations,
                    burnOverlays: burnOverlays,
                    clips: preparedClips.map { ClipSpec(filename: $0.filename, contentType: $0.contentType, bytes: $0.bytes) }
                )
            )

            if let refTarget = job.referenceUpload {
                guard let referenceFileURL, let refCT = referenceSpec.contentType else {
                    throw APIError.network("Missing reference upload file.")
                }
                try await uploadFile(to: refTarget.uploadURL, fileURL: referenceFileURL, contentType: refCT)
            }

            if job.clipUploads.count != preparedClips.count {
                throw APIError.network("Upload mismatch. Please try again.")
            }
            for (idx, target) in job.clipUploads.enumerated() {
                let file = preparedClips[idx]
                try await uploadFile(to: target.uploadURL, fileURL: file.url, contentType: file.contentType)
            }

            try await api.startJob(jobId: job.jobId)
            path.append(job.jobId)
        } catch {
            errorMessage = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }

    private struct PreparedUpload: Hashable {
        let url: URL
        let filename: String
        let contentType: String
        let bytes: Int
    }

    private func prepareUploads(_ urls: [URL]) async throws -> [PreparedUpload] {
        var out: [PreparedUpload] = []
        out.reserveCapacity(urls.count)

        for url in urls {
            let processedURL = try await VideoCompressor.compressIfNeeded(fileURL: url, enabled: compressBeforeUpload)
            let bytes = try fileSize(processedURL)
            let contentType = mimeType(for: processedURL)
            out.append(
                PreparedUpload(
                    url: processedURL,
                    filename: processedURL.lastPathComponent,
                    contentType: contentType,
                    bytes: bytes
                )
            )
        }

        return out
    }

    private func fileSize(_ url: URL) throws -> Int {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        let size = values.fileSize ?? 0
        if size <= 0 {
            throw APIError.network("Couldn’t read file size for upload.")
        }
        return size
    }

    private func mimeType(for url: URL) -> String {
        UTType(filenameExtension: url.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
    }

    private func uploadFile(to uploadURL: URL, fileURL: URL, contentType: String) async throws {
        var headers: [String: String] = [
            "Content-Type": contentType
        ]

        // Upload URLs can be:
        // - presigned S3 URLs (no Authorization header)
        // - local API PUT URLs (require Bearer auth)
        if uploadURL.host == AppConfig.apiBaseURL.host,
           let token = session.accessToken?.trimmingCharacters(in: .whitespacesAndNewlines),
           !token.isEmpty
        {
            // API Gateway integrations can strip the standard `Authorization` header, so send the
            // app-specific token header too.
            headers["X-Reelclaw-Token"] = token
            headers["Authorization"] = "Bearer \(token)"
        }

        try await UploadManager.shared.upload(fileURL: fileURL, to: uploadURL, headers: headers)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(SessionStore())
    }
}
