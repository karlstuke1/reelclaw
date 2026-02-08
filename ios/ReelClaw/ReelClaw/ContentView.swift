//
//  ContentView.swift
//  ReelClaw
//

import PhotosUI
import SwiftUI
import UniformTypeIdentifiers
import UIKit

struct ContentView: View {
    @EnvironmentObject private var session: SessionStore
    @EnvironmentObject private var recentReferences: RecentReferenceStore

    @State private var path: [String] = []

    private enum ReferenceMode: String, CaseIterable, Identifiable, Hashable {
        case link
        case upload

        var id: String { rawValue }

        var title: String {
            switch self {
            case .link:
                return "Link"
            case .upload:
                return "Upload"
            }
        }
    }

    @AppStorage(UserDefaultsKeys.createReferenceMode) private var referenceModeRaw: String = ReferenceMode.link.rawValue
    @AppStorage(UserDefaultsKeys.createReferenceLink) private var referenceLink: String = ""
    @State private var referencePickerItem: PhotosPickerItem?
    @State private var referenceClip: PickedClip?

    @AppStorage(UserDefaultsKeys.createBurnOverlays) private var burnOverlays: Bool = false
    @AppStorage(UserDefaultsKeys.createCompressBeforeUpload) private var compressBeforeUpload: Bool = true
    @AppStorage(UserDefaultsKeys.createVariations) private var variations: Int = 3
    @AppStorage(UserDefaultsKeys.createDirector) private var directorRaw: String = VariantDirector.code.rawValue
    @AppStorage(UserDefaultsKeys.createReferenceReusePct) private var referenceReusePct: Double = 0
    @State private var showAdvanced: Bool = false
    @State private var submitStatus: String?

    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var clipSlots: [PickedClip?] = []
    @State private var hasUserPickedClips: Bool = false
    @State private var didRestoreDraftClips: Bool = false

    @State private var isImportingClips: Bool = false
    @State private var importedClipCount: Int = 0
    @State private var importTotal: Int = 0

    @State private var isSubmitting: Bool = false
    @State private var errorMessage: String?

    @State private var previewClip: PickedClip?
    @State private var showRecentReferences: Bool = false

    private var readyClips: [PickedClip] {
        clipSlots.compactMap { $0 }
    }

    private var referenceMode: ReferenceMode {
        get { ReferenceMode(rawValue: referenceModeRaw) ?? .link }
        set { referenceModeRaw = newValue.rawValue }
    }

    private var referenceModeBinding: Binding<ReferenceMode> {
        Binding(
            get: { referenceMode },
            set: { referenceModeRaw = $0.rawValue }
        )
    }

    private var director: VariantDirector {
        get { VariantDirector(rawValue: directorRaw) ?? .code }
        set { directorRaw = newValue.rawValue }
    }

    private var directorBinding: Binding<VariantDirector> {
        Binding(
            get: { director },
            set: { directorRaw = $0.rawValue }
        )
    }

    var body: some View {
        NavigationStack(path: $path) {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    referenceSection
                    clipsSection
                    optionsSection
                    errorSection
                }
                .padding()
            }
            .safeAreaInset(edge: .bottom) {
                submitBar
            }
            .navigationTitle("Create")
            .navigationDestination(for: String.self) { jobId in
                JobProgressView(jobId: jobId)
            }
            .sheet(isPresented: $showAdvanced) {
                advancedSheet
            }
            .sheet(isPresented: $showRecentReferences) {
                RecentReferencesSheet { url in
                    referenceLink = url
                }
            }
            .sheet(item: $previewClip) { clip in
                ClipPreviewSheet(clip: clip)
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
                Picker("Reference", selection: referenceModeBinding) {
                    ForEach(ReferenceMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                if referenceMode == .link {
                    TextField("Paste reference URL (Instagram, YouTube, etc.)", text: $referenceLink, axis: .vertical)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)

                    if !recentReferences.items.isEmpty {
                        recentReferencesRow
                    }
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

    private var recentReferencesRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Recent")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Button("See all") {
                    showRecentReferences = true
                }
                .font(.footnote)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(recentReferences.items.prefix(12)) { ref in
                        Button {
                            referenceLink = ref.url
                            recentReferences.record(url: ref.url)
                        } label: {
                            RecentReferenceChip(reference: ref)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button {
                                UIPasteboard.general.string = ref.url
                            } label: {
                                Label("Copy URL", systemImage: "doc.on.doc")
                            }

                            Button(role: .destructive) {
                                recentReferences.remove(id: ref.id)
                            } label: {
                                Label("Remove", systemImage: "trash")
                            }
                        }
                    }
                }
                .padding(.vertical, 2)
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
                        Text("Preparing clips \(min(importedClipCount, importTotal))/\(importTotal)…")
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                }

                if clipSlots.isEmpty {
                    Text("No clips selected yet.")
                        .foregroundStyle(.secondary)
                } else {
                    Text("\(readyClips.count) clip\(readyClips.count == 1 ? "" : "s") selected")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(.secondary)

                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 92), spacing: 10, alignment: .top)],
                        spacing: 10
                    ) {
                        ForEach(clipSlots.indices, id: \.self) { idx in
                            if let clip = clipSlots[idx] {
                                Button {
                                    previewClip = clip
                                } label: {
                                    SelectedClipTileView(clip: clip, index: idx)
                                }
                                .buttonStyle(.plain)
                            } else {
                                selectedClipPlaceholder(index: idx)
                            }
                        }
                    }

                    Button("Clear clips", role: .destructive) {
                        ClipDraftStore.clear()
                        clipSlots = []
                        pickerItems = []
                        hasUserPickedClips = false
                        previewClip = nil
                        isImportingClips = false
                        importedClipCount = 0
                        importTotal = 0
                    }
                }
            }
        }
    }

    private func selectedClipPlaceholder(index: Int) -> some View {
        VStack(spacing: 8) {
            ProgressView()
            Text("Clip \(index + 1)")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .aspectRatio(1, contentMode: .fit)
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(Color(.separator).opacity(0.18), lineWidth: 1)
        )
    }

    private var optionsSection: some View {
        GroupBox("Advanced") {
            VStack(alignment: .leading, spacing: 10) {
                Button {
                    showAdvanced = true
                } label: {
                    HStack(spacing: 10) {
                        Label("Edit settings", systemImage: "slider.horizontal.3")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)
                .disabled(isSubmitting)

                Text(advancedSummaryText)
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
                .textSelection(.enabled)
        }
    }

    private var advancedSummaryText: String {
        let edits = "\(variations) edit" + (variations == 1 ? "" : "s")
        let compress = compressBeforeUpload ? "Compress on" : "Compress off"
        let captions = burnOverlays ? "Burn captions on" : "Burn captions off"
        var parts = [edits, director.title, compress, captions]
        if referenceReusePct >= 1 {
            parts.append("Ref keep \(Int(referenceReusePct))%")
        }
        return parts.joined(separator: " • ")
    }

    private var createEstimateText: String {
        let multiplier: Double
        switch director {
        case .code:
            multiplier = 1.0
        case .auto:
            multiplier = 1.3
        case .gemini:
            multiplier = 1.6
        }

        let expectedSeconds = (120.0 + 120.0 * Double(variations)) * multiplier
        let lo = max(1, Int(round((expectedSeconds * 0.8) / 60.0)))
        let hi = max(lo, Int(round((expectedSeconds * 1.2) / 60.0)))
        return "Est. \(lo)–\(hi) min"
    }

    private var submitBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                Task { await submit() }
            } label: {
                if isSubmitting {
                    HStack(spacing: 10) {
                        ProgressView()
                        Text(submitStatus?.isEmpty == false ? submitStatus! : "Working…")
                            .font(.headline)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 2)
                } else {
                    Text("Generate \(variations) edit\(variations == 1 ? "" : "s")")
                        .frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isSubmitting || isImportingClips || readyClips.isEmpty)

            if !isSubmitting {
                Text(createEstimateText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial)
    }

    private var advancedSheet: some View {
        NavigationStack {
            Form {
                Section("Edits") {
                    Stepper("Edits: \(variations)", value: $variations, in: 1...10, step: 1)
                        .disabled(isSubmitting)
                        .accessibilityIdentifier("variations_stepper")
                }

                Section("Director") {
                    Picker("Director", selection: directorBinding) {
                        ForEach(VariantDirector.allCases) { d in
                            Text(d.title).tag(d)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(isSubmitting)

                    Text(director.subtitle)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Reference") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Keep reference clips")
                            Spacer()
                            Text("\(Int(referenceReusePct))%")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }

                        Slider(value: $referenceReusePct, in: 0...100, step: 5)
                            .disabled(isSubmitting)

                        Text("Reuse some segments directly from the reference reel in the output. 0% means everything is cut from your clips.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Media") {
                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("Compress videos before upload (recommended)", isOn: $compressBeforeUpload)
                            .disabled(isSubmitting)
                        Text("Faster uploads and fewer failures. Slightly lower quality.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Toggle("Burn overlay captions (beta)", isOn: $burnOverlays)
                            .disabled(isSubmitting)
                        Text("If the editor generates overlay captions, bake them into the exported videos.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Advanced")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { showAdvanced = false }
                }
            }
        }
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
        submitStatus = nil

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
        defer {
            isSubmitting = false
            submitStatus = nil
        }

        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })

        do {
            submitStatus = "Preparing clips…"
            let preparedClips = try await prepareUploads(readyClips.map(\.url))

            var referenceSpec: ReferenceSpec
            var referenceFileURL: URL?
            var referenceURLToRecord: String?
            switch referenceMode {
            case .link:
                var raw = referenceLink.trimmingCharacters(in: .whitespacesAndNewlines)
                if !raw.lowercased().hasPrefix("http://") && !raw.lowercased().hasPrefix("https://") {
                    raw = "https://" + raw
                }
                guard let url = URL(string: raw), url.scheme != nil, url.host != nil else {
                    throw APIError.network("Please paste a valid http(s) URL (or switch to Upload).")
                }
                referenceSpec = ReferenceSpec(type: .url, url: url.absoluteString, filename: nil, contentType: nil, bytes: nil, sha256: nil)
                referenceURLToRecord = url.absoluteString
            case .upload:
                guard let clip = referenceClip else { throw APIError.network("Missing reference video.") }
                submitStatus = "Preparing reference…"
                let refURL = try await VideoCompressor.compressIfNeeded(fileURL: clip.url, enabled: compressBeforeUpload)
                let refBytes = try fileSize(refURL)
                let refContentType = mimeType(for: refURL)
                submitStatus = "Hashing reference…"
                let refSha = try await sha256Hex(of: refURL)
                referenceSpec = ReferenceSpec(
                    type: .upload,
                    url: nil,
                    filename: refURL.lastPathComponent,
                    contentType: refContentType,
                    bytes: refBytes,
                    sha256: refSha
                )
                referenceFileURL = refURL
            }

            submitStatus = "Creating job…"
            let job = try await api.createJob(
                CreateJobRequest(
                    reference: referenceSpec,
                    variations: variations,
                    burnOverlays: burnOverlays,
                    referenceReusePct: referenceReusePct >= 1 ? referenceReusePct : nil,
                    director: director,
                    clips: preparedClips.map { ClipSpec(filename: $0.filename, contentType: $0.contentType, bytes: $0.bytes, sha256: $0.sha256) }
                )
            )
            if let referenceURLToRecord {
                recentReferences.record(url: referenceURLToRecord)
            }

            if let refTarget = job.referenceUpload, refTarget.alreadyUploaded != true {
                guard let referenceFileURL, let refCT = referenceSpec.contentType else {
                    throw APIError.network("Missing reference upload file.")
                }
                submitStatus = "Uploading reference…"
                try await uploadFile(to: refTarget.uploadUrl, fileURL: referenceFileURL, contentType: refCT)
            }

            if job.clipUploads.count != preparedClips.count {
                throw APIError.network("Upload mismatch. Please try again.")
            }
            for (idx, target) in job.clipUploads.enumerated() {
                if target.alreadyUploaded == true {
                    continue
                }
                let file = preparedClips[idx]
                submitStatus = "Uploading clip \(idx + 1)/\(preparedClips.count)…"
                try await uploadFile(to: target.uploadUrl, fileURL: file.url, contentType: file.contentType)
            }

            submitStatus = "Starting job…"
            try await api.startJob(jobId: job.jobId)
            path.append(job.jobId)
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

    private struct PreparedUpload: Hashable {
        let url: URL
        let filename: String
        let contentType: String
        let bytes: Int
        let sha256: String
    }

    private func prepareUploads(_ urls: [URL]) async throws -> [PreparedUpload] {
        var out: [PreparedUpload] = []
        out.reserveCapacity(urls.count)

        for (idx, url) in urls.enumerated() {
            await MainActor.run {
                submitStatus = "Preparing clip \(idx + 1)/\(urls.count)…"
            }
            let processedURL = try await VideoCompressor.compressIfNeeded(fileURL: url, enabled: compressBeforeUpload)
            let bytes = try fileSize(processedURL)
            let contentType = mimeType(for: processedURL)
            await MainActor.run {
                submitStatus = "Hashing clip \(idx + 1)/\(urls.count)…"
            }
            let sha = try await sha256Hex(of: processedURL)
            out.append(
                PreparedUpload(
                    url: processedURL,
                    filename: processedURL.lastPathComponent,
                    contentType: contentType,
                    bytes: bytes,
                    sha256: sha
                )
            )
        }

        return out
    }

    private func sha256Hex(of url: URL) async throws -> String {
        try await Task.detached(priority: .utility) {
            try FileHasher.sha256Hex(of: url)
        }.value
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
        var targetURL = uploadURL

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
            targetURL = appendQueryItem(url: uploadURL, name: "token", value: token)
        }

        try await UploadManager.shared.upload(fileURL: fileURL, to: targetURL, headers: headers)
    }

    private func appendQueryItem(url: URL, name: String, value: String) -> URL {
        guard var comps = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return url }
        var items = comps.queryItems ?? []
        items.append(URLQueryItem(name: name, value: value))
        comps.queryItems = items
        return comps.url ?? url
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(SessionStore())
            .environmentObject(RecentReferenceStore())
    }
}
