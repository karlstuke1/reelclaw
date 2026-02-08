import Foundation

enum ReferenceType: String, Codable {
    case url
    case upload
}

struct ReferenceSpec: Codable, Hashable {
    let type: ReferenceType
    var url: String?
    var filename: String?
    var contentType: String?
    var bytes: Int?
    var sha256: String?
}

struct ClipSpec: Codable, Hashable {
    let filename: String
    let contentType: String
    let bytes: Int
    let sha256: String?
}

enum VariantDirector: String, Codable, CaseIterable, Identifiable, Hashable {
    case auto
    case code
    case gemini

    var id: String { rawValue }

    var title: String {
        switch self {
        case .auto:
            return "Auto"
        case .code:
            return "Code"
        case .gemini:
            return "Gemini"
        }
    }

    var subtitle: String {
        switch self {
        case .auto:
            return "Try Gemini, fall back to code"
        case .code:
            return "Deterministic heuristic director"
        case .gemini:
            return "LLM chooses shots (higher cost)"
        }
    }
}

struct CreateJobRequest: Encodable {
    let reference: ReferenceSpec
    let variations: Int
    let burnOverlays: Bool
    // Percent (0..100) of segments to keep directly from the reference reel.
    // When nil/0, the backend uses only the user's clips.
    let referenceReusePct: Double?
    let director: VariantDirector?
    let clips: [ClipSpec]
}

struct UploadTarget: Decodable, Hashable {
    let s3Key: String?
    let uploadUrl: URL
    let alreadyUploaded: Bool?
}

struct ClipUploadTarget: Decodable, Hashable {
    let clipId: String
    let s3Key: String?
    let uploadUrl: URL
    let alreadyUploaded: Bool?
}

struct CreateJobResponse: Decodable {
    let jobId: String
    let referenceUpload: UploadTarget?
    let clipUploads: [ClipUploadTarget]
}

struct AppleAuthResponse: Decodable {
    let accessToken: String
}

struct JobStatusResponse: Decodable {
    enum Status: String, Codable, Hashable {
        case queued
        case uploading
        case running
        case succeeded
        case failed
    }

    let jobId: String
    let createdAt: Date?
    let updatedAt: Date?
    let queuedAt: Date?
    let startedAt: Date?
    let finishedAt: Date?
    let status: Status
    let stage: String?
    let message: String?
    let progressCurrent: Int?
    let progressTotal: Int?
    let etaSeconds: Int?
    let etaFinishAt: Date?
    let errorCode: String?
    let errorDetail: String?
}

struct ListJobsResponse: Decodable {
    let jobs: [JobSummaryResponse]
}

struct JobSummaryResponse: Decodable, Identifiable, Hashable {
    let jobId: String
    let createdAt: Date?
    let updatedAt: Date?
    let status: JobStatusResponse.Status
    let stage: String?
    let message: String?
    let progressCurrent: Int?
    let progressTotal: Int?
    let etaSeconds: Int?
    let variantsCount: Int?
    let previewThumbnailUrl: URL?

    var id: String { jobId }
}

struct VariantsResponse: Decodable {
    let jobId: String
    let variants: [Variant]

    struct Variant: Decodable, Identifiable, Hashable {
        let id: String
        let title: String?
        let score: Double?
        let videoUrl: URL
        let thumbnailUrl: URL?
    }
}

struct JobSummary: Codable, Identifiable {
    let id: String
    let createdAt: Date
    let referenceReelURL: String
    var status: JobStatusResponse.Status
}
