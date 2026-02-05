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
}

struct ClipSpec: Codable, Hashable {
    let filename: String
    let contentType: String
    let bytes: Int
}

struct CreateJobRequest: Encodable {
    let reference: ReferenceSpec
    let variations: Int
    let burnOverlays: Bool
    let clips: [ClipSpec]
}

struct UploadTarget: Decodable, Hashable {
    let s3Key: String?
    let uploadURL: URL
}

struct ClipUploadTarget: Decodable, Hashable {
    let clipId: String
    let s3Key: String?
    let uploadURL: URL
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
    let status: Status
    let stage: String?
    let message: String?
    let progressCurrent: Int?
    let progressTotal: Int?
    let errorCode: String?
    let errorDetail: String?
}

struct ListJobsResponse: Decodable {
    let jobs: [JobSummaryResponse]
}

struct JobSummaryResponse: Decodable, Identifiable, Hashable {
    let jobId: String
    let createdAt: Date?
    let status: JobStatusResponse.Status
    let stage: String?
    let message: String?

    var id: String { jobId }
}

struct VariantsResponse: Decodable {
    let jobId: String
    let variants: [Variant]

    struct Variant: Decodable, Identifiable, Hashable {
        let id: String
        let title: String?
        let score: Double?
        let videoURL: URL
        let thumbnailURL: URL?
    }
}

struct JobSummary: Codable, Identifiable {
    let id: String
    let createdAt: Date
    let referenceReelURL: String
    var status: JobStatusResponse.Status
}
