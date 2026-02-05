import Foundation

struct APIClient {
    let baseURL: URL
    let accessTokenProvider: () -> String?

    init(baseURL: URL, accessTokenProvider: @escaping () -> String?) {
        self.baseURL = baseURL
        self.accessTokenProvider = accessTokenProvider
    }

    func signInWithApple(identityToken: String, authorizationCode: String) async throws -> AppleAuthResponse {
        struct Body: Encodable {
            let identityToken: String
            let authorizationCode: String
        }
        return try await request(
            method: "POST",
            path: "/v1/auth/apple",
            body: Body(identityToken: identityToken, authorizationCode: authorizationCode)
        )
    }

    func createJob(_ requestBody: CreateJobRequest) async throws -> CreateJobResponse {
        try await request(method: "POST", path: "/v1/jobs", body: requestBody)
    }

    func startJob(jobId: String) async throws {
        struct Empty: Encodable {}
        let _: EmptyResponse = try await request(method: "POST", path: "/v1/jobs/\(jobId)/start", body: Empty())
    }

    func getJob(jobId: String) async throws -> JobStatusResponse {
        try await request(method: "GET", path: "/v1/jobs/\(jobId)", body: Optional<EmptyBody>.none)
    }

    func listJobs() async throws -> ListJobsResponse {
        try await request(method: "GET", path: "/v1/jobs", body: Optional<EmptyBody>.none)
    }

    func getVariants(jobId: String) async throws -> VariantsResponse {
        try await request(method: "GET", path: "/v1/jobs/\(jobId)/variants", body: Optional<EmptyBody>.none)
    }

    func registerAPNSDevice(deviceToken: String, environment: String) async throws {
        struct Body: Encodable {
            let deviceToken: String
            let environment: String
        }
        let _: EmptyResponse = try await request(method: "POST", path: "/v1/devices/apns", body: Body(deviceToken: deviceToken, environment: environment))
    }

    private func request<Response: Decodable, Body: Encodable>(
        method: String,
        path: String,
        body: Body?
    ) async throws -> Response {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw APIError.invalidURL
        }

        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue("application/json", forHTTPHeaderField: "Accept")

        if let token = accessTokenProvider()?.trimmingCharacters(in: .whitespacesAndNewlines),
           !token.isEmpty
        {
            // API Gateway HTTP APIs can drop/strip headers containing "Authorization" before reaching ALB/ECS.
            // Keep `Authorization` for local/dev tooling, but always send an app-specific token header too.
            req.setValue(token, forHTTPHeaderField: "X-Reelclaw-Token")
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            req.httpBody = try encoder.encode(body)
        }

        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else {
            throw APIError.network("Invalid response")
        }
        guard (200...299).contains(http.statusCode) else {
            let message = (String(data: data, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            throw APIError.server(http.statusCode, message.isEmpty ? "Request failed" : message)
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode(Response.self, from: data)
        } catch {
            throw APIError.decoding(error.localizedDescription)
        }
    }
}

enum APIError: LocalizedError {
    case invalidURL
    case network(String)
    case server(Int, String)
    case decoding(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL."
        case .network(let message):
            return message
        case .server(let code, let message):
            return "Server error (\(code)): \(message)"
        case .decoding(let message):
            return "Failed to decode response: \(message)"
        }
    }
}

private struct EmptyResponse: Decodable {}
private struct EmptyBody: Encodable {}
