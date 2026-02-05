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

    func healthz() async throws -> HealthzResponse {
        try await request(method: "GET", path: "/healthz", body: Optional<EmptyBody>.none)
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
            var message = (String(data: data, encoding: .utf8) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if let parsed = parseErrorDetail(data: data) {
                message = parsed
            }
            throw APIError.server(http.statusCode, message.isEmpty ? "Request failed" : message)
        }

        if Response.self == EmptyResponse.self {
            return EmptyResponse() as! Response
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .iso8601
        do {
            let trimmed = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if data.isEmpty || trimmed.isEmpty {
                throw APIError.decoding("Empty response body for \(method) \(url.absoluteString) (HTTP \(http.statusCode)).")
            }
            return try decoder.decode(Response.self, from: data)
        } catch {
            let bodyPreview = bodyPreviewString(data: data, maxChars: 800)
            let detail = decodingErrorDetail(error)
            let contentType = (http.value(forHTTPHeaderField: "Content-Type") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let hint: String
            if let parsed = parseErrorDetail(data: data), !parsed.isEmpty {
                hint = " Parsed error detail: \(parsed)"
            } else {
                hint = ""
            }
            throw APIError.decoding("HTTP \(http.statusCode) \(method) \(url.absoluteString): \(detail). Content-Type: \(contentType.isEmpty ? "<missing>" : contentType). Body: \(bodyPreview).\(hint)")
        }
    }

    private func decodingErrorDetail(_ error: Error) -> String {
        if let apiError = error as? APIError {
            return apiError.errorDescription ?? apiError.localizedDescription
        }
        guard let e = error as? DecodingError else {
            return error.localizedDescription
        }
        func pathString(_ path: [CodingKey]) -> String {
            if path.isEmpty { return "<root>" }
            return path.map { $0.stringValue }.joined(separator: ".")
        }
        switch e {
        case .keyNotFound(let key, let context):
            return "Missing key '\(key.stringValue)' at \(pathString(context.codingPath)): \(context.debugDescription)"
        case .typeMismatch(let type, let context):
            return "Type mismatch (\(type)) at \(pathString(context.codingPath)): \(context.debugDescription)"
        case .valueNotFound(let type, let context):
            return "Missing value (\(type)) at \(pathString(context.codingPath)): \(context.debugDescription)"
        case .dataCorrupted(let context):
            return "Corrupted data at \(pathString(context.codingPath)): \(context.debugDescription)"
        @unknown default:
            return error.localizedDescription
        }
    }

    private func bodyPreviewString(data: Data, maxChars: Int) -> String {
        guard !data.isEmpty else { return "<empty>" }
        if let s = String(data: data, encoding: .utf8) {
            let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                return "<whitespace>"
            }
            if trimmed.count > maxChars {
                return String(trimmed.prefix(maxChars)) + "…"
            }
            return trimmed
        }
        return "<\(data.count) bytes>"
    }

    private func parseErrorDetail(data: Data) -> String? {
        guard !data.isEmpty else { return nil }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        guard let detail = json["detail"] else { return nil }

        if let s = detail as? String {
            return s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if let arr = detail as? [Any] {
            let items = arr.compactMap { item -> String? in
                if let s = item as? String { return s }
                if let d = item as? [String: Any] {
                    if let msg = d["msg"] as? String { return msg }
                    if let msg = d["message"] as? String { return msg }
                }
                return nil
            }
            if !items.isEmpty {
                return items.joined(separator: "\n")
            }
        }
        if let d = detail as? [String: Any] {
            if let msg = d["msg"] as? String { return msg }
            if let msg = d["message"] as? String { return msg }
        }
        return nil
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

struct HealthzResponse: Decodable {
    let status: String
}
