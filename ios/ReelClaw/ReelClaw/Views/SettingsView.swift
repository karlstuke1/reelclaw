import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var session: SessionStore
    @State private var apiBaseURL: String = UserDefaults.standard.string(forKey: UserDefaultsKeys.apiBaseURL) ?? ""

    @State private var isTestingConnection: Bool = false
    @State private var connectionStatus: String?

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Button("Sign out", role: .destructive) {
                        session.signOut()
                    }
                } header: {
                    Text("Account")
                }

                Section {
                    TextField("API Base URL", text: $apiBaseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)

                    Text("For production, use the HTTPS API Gateway URL from Terraform (looks like https://<id>.execute-api.us-east-1.amazonaws.com).")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    Button("Save") {
                        let before = AppConfig.apiBaseURL

                        let trimmed = apiBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
                        if trimmed.isEmpty {
                            UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.apiBaseURL)
                        } else {
                            UserDefaults.standard.setValue(trimmed, forKey: UserDefaultsKeys.apiBaseURL)
                        }

                        let after = AppConfig.apiBaseURL
                        if !sameBackend(before, after) {
                            session.signOut(reason: "Backend changed. Please sign in again.")
                            connectionStatus = nil
                        }
                    }
                } header: {
                    Text("Backend")
                }

                Section {
                    Text(AppConfig.apiBaseURL.absoluteString)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)

                    Button {
                        Task { await testConnection() }
                    } label: {
                        if isTestingConnection {
                            HStack(spacing: 10) {
                                ProgressView()
                                Text("Testing…")
                            }
                        } else {
                            Text("Test connection")
                        }
                    }
                    .disabled(isTestingConnection)

                    if let connectionStatus, !connectionStatus.isEmpty {
                        Text(connectionStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                } header: {
                    Text("Connection")
                }
            }
            .navigationTitle("Settings")
        }
    }

    private func sameBackend(_ a: URL, _ b: URL) -> Bool {
        let aHost = (a.host ?? "").lowercased()
        let bHost = (b.host ?? "").lowercased()
        if aHost != bHost {
            return false
        }
        let aScheme = (a.scheme ?? "").lowercased()
        let bScheme = (b.scheme ?? "").lowercased()
        if aScheme != bScheme {
            return false
        }
        if a.port != b.port {
            return false
        }
        return true
    }

    private func testConnection() async {
        isTestingConnection = true
        defer { isTestingConnection = false }

        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { session.accessToken })
        do {
            let resp = try await api.healthz()
            connectionStatus = resp.status == "ok" ? "OK" : "Unexpected response: \(resp.status)"
        } catch {
            connectionStatus = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
        }
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
            .environmentObject(SessionStore())
    }
}
