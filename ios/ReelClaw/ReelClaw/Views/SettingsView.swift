import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var session: SessionStore
    @State private var apiBaseURL: String = UserDefaults.standard.string(forKey: UserDefaultsKeys.apiBaseURL) ?? ""

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
                        let trimmed = apiBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)
                        if trimmed.isEmpty {
                            UserDefaults.standard.removeObject(forKey: UserDefaultsKeys.apiBaseURL)
                        } else {
                            UserDefaults.standard.setValue(trimmed, forKey: UserDefaultsKeys.apiBaseURL)
                        }
                    }
                } header: {
                    Text("Backend")
                }
            }
            .navigationTitle("Settings")
        }
    }
}

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
            .environmentObject(SessionStore())
    }
}
