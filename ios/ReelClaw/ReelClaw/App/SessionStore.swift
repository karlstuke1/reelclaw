import AuthenticationServices
import Foundation
import UIKit
import UserNotifications

@MainActor
final class SessionStore: ObservableObject {
    @Published private(set) var accessToken: String?
    @Published private(set) var isSigningIn: Bool = false
    @Published private(set) var isRegisteringPush: Bool = false
    @Published private(set) var signInError: String?

    init() {
        let stored = KeychainStore.readString(key: KeychainStore.Keys.accessToken)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if let stored, !stored.isEmpty {
            self.accessToken = stored
        } else {
            // Avoid treating empty tokens as a logged-in session (causes 401 "missing Authorization header").
            KeychainStore.delete(key: KeychainStore.Keys.accessToken)
            self.accessToken = nil
        }

        NotificationCenter.default.addObserver(
            forName: .reelclawAPNSToken,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let self else { return }
            guard let token = note.object as? String, !token.isEmpty else { return }
            Task { await self.registerAPNSTokenIfNeeded(token) }
        }
    }

    func signOut() {
        KeychainStore.delete(key: KeychainStore.Keys.accessToken)
        accessToken = nil
        signInError = nil
    }

    func handleAppleSignIn(result: Result<ASAuthorization, Error>) async {
        switch result {
        case .failure(let error):
            isSigningIn = false
            signInError = error.localizedDescription
        case .success(let authorization):
            guard let credential = authorization.credential as? ASAuthorizationAppleIDCredential else {
                isSigningIn = false
                return
            }
            await signIn(with: credential)
        }
    }

    func signIn(with credential: ASAuthorizationAppleIDCredential) async {
        guard let identityTokenData = credential.identityToken,
              let identityToken = String(data: identityTokenData, encoding: .utf8),
              let authCodeData = credential.authorizationCode,
              let authorizationCode = String(data: authCodeData, encoding: .utf8)
        else {
            isSigningIn = false
            return
        }

        isSigningIn = true
        defer { isSigningIn = false }
        signInError = nil

        do {
            let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { nil })
            let response = try await api.signInWithApple(identityToken: identityToken, authorizationCode: authorizationCode)
            KeychainStore.saveString(response.accessToken, key: KeychainStore.Keys.accessToken)
            accessToken = response.accessToken
            await ensurePushRegistered()
        } catch {
            signInError = (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
            KeychainStore.delete(key: KeychainStore.Keys.accessToken)
            accessToken = nil
        }
    }

    func ensurePushRegistered() async {
        guard accessToken != nil else { return }

        isRegisteringPush = true
        defer { isRegisteringPush = false }

        do {
            let center = UNUserNotificationCenter.current()
            let settings = await center.notificationSettings()
            if settings.authorizationStatus == .notDetermined {
                let granted = try await center.requestAuthorization(options: [.alert, .badge, .sound])
                if !granted { return }
            } else if settings.authorizationStatus != .authorized && settings.authorizationStatus != .provisional {
                return
            }

            UIApplication.shared.registerForRemoteNotifications()
        } catch {
            // Best-effort; app can still poll.
        }
    }

    private func registerAPNSTokenIfNeeded(_ token: String) async {
        guard let accessToken else { return }

        let key = "reelclaw.lastAPNSToken"
        let last = UserDefaults.standard.string(forKey: key) ?? ""
        if last == token {
            return
        }

        let api = APIClient(baseURL: AppConfig.apiBaseURL, accessTokenProvider: { accessToken })
        do {
            #if DEBUG
            let env = "sandbox"
            #else
            let env = "production"
            #endif
            try await api.registerAPNSDevice(deviceToken: token, environment: env)
            UserDefaults.standard.setValue(token, forKey: key)
        } catch {
            // Keep token; we'll retry later (next launch/sign-in).
        }
    }
}
