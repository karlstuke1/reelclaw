import AuthenticationServices
import SwiftUI

struct SignInView: View {
    @EnvironmentObject private var session: SessionStore

    var body: some View {
        VStack(spacing: 16) {
            VStack(spacing: 8) {
                Text("ReelClaw")
                    .font(.largeTitle.bold())
                Text("Paste a reel. Upload clips. Get 3 edits.")
                    .font(.headline)
                    .foregroundStyle(.secondary)
            }

            SignInWithAppleButton(.signIn) { request in
                request.requestedScopes = [.fullName, .email]
            } onCompletion: { result in
                Task { await session.handleAppleSignIn(result: result) }
            }
            .frame(height: 48)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .padding(.top, 8)

            if session.isSigningIn {
                ProgressView()
                    .padding(.top, 8)
            }

            if let message = session.signInError, !message.isEmpty {
                Text(message)
                    .foregroundStyle(.red)
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .padding(.top, 4)
            }

            Text("You must have rights to the videos and music you process.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.top, 12)
        }
        .padding(24)
        .frame(maxWidth: 460)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(.systemGroupedBackground))
    }
}

struct SignInView_Previews: PreviewProvider {
    static var previews: some View {
        SignInView()
            .environmentObject(SessionStore())
    }
}
