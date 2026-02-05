import SwiftUI

struct RootView: View {
    @EnvironmentObject private var session: SessionStore

    var body: some View {
        Group {
            if session.accessToken == nil {
                SignInView()
            } else {
                MainView()
            }
        }
        .task(id: session.accessToken) {
            if session.accessToken != nil {
                await session.ensurePushRegistered()
            }
        }
    }
}

struct RootView_Previews: PreviewProvider {
    static var previews: some View {
        RootView()
            .environmentObject(SessionStore())
    }
}
