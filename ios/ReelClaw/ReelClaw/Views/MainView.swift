import SwiftUI

struct MainView: View {
    @EnvironmentObject private var router: AppRouter

    var body: some View {
        TabView(selection: $router.selectedTab) {
            ContentView()
                .tabItem { Label("Create", systemImage: "wand.and.stars") }
                .tag(AppTab.create)

            EditsView()
                .tabItem { Label("My Edits", systemImage: "play.rectangle.on.rectangle") }
                .tag(AppTab.jobs)

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
                .tag(AppTab.settings)
        }
    }
}

struct MainView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
            .environmentObject(SessionStore())
            .environmentObject(AppRouter())
            .environmentObject(RecentReferenceStore())
    }
}
