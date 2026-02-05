//
//  ReelClawApp.swift
//  ReelClaw
//
//  Created by Work on 31/01/2026.
//

import SwiftUI

@main
struct ReelClawApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var session = SessionStore()
    @StateObject private var router = AppRouter()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(session)
                .environmentObject(router)
        }
    }
}
