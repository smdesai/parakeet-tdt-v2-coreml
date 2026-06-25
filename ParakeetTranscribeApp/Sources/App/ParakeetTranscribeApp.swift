import SwiftUI

@main
struct ParakeetTranscribeApp: App {
    @StateObject private var engine = TranscriptionEngine()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(engine)
                .preferredColorScheme(.dark)
                .tint(Theme.aurora1)
        }
    }
}
