import SwiftUI

struct RootView: View {
    @EnvironmentObject private var engine: TranscriptionEngine

    var body: some View {
        ZStack {
            Theme.backgroundGradient.ignoresSafeArea()
            TranscriptionView()

            if case .failed(let message) = engine.phase {
                FailureOverlay(message: message)
            } else if engine.isPreparing {
                ProgressOverlay(
                    title: "Loading Parakeet models…",
                    message: "First load compiles and maps model weights."
                )
            } else if engine.isTranscribing {
                ProgressOverlay(
                    title: "Transcribing audio…",
                    message: "Running offline batch inference."
                )
            }
        }
        .tint(Theme.aurora1)
        .task {
            await engine.prepareIfNeeded()
        }
    }
}

private struct ProgressOverlay: View {
    let title: String
    let message: String

    var body: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()
            VStack(spacing: 14) {
                ProgressView()
                    .controlSize(.large)
                    .tint(Theme.aurora1)
                Text(title)
                    .font(.headline)
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(Theme.textSecondary)
            }
            .glassCard(padding: 28)
            .padding(40)
        }
    }
}

private struct FailureOverlay: View {
    @EnvironmentObject private var engine: TranscriptionEngine
    let message: String
    var body: some View {
        ZStack {
            Color.black.opacity(0.55).ignoresSafeArea()
            VStack(spacing: 14) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 36))
                    .foregroundStyle(Theme.aurora3)
                Text("Something went wrong")
                    .font(.headline)
                Text(message)
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(Theme.textSecondary)
                Button("Retry") {
                    Task { await engine.prepareIfNeeded() }
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.aurora1)
            }
            .glassCard(padding: 28)
            .padding(40)
        }
    }
}
