import SwiftUI

struct RootView: View {
    @EnvironmentObject private var engine: TranscriptionEngine

    var body: some View {
        ZStack {
            Theme.backgroundGradient.ignoresSafeArea()
            TranscriptionView()

            if case .failed(let message) = engine.phase {
                FailureOverlay(message: message)
            } else if case .downloading(let progress) = engine.phase {
                DownloadOverlay(progress: progress)
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

/// First-launch determinate progress screen shown while the Parakeet model
/// bundle downloads from Hugging Face. Distinct from the indeterminate
/// `ProgressOverlay` used for the compile/map and transcribe waits.
///
/// The download is range-chunked and parallel, so many files advance at once
/// and there is no meaningful "current file" — the honest, monotonic readout is
/// aggregate bytes completed vs. total (see `DownloadProgress`).
private struct DownloadOverlay: View {
    let progress: DownloadProgress

    private static let byteFormatter: ByteCountFormatter = {
        let f = ByteCountFormatter()
        f.countStyle = .file
        f.allowedUnits = [.useMB, .useGB]
        return f
    }()

    private var byteReadout: String {
        let done = Self.byteFormatter.string(fromByteCount: progress.bytesCompleted)
        let total = Self.byteFormatter.string(fromByteCount: max(progress.bytesTotal, 0))
        return "\(done) of \(total)"
    }

    /// Whole-percent readout, shown once the total is known.
    private var percentText: String? {
        guard progress.bytesTotal > 0 else { return nil }
        return "\(Int((progress.fraction * 100).rounded()))%"
    }

    var body: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()
            VStack(spacing: 14) {
                Text("Downloading Parakeet models…")
                    .font(.headline)
                Text("One-time first-launch download of the speech models.")
                    .font(.footnote)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(Theme.textSecondary)

                ProgressView(value: progress.fraction)
                    .tint(Theme.aurora1)
                    .frame(maxWidth: 260)

                Text(byteReadout)
                    .font(.subheadline.monospacedDigit())

                if let percentText {
                    Text(percentText)
                        .font(.footnote.monospacedDigit())
                        .foregroundStyle(Theme.textSecondary)
                }
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
