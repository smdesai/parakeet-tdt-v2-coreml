import SwiftUI
import UIKit
import UniformTypeIdentifiers
import ParakeetCore

struct TranscriptionView: View {
    @EnvironmentObject private var engine: TranscriptionEngine
    @State private var showAudioImporter = false

    var body: some View {
        VStack(spacing: 16) {
            header
            transcriptCard
            if let metrics = engine.batchMetrics {
                BatchPerformanceView(metrics: metrics)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
            WaveformView(level: engine.level, active: engine.isRecording)
                .frame(height: 56)
                .padding(.horizontal, 4)
            controls
                .padding(.bottom, 8)
        }
        .padding(20)
        .fileImporter(
            isPresented: $showAudioImporter,
            allowedContentTypes: [.audio],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            Task { await engine.transcribeFile(url) }
        }
    }

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Parakeet TDT")
                    .font(.title2.bold())
                HStack(spacing: 6) {
                    Image(systemName: "cpu")
                        .font(.caption2)
                    Text("\(engine.backendLabel)")
                        .font(.caption.weight(.medium))
                }
                .foregroundStyle(Theme.aurora1)
            }
            Spacer()
        }
    }

    private var transcriptCard: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Text(engine.transcript.isEmpty ? placeholder : engine.transcript)
                    .font(.system(.body, design: .rounded))
                    .foregroundStyle(engine.transcript.isEmpty ? Theme.textSecondary : .white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .id("transcript")
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .glassCard()
            .overlay(alignment: .topTrailing) {
                if engine.isRecording || engine.isTranscribing {
                    Label(engine.isTranscribing ? "BATCH" : "REC",
                          systemImage: engine.isTranscribing ? "waveform" : "dot.radiowaves.left.and.right")
                        .font(.caption2.bold())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Theme.aurora3, in: Capsule())
                        .padding(12)
                }
            }
            .overlay(alignment: .bottomTrailing) {
                if !engine.transcript.isEmpty {
                    Button {
                        UIPasteboard.general.string = engine.transcript
                    } label: {
                        Image(systemName: "doc.on.doc")
                            .padding(10)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .padding(12)
                    .tint(.white)
                }
            }
            .onChange(of: engine.transcript) { _, _ in
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo("transcript", anchor: .bottom)
                }
            }
        }
    }

    private var placeholder: String {
        if engine.isTranscribing { return "Transcribing…" }
        if engine.isRecording { return "Recording… tap stop to transcribe." }
        return "Tap the mic to record, or the folder to pick an audio file. "
             + "Speech is transcribed entirely on-device."
    }

    /// Two actions: record (mic) and import a file. Disabled while busy.
    private var controls: some View {
        HStack(spacing: 36) {
            recordButton
            fileButton
        }
    }

    private var recordButton: some View {
        Button {
            engine.toggleRecording()
        } label: {
            ZStack {
                Circle()
                    .fill(engine.isRecording ? AnyShapeStyle(Theme.aurora3) : AnyShapeStyle(Theme.accentGradient))
                    .frame(width: 84, height: 84)
                    .shadow(color: (engine.isRecording ? Theme.aurora3 : Theme.aurora1).opacity(0.5), radius: 16)
                Image(systemName: engine.isRecording ? "stop.fill" : "mic.fill")
                    .font(.system(size: 32, weight: .bold))
                    .foregroundStyle(.white)
            }
        }
        .disabled(busy)
        .opacity(busy ? 0.5 : 1)
        .animation(.spring(duration: 0.3), value: engine.isRecording)
    }

    private var fileButton: some View {
        Button {
            showAudioImporter = true
        } label: {
            ZStack {
                Circle()
                    .fill(.ultraThinMaterial)
                    .frame(width: 64, height: 64)
                    .overlay(Circle().strokeBorder(Color.white.opacity(0.12)))
                Image(systemName: "folder.fill")
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundStyle(.white)
            }
        }
        .disabled(busyOrRecording)
        .opacity(busyOrRecording ? 0.5 : 1)
    }

    private var busy: Bool { engine.isPreparing || engine.isTranscribing }
    private var busyOrRecording: Bool { busy || engine.isRecording }
}

private struct BatchPerformanceView: View {
    let metrics: TranscriptionEngine.BatchMetrics
    /// Per-stage breakdown is collapsed by default; tap the header to reveal it.
    @State private var showStages = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeOut(duration: 0.2)) { showStages.toggle() }
            } label: {
                headerRow
            }
            .buttonStyle(.plain)
            .disabled(metrics.stages.isEmpty)

            if showStages && !metrics.stages.isEmpty {
                stageBreakdown
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        }
        .animation(.easeOut(duration: 0.2), value: metrics)
    }

    /// Always-visible RTFx headline. Carries the disclosure chevron when there's a
    /// per-stage breakdown to expand (file path only; the mic path has no stages).
    private var headerRow: some View {
        HStack(spacing: 12) {
            Image(systemName: "speedometer")
                .font(.title3.weight(.semibold))
                .foregroundStyle(Theme.aurora1)
                .frame(width: 36, height: 36)
                .background(Theme.aurora1.opacity(0.14), in: Circle())

            VStack(alignment: .leading, spacing: 3) {
                HStack(alignment: .firstTextBaseline, spacing: 5) {
                    Text(formatRTFx(metrics.rtfx))
                        .font(.headline.monospacedDigit().weight(.semibold))
                        .foregroundStyle(.white)
                    Text("RTFx")
                        .font(.caption.bold())
                        .foregroundStyle(Theme.textSecondary)
                }

                Text("\(metrics.backendLabel) · \(formatDuration(metrics.audioDuration)) audio · \(formatDuration(metrics.transcriptionDuration)) transcribe")
                    .font(.caption2)
                    .foregroundStyle(Theme.textSecondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)

            if !metrics.stages.isEmpty {
                Image(systemName: "chevron.down")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Theme.textSecondary)
                    .rotationEffect(.degrees(showStages ? 180 : 0))
            }
        }
        .contentShape(Rectangle())
    }

    /// Per-stage CoreML busy time + an overlap verdict. The 3-stage pipeline runs
    /// preprocess (CPU+GPU) ∥ encode (ANE) ∥ decode (CPU); if it overlaps well the
    /// wall time ≈ the busiest stage, and overlap% → 100. If it serializes the wall
    /// ≈ the sum of stages and overlap% → 0.
    private var stageBreakdown: some View {
        let wall = metrics.transcriptionDuration
        let sum = metrics.stages.reduce(0) { $0 + $1.seconds }
        let maxStage = metrics.stages.map(\.seconds).max() ?? 0
        // 0% when wall == sum (no overlap), 100% when wall == maxStage (full overlap).
        let denom = max(sum - maxStage, 0.0001)
        let overlap = max(0, min(1, (sum - wall) / denom))

        return VStack(alignment: .leading, spacing: 4) {
            Divider().overlay(Color.white.opacity(0.08))
            ForEach(metrics.stages, id: \.label) { s in
                HStack(spacing: 6) {
                    Text(s.label)
                        .font(.caption2.monospaced())
                        .foregroundStyle(Theme.textSecondary)
                        .frame(width: 84, alignment: .leading)
                    Text(formatDuration(s.seconds))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.white)
                        .frame(width: 52, alignment: .trailing)
                    Text("\(percent(sum > 0 ? s.seconds / sum : 0))")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(Theme.textSecondary)
                    Spacer(minLength: 0)
                }
            }
            Text("Σ busy \(formatDuration(sum)) · wall \(formatDuration(wall)) · overlap \(percent(overlap))")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(Theme.aurora1)
                .padding(.top, 1)
        }
    }

    private func percent(_ f: Double) -> String { String(format: "%.0f%%", f * 100) }

    private func formatRTFx(_ value: Double) -> String {
        if value >= 100 { return String(format: "%.0fx", value) }
        if value >= 10 { return String(format: "%.1fx", value) }
        return String(format: "%.2fx", value)
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        if duration >= 60 {
            let totalSeconds = Int(duration.rounded())
            return "\(totalSeconds / 60):\(String(format: "%02d", totalSeconds % 60))"
        }
        if duration >= 10 { return String(format: "%.1fs", duration) }
        return String(format: "%.2fs", duration)
    }
}
