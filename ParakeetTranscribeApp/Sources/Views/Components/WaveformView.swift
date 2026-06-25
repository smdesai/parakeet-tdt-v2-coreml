import SwiftUI

/// Rolling level meter: a fixed number of bars scrolling left, the newest bar on
/// the right driven by the engine's input level.
struct WaveformView: View {
    var level: Float
    var active: Bool

    private let barCount = 48
    @State private var bars: [Float] = Array(repeating: 0, count: 48)
    private let timer = Timer.publish(every: 0.05, on: .main, in: .common).autoconnect()

    var body: some View {
        GeometryReader { geo in
            HStack(alignment: .center, spacing: 3) {
                ForEach(bars.indices, id: \.self) { i in
                    Capsule()
                        .fill(active ? AnyShapeStyle(Theme.accentGradient) : AnyShapeStyle(Color.white.opacity(0.12)))
                        .frame(height: max(3, CGFloat(bars[i]) * geo.size.height))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        }
        .onReceive(timer) { _ in
            var next = bars
            next.removeFirst()
            next.append(active ? max(0, level).squareRoot() : 0)
            bars = next
        }
    }
}
