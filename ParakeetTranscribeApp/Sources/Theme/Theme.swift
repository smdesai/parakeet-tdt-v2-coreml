import SwiftUI

enum Theme {
    static let background = Color(hex: 0x0B1120)
    static let aurora1 = Color(hex: 0x2DD4BF)   // teal
    static let aurora2 = Color(hex: 0x6366F1)   // indigo
    static let aurora3 = Color(hex: 0xEC4899)   // pink
    static let textSecondary = Color.white.opacity(0.6)

    static var backgroundGradient: LinearGradient {
        LinearGradient(
            colors: [Color(hex: 0x0B1120), Color(hex: 0x111A33), Color(hex: 0x0B1120)],
            startPoint: .top,
            endPoint: .bottom
        )
    }

    static var accentGradient: LinearGradient {
        LinearGradient(colors: [aurora1, aurora2], startPoint: .topLeading, endPoint: .bottomTrailing)
    }
}

extension Color {
    init(hex: UInt32) {
        let r = Double((hex >> 16) & 0xFF) / 255
        let g = Double((hex >> 8) & 0xFF) / 255
        let b = Double(hex & 0xFF) / 255
        self.init(.sRGB, red: r, green: g, blue: b, opacity: 1)
    }
}

struct GlassCard: ViewModifier {
    var padding: CGFloat = 16
    func body(content: Content) -> some View {
        content
            .padding(padding)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08))
            )
    }
}

extension View {
    func glassCard(padding: CGFloat = 16) -> some View { modifier(GlassCard(padding: padding)) }
}
