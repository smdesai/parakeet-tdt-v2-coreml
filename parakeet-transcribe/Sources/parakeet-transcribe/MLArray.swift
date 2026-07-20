import Accelerate
import CoreML
import Foundation

/// MLMultiArray <-> Swift helpers, fp16/fp32 aware, with stride-correct reads.
/// Adapted from the verified idioms in parakeet-unified's CoreMLParakeet.swift.
enum MLArray {
    static func empty(shape: [Int], dataType: MLMultiArrayDataType) -> MLMultiArray {
        try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
    }

    static func float(_ values: [Float], shape: [Int], dataType: MLMultiArrayDataType) -> MLMultiArray {
        let arr = empty(shape: shape, dataType: dataType)
        let count = values.count
        switch dataType {
        case .float16:
            arr.withUnsafeMutableBytes { ptr, _ in
                let dst = ptr.bindMemory(to: Float16.self)
                for i in 0..<count { dst[i] = Float16(values[i]) }
            }
        case .float32:
            arr.withUnsafeMutableBytes { ptr, _ in
                let dst = ptr.bindMemory(to: Float32.self)
                for i in 0..<count { dst[i] = values[i] }
            }
        default:
            for i in 0..<count { arr[i] = NSNumber(value: values[i]) }
        }
        return arr
    }

    static func int32(_ values: [Int32], shape: [Int]) -> MLMultiArray {
        let arr = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        arr.withUnsafeMutableBytes { ptr, _ in
            let dst = ptr.bindMemory(to: Int32.self)
            for i in 0..<values.count { dst[i] = values[i] }
        }
        return arr
    }

    /// First scalar of an array as Int (handles int32/float backing).
    static func firstInt(_ arr: MLMultiArray) -> Int {
        guard arr.count > 0 else { return 0 }
        switch arr.dataType {
        case .int32:
            return Int(arr.withUnsafeBytes { $0.bindMemory(to: Int32.self)[0] })
        case .float32:
            return Int(arr.withUnsafeBytes { $0.bindMemory(to: Float32.self)[0] }.rounded())
        case .float16:
            return Int(Float(arr.withUnsafeBytes { $0.bindMemory(to: Float16.self)[0] }).rounded())
        default:
            return arr[0].intValue
        }
    }

    /// Read an MLMultiArray into a logical C-contiguous `[Float]`, respecting
    /// strides. ANE outputs can be padded / non-contiguous; a linear read of the
    /// raw buffer would scramble the values.
    static func floats(_ arr: MLMultiArray) -> [Float] {
        let shape = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }
        let count = arr.count

        var contiguous = [Int](repeating: 1, count: shape.count)
        if shape.count >= 2 {
            for d in stride(from: shape.count - 2, through: 0, by: -1) {
                contiguous[d] = contiguous[d + 1] * shape[d + 1]
            }
        }
        let isContiguous = (strides == contiguous)

        var out = [Float](repeating: 0, count: count)
        arr.withUnsafeBytes { raw in
            @inline(__always) func value(at off: Int) -> Float {
                switch arr.dataType {
                case .float16: return Float(raw.load(fromByteOffset: off * 2, as: Float16.self))
                case .float32: return raw.load(fromByteOffset: off * 4, as: Float32.self)
                case .float64, .double: return Float(raw.load(fromByteOffset: off * 8, as: Float64.self))
                default: return 0
                }
            }
            if isContiguous {
                for i in 0..<count { out[i] = value(at: i) }
            } else {
                var index = [Int](repeating: 0, count: shape.count)
                for linear in 0..<count {
                    var offset = 0
                    for d in 0..<shape.count { offset += index[d] * strides[d] }
                    out[linear] = value(at: offset)
                    var d = shape.count - 1
                    while d >= 0 {
                        index[d] += 1
                        if index[d] < shape[d] { break }
                        index[d] = 0
                        d -= 1
                    }
                }
            }
        }
        return out
    }

    /// Argmax over the first `count` token logits, with optional additive keyword
    /// bonuses (shallow fusion). `count` is the number of token logits to scan —
    /// pass `Const.vocabSize + 1` (== 1025) so the blank id (index 1024) stays
    /// selectable.
    ///
    /// Parity contract: when `bonuses` is empty this returns the PLAIN argmax over
    /// the raw logits, identical to the joint model's internal argmax. When
    /// non-empty, we take the plain argmax (fast vDSP path on fp32, the joint's
    /// CPU_ONLY output dtype) and then compare only the small set of boosted ids
    /// against the current winner — so the cost is O(count) + O(#bonuses), not
    /// O(count · #bonuses). Out-of-range boosted ids (t < 0 || t >= count) are
    /// ignored.
    static func argmax(logits: MLMultiArray, count: Int, bonuses: [Int32: Float]) -> Int {
        precondition(count > 0, "argmax needs a positive count")
        let n = min(count, logits.count)

        // 1) Plain max over the raw logits, dtype-aware.
        var bestIdx = 0
        var bestVal: Float = -.greatestFiniteMagnitude

        switch logits.dataType {
        case .float32 where logits.strides.last?.intValue == 1:
            logits.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: Float32.self)
                var maxVal: Float = 0
                var maxIdx: vDSP_Length = 0
                vDSP_maxvi(p.baseAddress!, 1, &maxVal, &maxIdx, vDSP_Length(n))
                bestVal = maxVal
                bestIdx = Int(maxIdx)
            }
        case .float16 where logits.strides.last?.intValue == 1:
            logits.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: Float16.self)
                for i in 0..<n {
                    let v = Float(p[i])
                    if v > bestVal { bestVal = v; bestIdx = i }
                }
            }
        default:
            // Strided / other dtype: fall back to a stride-correct float read.
            let vals = floats(logits)
            for i in 0..<n {
                let v = vals[i]
                if v > bestVal { bestVal = v; bestIdx = i }
            }
        }

        // 2) Empty bonuses => plain argmax (byte-identical baseline parity).
        if bonuses.isEmpty { return bestIdx }

        // 3) Non-empty: only the boosted ids can change the winner. Read each
        //    boosted id's raw logit, add its bonus, and keep the best. Ties resolve
        //    to the plain winner (strictly-greater comparison, plain max seeded).
        for (id, bonus) in bonuses {
            let t = Int(id)
            guard t >= 0, t < n else { continue }   // bounds-guard boosted ids
            let raw = rawLogit(logits, at: t)
            let boosted = raw + bonus
            if boosted > bestVal { bestVal = boosted; bestIdx = t }
        }
        return bestIdx
    }

    /// Single logit at a flat index, respecting the last-axis stride and dtype.
    /// Used by `argmax` to re-read only the boosted ids without materializing the
    /// whole vector.
    @inline(__always)
    private static func rawLogit(_ arr: MLMultiArray, at index: Int) -> Float {
        let stride = arr.strides.last?.intValue ?? 1
        let off = index * stride
        return arr.withUnsafeBytes { raw -> Float in
            switch arr.dataType {
            case .float16: return Float(raw.load(fromByteOffset: off * 2, as: Float16.self))
            case .float32: return raw.load(fromByteOffset: off * 4, as: Float32.self)
            case .float64, .double: return Float(raw.load(fromByteOffset: off * 8, as: Float64.self))
            default: return 0
            }
        }
    }

    /// Fill a [1, channels, 1] encoder-step array from a flat C-major buffer:
    /// value(channel) = data[channel * frames + frame].
    static func fillChannelStep(_ out: MLMultiArray, from data: [Float],
                                channels: Int, frames: Int, frame: Int) {
        switch out.dataType {
        case .float16:
            out.withUnsafeMutableBytes { ptr, _ in
                let dst = ptr.bindMemory(to: Float16.self)
                for ch in 0..<channels { dst[ch] = Float16(data[ch * frames + frame]) }
            }
        case .float32:
            out.withUnsafeMutableBytes { ptr, _ in
                let dst = ptr.bindMemory(to: Float32.self)
                for ch in 0..<channels { dst[ch] = data[ch * frames + frame] }
            }
        default:
            for ch in 0..<channels { out[ch] = NSNumber(value: data[ch * frames + frame]) }
        }
    }
}
