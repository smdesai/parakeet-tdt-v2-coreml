//
//  Float16Compat.swift
//  ParakeetCore
//
//  ParakeetCore only ever runs on Apple silicon (the Core ML models are ANE/GPU
//  fp16 and the app targets are arm64-only). The Swift standard library marks
//  `Float16` unavailable on macOS x86_64, so a universal Xcode build (e.g. a
//  Release configuration where ONLY_ACTIVE_ARCH is off) could not even compile
//  the x86_64 slice of this module.
//
//  This 2-byte stand-in keeps that dead slice compiling. It is never executed:
//  every entry point traps, and `MemoryLayout<Float16>.size` stays 2 so the
//  byte-offset arithmetic in MLArray matches the real type's layout.
//

#if !arch(arm64)
    struct Float16 {
        var bitPattern: UInt16

        init(_ value: Float) {
            fatalError("ParakeetCore: Float16 is unsupported on x86_64; this slice must never run.")
        }
    }

    extension Float {
        init(_ half: Float16) {
            fatalError("ParakeetCore: Float16 is unsupported on x86_64; this slice must never run.")
        }
    }
#endif
