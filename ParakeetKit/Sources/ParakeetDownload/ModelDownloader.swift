//
//  ModelDownloader.swift
//  ParakeetTranscribe
//
//  Range-chunked PARALLEL downloader for the Parakeet-TDT-v2 CoreML model
//  bundle. Ported from `coreai-model-zoo/apps/AppShared/ModelDownloader.swift`
//  and adapted for this app's FLAT Hugging Face repo
//  (`smdesai/parakeet-tdt-0.6b-v2-coreml`): the four runtime `.mlmodelc`
//  directories and `parakeet_vocab.json` sit directly under the repo root, so
//  the on-device install root IS the `modelsDir` the rest of the app consumes.
//
//  Why range-chunked parallelism: the payload is one huge file
//  (`parakeet_encoder.mlmodelc/weights/weight.bin` ≈ 594 MB — 96.8% of the
//  ~613 MB total) plus ~20 tiny files. A single HF/CDN connection is
//  bandwidth-capped, so a one-stream-per-file loop gives that one big file zero
//  parallelism. Instead every file larger than `chunkSize` is split into
//  byte-range chunks pulled CONCURRENTLY over a pool of independent
//  `URLSession`s (one HTTP/2 connection each); aggregate throughput scales with
//  the pool size until the radio / bandwidth-delay-product caps it.
//
//  Resumable across launches: each file has a sidecar chunk bitmap (one byte per
//  chunk) under `<root>/.dl-progress/<rel>.bits`. A chunk sets its bit only AFTER
//  its bytes are written and the file handle is closed, so `bit == 1 ⟹ bytes
//  present`. A re-run reads the bitmaps and re-downloads only the missing chunks,
//  so a quit/crash/OS-kill costs at most the one in-flight chunk, never the whole
//  594 MB.
//
//  On-device layout (writable; excluded from backup):
//
//    Application Support/parakeet-tdt-v2-coreml/
//        ├── parakeet_preprocessor.mlmodelc/...
//        ├── parakeet_encoder.mlmodelc/...
//        ├── parakeet_decoder.mlmodelc/...
//        ├── parakeet_joint_decision_single_step.mlmodelc/...
//        ├── parakeet_vocab.json
//        ├── .dl-progress/...                      (chunk bitmaps; swept on success)
//        └── .complete                             (sentinel — written LAST)
//
//  The `.complete` sentinel is the atomic completion barrier for this single
//  flat install root (the flat-repo analog of coreai's per-bundle atomic
//  rename): it is written only after every chunk of every file is durable, and
//  `isInstalled()` checks solely for it. A partial download (no sentinel) is
//  treated as "not installed" and resumed on the next launch. Nothing loads the
//  models until the sentinel exists (load is gated behind `ensureInstalled`), so
//  CoreML never sees a half-written `.mlmodelc`.
//
//  TESTABILITY: the download-*planning* logic (repo-id parsing, JSON decode,
//  allowlist filter, resolve-URL construction, chunk geometry, bitmap-resume
//  decision) is factored into pure, synchronous `internal static` functions
//  callable with no URLSession. The impure edges — `fetchFileList()` (network)
//  and the chunk transfer core — are thin wrappers around that pure core.
//

import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Repository-level configuration for the Hugging Face download source.
/// Re-pointing the app at a new model export is a one-line change here.
public enum HFRepo {
    /// Owner + repo name on huggingface.co. Files are resolved from
    /// `https://huggingface.co/<id>/resolve/main/<path>`.
    public static let id: String = "smdesai/parakeet-tdt-0.6b-v2-coreml"
    /// Git revision. "main" tracks the latest commit on the default branch.
    public static let revision: String = "main"
}

/// Download progress snapshot emitted on the main actor during the download.
///
/// This is an AGGREGATE model: with range-chunked parallelism, chunks of
/// multiple files land out of order, so a "file X of N / current filename"
/// readout is meaningless (several files advance at once). The honest,
/// monotonic signal is total bytes completed vs. total bytes expected — that's
/// all this carries.
public struct DownloadProgress: Sendable, Equatable {
    /// Bytes durably written so far (sum across all chunks of all files).
    public let bytesCompleted: Int64
    /// Total bytes expected (sum across all files). Zero until the file list
    /// has been fetched and the plan is built.
    public let bytesTotal: Int64

    public init(bytesCompleted: Int64, bytesTotal: Int64) {
        self.bytesCompleted = bytesCompleted
        self.bytesTotal = bytesTotal
    }

    public static let zero = DownloadProgress(bytesCompleted: 0, bytesTotal: 0)

    /// Completed fraction in 0...1 (0 until the total is known).
    public var fraction: Double {
        bytesTotal > 0 ? min(Double(bytesCompleted) / Double(bytesTotal), 1) : 0
    }
}

/// Errors surfaced by the downloader. All cases include a user-facing
/// description via `LocalizedError`.
public enum ModelDownloadError: LocalizedError {
    case networkError(String)
    case httpStatus(Int, String)
    case invalidResponse(String)
    case noFilesFound(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .networkError(let m): return "Network error: \(m)"
        case .httpStatus(let code, let m): return "HTTP \(code): \(m)"
        case .invalidResponse(let m): return "Invalid response: \(m)"
        case .noFilesFound(let m): return "No files found: \(m)"
        case .cancelled: return "Download cancelled"
        }
    }
}

/// Downloads the Parakeet-TDT-v2 model bundle from Hugging Face into
/// Application Support using range-chunked parallelism with cross-launch resume.
///
/// Usage:
///   let dl = ModelDownloader()
///   try await dl.ensureInstalled { progress in
///       // update UI from progress.fraction / bytes
///   }
public final class ModelDownloader: Sendable {

    public init() {}

    // MARK: - Install layout

    /// Folder name (under Application Support) that holds the model bundle.
    static let installFolderName = "parakeet-tdt-v2-coreml"

    /// Empty file written last, only after every chunk of every file is durable.
    /// `isInstalled()` checks solely for this.
    static let sentinelName = ".complete"

    /// Hidden directory under the install root holding per-file chunk bitmaps.
    static let progressDirName = ".dl-progress"

    // MARK: - Transfer tuning

    /// Size of the session pool = number of independent TCP connections
    /// streaming chunks at once. Each is a separate `URLSession` (one HTTP/2
    /// connection each), so aggregate throughput scales with this until the
    /// device radio / bandwidth-delay-product caps it. 8 is a safe default,
    /// well under HF's per-client request-COUNT rate limit.
    static let maxConnections = 8

    /// Files larger than this are split into byte-range chunks of this size.
    /// 16 MiB keeps peak RAM (chunkSize × maxConnections ≈ 128 MB) modest while
    /// making the progress bar advance smoothly.
    static let chunkSize: Int64 = 16 * 1024 * 1024

    /// Per-chunk retry budget (a failed chunk re-fetches only its own ≤chunkSize
    /// slice). Each retry re-hits the HF resolve URL, re-rolling onto a
    /// possibly-different (live) CDN edge.
    static let maxChunkRetries = 6

    /// Wall-clock deadline for one chunk GET. 30 s for 16 MiB ⇒ a ~0.5 MB/s
    /// floor — far below real Wi-Fi, so only genuinely stuck transfers are cut.
    static let chunkDeadlineSeconds: UInt64 = 30

    /// Re-applies the `Range` header across the HF→CDN 302 (see
    /// `RangePreservingRedirector`).
    private static let redirector = RangePreservingRedirector()

    // MARK: - Runtime-set allowlist
    //
    // We keep a tree entry ONLY if its top-level path segment is one of the
    // four runtime model directories, OR the path is exactly the vocab file.
    // Everything else (`.gitattributes`, hidden `.`-prefixed paths, README.md,
    // any future stray `*.mlmodelc` such as the removed raw-logit joint) is
    // excluded. Expressed as an ALLOWLIST so a newly re-added stray artifact is
    // excluded by default rather than shipped by accident.

    /// Top-level model directory names whose entire subtree is downloaded.
    static let allowedModelDirectories: Set<String> = [
        "parakeet_preprocessor.mlmodelc",
        "parakeet_encoder.mlmodelc",
        "parakeet_decoder.mlmodelc",
        "parakeet_joint_decision_single_step.mlmodelc",
    ]

    /// Standalone files at the repo root that are allowed (the vocab).
    static let allowedRootFiles: Set<String> = [
        "parakeet_vocab.json"
    ]

    /// Allowlist predicate: is `path` (relative to the repo root) a runtime
    /// artifact we should download? Pure and synchronous.
    static func isAllowed(path: String) -> Bool {
        // Exact match on an allowed standalone root file (the vocab).
        if allowedRootFiles.contains(path) { return true }
        // Otherwise the top-level path segment must be an allowed model
        // directory (i.e. the entry lives inside one of the four models).
        guard let top = path.split(separator: "/").first else { return false }
        return allowedModelDirectories.contains(String(top))
    }

    // MARK: - Paths

    /// Install root for the model bundle. Created on demand; survives app
    /// restarts; excluded from iCloud backup.
    public static func rootDirectory() -> URL {
        let fm = FileManager.default
        let base: URL
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            base = appSupport
        } else {
            base = fm.temporaryDirectory
        }
        let root = base.appendingPathComponent(installFolderName, isDirectory: true)
        if !fm.fileExists(atPath: root.path) {
            try? fm.createDirectory(at: root, withIntermediateDirectories: true)
            // Mark the whole tree as "do not back up" — these files are
            // large, derived, and re-downloadable on demand.
            excludeFromBackup(root)
        }
        return root
    }

    /// True if the bundle has been fully downloaded previously (sentinel
    /// file exists). Does NOT re-validate every artifact — the sentinel is
    /// only written after every chunk has been successfully written.
    public static func isInstalled() -> Bool {
        let sentinel = rootDirectory().appendingPathComponent(sentinelName)
        return FileManager.default.fileExists(atPath: sentinel.path)
    }

    // MARK: - Public API

    /// Ensure the bundle is installed locally. If it already is (sentinel
    /// present), returns immediately. Otherwise fetches the file list from the
    /// HF tree API, plans byte-range chunks (resuming any partial from a prior
    /// run via the on-disk bitmaps), and downloads the missing chunks in
    /// parallel. Progress is reported via `onProgress` on the main actor.
    ///
    /// Single-flight is provided by the caller: `TranscriptionEngine`'s
    /// `.downloading` phase guard prevents a re-entrant call while one is in
    /// flight. Retry from the failure overlay re-enters here and resumes,
    /// because already-downloaded chunks are skipped by their bitmap bit.
    @MainActor
    public func ensureInstalled(
        onProgress: @escaping @Sendable @MainActor (DownloadProgress) -> Void = { _ in }
    ) async throws {
        if Self.isInstalled() { return }

        onProgress(.zero)

        let files = try await fetchFileList()
        guard !files.isEmpty else {
            throw ModelDownloadError.noFilesFound("HF repo \(HFRepo.id) returned no files")
        }

        let root = Self.rootDirectory()
        let fm = FileManager.default
        let progressRoot = root.appendingPathComponent(Self.progressDirName)
        try fm.createDirectory(at: progressRoot, withIntermediateDirectories: true)
        Self.excludeFromBackup(progressRoot)

        // Plan: for each file load its completed-chunk bitmap (resuming a prior
        // partial) or start it fresh, and enqueue only the missing chunks.
        var segments: [Segment] = []
        var totalBytes: Int64 = 0
        var completedBytes: Int64 = 0

        for file in files {
            try Task.checkCancellation()
            let destFile = root.appendingPathComponent(file.relativePath)
            let bits = progressRoot.appendingPathComponent(file.relativePath + ".bits")
            try fm.createDirectory(at: destFile.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            try fm.createDirectory(at: bits.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)

            let geo = Self.chunkGeometry(file.size)
            // Trust a prior partial only if its bitmap matches this file's
            // chunking AND the staging file is still there; otherwise reset
            // this file (empty file + zero map).
            var done = [UInt8](repeating: 0, count: geo.count)
            let saved = try? Data(contentsOf: bits)
            if Self.canResume(savedBitmapCount: saved?.count,
                              expectedChunkCount: geo.count,
                              destinationExists: fm.fileExists(atPath: destFile.path)),
               let saved {
                done = [UInt8](saved)
            } else {
                fm.createFile(atPath: destFile.path, contents: nil)
                try Data(count: geo.count).write(to: bits)
            }

            let url = Self.downloadURL(relativePath: file.relativePath)
            for (ci, chunk) in geo.enumerated() {
                totalBytes += chunk.length
                if done[ci] != 0 {
                    completedBytes += chunk.length          // already on disk from a prior run
                } else {
                    segments.append(Segment(url: url, dest: destFile, offset: chunk.offset,
                                            length: chunk.length, ranged: chunk.ranged,
                                            chunkIndex: ci, bits: bits))
                }
            }
        }

        // Progress emission, throttled to ~50 UI updates per download so the
        // bar advances smoothly without flooding the main actor.
        var shownFraction = -1.0
        func emit(force: Bool = false) {
            let f = totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 0
            guard force || f - shownFraction >= 0.002 || f >= 1 else { return }
            shownFraction = f
            onProgress(DownloadProgress(bytesCompleted: completedBytes, bytesTotal: totalBytes))
        }
        emit(force: true)

        if !segments.isEmpty {
            // Hold a background-task assertion for the whole transfer so a brief
            // trip to the background (app switcher / screen lock) doesn't get
            // the app suspended out from under the download.
            let bg = BackgroundAssertion(name: "parakeet-model-download")
            defer { bg.end() }

            // A POOL of independent sessions = independent TCP connections. A
            // single URLSession multiplexes all its tasks onto ONE HTTP/2
            // connection per host, so fanning N chunks over one session caps at
            // ~1 stream and collapses (-1005) when overloaded. Each fan-out slot
            // gets its OWN long-lived session so aggregate throughput scales.
            let cfg = URLSessionConfiguration.default
            cfg.httpMaximumConnectionsPerHost = 1
            // MUST be false: on iOS a dead CDN connection drops with -1005; with
            // waitsForConnectivity=true the retry then blocks forever "waiting
            // for connectivity" and the download hangs silently at 0 bytes.
            cfg.waitsForConnectivity = false
            cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
            // IDLE timeout (resets on each received packet); the per-chunk
            // wall-clock deadline below is what actually caps a stalled transfer.
            cfg.timeoutIntervalForRequest = 25
            cfg.timeoutIntervalForResource = 7 * 24 * 60 * 60
            let pool = (0..<max(1, Self.maxConnections)).map { _ in URLSession(configuration: cfg) }
            defer { pool.forEach { $0.invalidateAndCancel() } }

            // Bounded fan-out: keep `maxConnections` chunks in flight, refilling
            // as each lands. Bookkeeping runs serially here on the main actor; a
            // thrown chunk cancels the rest and falls through to the caller.
            // Each fan-out slot is pinned to its own pool session (= its own
            // connection); when a slot's chunk lands, the next chunk reuses that
            // same session so the connection persists.
            try await withThrowingTaskGroup(of: (Int64, Int).self) { group in
                var iterator = segments.makeIterator()
                var inFlight = 0
                var slot = 0
                for _ in 0..<pool.count {
                    guard let seg = iterator.next() else { break }
                    let s = slot; slot += 1
                    let sess = pool[s]
                    group.addTask {
                        try await Self.fetchChunk(seg, via: sess, deadline: Self.chunkDeadlineSeconds)
                        return (seg.length, s)
                    }
                    inFlight += 1
                }
                while inFlight > 0 {
                    let (len, freed) = try await group.next()!
                    inFlight -= 1
                    completedBytes += len
                    emit()
                    if let next = iterator.next() {
                        let sess = pool[freed]
                        group.addTask {
                            try await Self.fetchChunk(next, via: sess, deadline: Self.chunkDeadlineSeconds)
                            return (next.length, freed)
                        }
                        inFlight += 1
                    }
                }
            }
        }

        // Every chunk is durable now → sweep the bitmap dir and write the
        // sentinel LAST. The sentinel is the ONLY signal used to decide "is the
        // bundle installed" on subsequent launches.
        try? fm.removeItem(at: progressRoot)
        try Data().write(to: root.appendingPathComponent(Self.sentinelName), options: .atomic)
        completedBytes = totalBytes
        emit(force: true)
    }

    // MARK: - Pure planning core (network-free, unit-testable)

    /// A single file to download, resolved from the HF tree.
    struct RemoteFile: Sendable, Equatable {
        /// Path relative to the install root, e.g.
        /// "parakeet_encoder.mlmodelc/weights/weight.bin" or
        /// "parakeet_vocab.json". Never starts with `/`.
        let relativePath: String
        /// Size in bytes. For LFS files this is the resolved size (from the
        /// `lfs` object), not the pointer size. 0 if the API didn't report it.
        let size: Int64

        init(relativePath: String, size: Int64) {
            self.relativePath = relativePath
            self.size = size
        }
    }

    /// One byte-range of one file: where to GET it, where to write it, and which
    /// bitmap bit marks it done. Files at or below `chunkSize` are a single
    /// un-ranged segment (a plain whole-file GET).
    private struct Segment: Sendable {
        let url: URL
        let dest: URL           // staging file this chunk writes into
        let offset: Int64       // byte offset within the file / dest
        let length: Int64
        let ranged: Bool        // send a Range header (false = whole small file)
        let chunkIndex: Int     // index into the file's bitmap
        let bits: URL           // sidecar bitmap file (one byte per chunk)
    }

    /// One planned chunk of a file: its byte range and whether it needs a Range
    /// header. Pure output of `chunkGeometry`, exposed for unit tests.
    struct Chunk: Equatable, Sendable {
        let offset: Int64
        let length: Int64
        let ranged: Bool
    }

    /// HF tree API entry shape. We only decode the fields we use. LFS files
    /// carry their resolved size in a nested `lfs` object; a plain file uses the
    /// top-level `size`.
    struct TreeEntry: Decodable, Equatable, Sendable {
        let type: String   // "file" or "directory"
        let path: String   // path from repo root, e.g. "parakeet_encoder.mlmodelc/model.mil"
        let size: Int64?   // bytes; present for files, absent for directories
        let lfs: LFS?

        struct LFS: Decodable, Equatable, Sendable {
            let size: Int64?
        }

        init(type: String, path: String, size: Int64?, lfs: LFS? = nil) {
            self.type = type
            self.path = path
            self.size = size
            self.lfs = lfs
        }
    }

    /// Parse a Hugging Face repo id from either a full URL
    /// ("https://huggingface.co/<org>/<name>[/...]") or a bare "<org>/<name>".
    /// Pure; returns nil if the string isn't a recognizable HF repo reference.
    static func repoId(from s: String) -> String? {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if let u = URL(string: t), let host = u.host, host.hasSuffix("huggingface.co") {
            let parts = u.path.split(separator: "/").map(String.init)
            return parts.count >= 2 ? "\(parts[0])/\(parts[1])" : nil
        }
        let parts = t.split(separator: "/").map(String.init)
        return parts.count == 2 ? t : nil
    }

    /// Decode the HF tree API response bytes into `[TreeEntry]`. Deterministic
    /// for a given `Data`; throws a typed parse error.
    static func decodeTree(from data: Data) throws -> [TreeEntry] {
        do {
            return try JSONDecoder().decode([TreeEntry].self, from: data)
        } catch {
            throw ModelDownloadError.invalidResponse(
                "Could not parse tree response: \(error.localizedDescription)"
            )
        }
    }

    /// Pure planner: given the decoded tree entries, return the filtered,
    /// deterministically-sorted download plan. Keeps only `file` entries that
    /// pass the runtime-set allowlist; directories and non-runtime artifacts are
    /// dropped. LFS resolved size (`lfs.size`) is preferred over the top-level
    /// `size` so a repo that reports a pointer size still chunks correctly. No
    /// network, no filesystem.
    static func planFiles(from entries: [TreeEntry]) -> [RemoteFile] {
        var out: [RemoteFile] = []
        for entry in entries where entry.type == "file" {
            guard isAllowed(path: entry.path) else { continue }
            let size = entry.lfs?.size ?? entry.size ?? 0
            out.append(RemoteFile(relativePath: entry.path, size: size))
        }
        out.sort { $0.relativePath < $1.relativePath }
        return out
    }

    /// Build the `resolve` URL for a single file. Pattern:
    /// https://huggingface.co/<repo>/resolve/<rev>/<relPath>
    static func downloadURL(relativePath: String) -> URL {
        var u = URL(string: "https://huggingface.co")!
            .appendingPathComponent(HFRepo.id)
            .appendingPathComponent("resolve")
            .appendingPathComponent(HFRepo.revision)
        for segment in relativePath.split(separator: "/") {
            u.appendPathComponent(String(segment))
        }
        return u
    }

    /// Byte-range geometry for a file: a single whole-file segment if it's at or
    /// below `chunkSize`, otherwise contiguous `chunkSize` ranges covering
    /// `[0, size)` with no gaps or overlap (the last chunk is the remainder).
    /// Pure; unit-tested directly.
    static func chunkGeometry(_ size: Int64) -> [Chunk] {
        guard size > chunkSize else { return [Chunk(offset: 0, length: max(size, 0), ranged: false)] }
        var out: [Chunk] = []
        var off: Int64 = 0
        while off < size {
            let len = min(chunkSize, size - off)
            out.append(Chunk(offset: off, length: len, ranged: true))
            off += len
        }
        return out
    }

    /// Bitmap-resume decision: can a saved per-file chunk bitmap be trusted to
    /// resume a partial download? Its length must match this file's chunk count
    /// AND the staging file must still exist (else the bits would point at bytes
    /// that aren't there). Pure; used by `ensureInstalled` and unit-tested.
    static func canResume(savedBitmapCount: Int?, expectedChunkCount: Int,
                          destinationExists: Bool) -> Bool {
        guard let n = savedBitmapCount else { return false }
        return n == expectedChunkCount && destinationExists
    }

    // MARK: - Impure edges

    /// Enumerate every runtime file in the repo via the HF tree API in one call.
    /// Thin network wrapper: fetch bytes → decode → run the pure allowlist
    /// planner.
    func fetchFileList() async throws -> [RemoteFile] {
        let u = URL(string: "https://huggingface.co/api/models")!
            .appendingPathComponent(HFRepo.id)
            .appendingPathComponent("tree")
            .appendingPathComponent(HFRepo.revision)
        var comps = URLComponents(url: u, resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "recursive", value: "true")]
        let apiURL = comps.url!

        var req = URLRequest(url: apiURL)
        req.httpMethod = "GET"
        req.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await Self.listingSession.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw ModelDownloadError.invalidResponse("Non-HTTP response from \(apiURL)")
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw ModelDownloadError.httpStatus(
                http.statusCode, "Failed to list files at \(apiURL.path)")
        }

        let entries = try Self.decodeTree(from: data)
        return Self.planFiles(from: entries)
    }

    /// Session for the small tree-API listing call (no caching; short timeout).
    private static let listingSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.timeoutIntervalForRequest = 30
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }()

    /// Fetch one chunk, write it at its offset in the staging file, then mark
    /// its bitmap bit. The data write+close happens BEFORE the bit is set so a
    /// crash can never leave bit==1 over missing bytes (a re-download just
    /// rewrites the same slice). Retries re-fetch only this chunk.
    private static func fetchChunk(_ seg: Segment, via session: URLSession,
                                   deadline: UInt64) async throws {
        var attempt = 0
        while true {
            do {
                var req = URLRequest(url: seg.url)
                if seg.ranged {
                    req.setValue("bytes=\(seg.offset)-\(seg.offset + seg.length - 1)",
                                 forHTTPHeaderField: "Range")
                }
                let (data, resp) = try await dataWithDeadline(req, via: session, seconds: deadline)
                let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
                // A ranged GET must answer 206; a whole-file GET 200. Anything
                // else (e.g. the CDN ignored the Range and sent 200) would
                // mis-place bytes, so fail loudly instead.
                guard seg.ranged ? code == 206 : code == 200 else {
                    throw ModelDownloadError.httpStatus(code, seg.url.lastPathComponent)
                }
                let fh = try FileHandle(forWritingTo: seg.dest)
                do {
                    try fh.seek(toOffset: UInt64(seg.offset))
                    try fh.write(contentsOf: data)
                    try fh.close()
                } catch { try? fh.close(); throw error }
                // Bytes are durable in the OS file now → record the chunk done.
                let bh = try FileHandle(forWritingTo: seg.bits)
                do {
                    try bh.seek(toOffset: UInt64(seg.chunkIndex))
                    try bh.write(contentsOf: Data([1]))
                    try bh.close()
                } catch { try? bh.close(); throw error }
                return
            } catch {
                if Task.isCancelled { throw error }     // group is tearing down — don't retry
                attempt += 1
                if attempt > maxChunkRetries { throw error }
                try? await Task.sleep(nanoseconds: UInt64(attempt) * 500_000_000)  // 0.5s, 1s, …
            }
        }
    }

    /// A wall-clock deadline around one GET (redirect + body).
    /// `timeoutIntervalForRequest` is only an IDLE timer that resets on every
    /// received byte, so a connection that degrades to a crawl (HF's shared
    /// HTTP/2 connection sometimes does) never trips it and the download wedges
    /// at a fixed byte count. This races the transfer against a hard timeout and
    /// cancels the loser, so a stalled chunk fails fast and its retry re-rolls
    /// the resolve onto a fresh connection (possibly a different CDN edge).
    private static func dataWithDeadline(_ req: URLRequest, via session: URLSession,
                                         seconds: UInt64) async throws -> (Data, URLResponse) {
        try await withThrowingTaskGroup(of: (Data, URLResponse).self) { group in
            group.addTask { try await session.data(for: req, delegate: redirector) }
            group.addTask {
                try await Task.sleep(nanoseconds: seconds * 1_000_000_000)
                throw ModelDownloadError.networkError("chunk stalled > \(seconds)s")
            }
            defer { group.cancelAll() }                 // cancel the loser (transfer or timer)
            guard let first = try await group.next() else {
                throw ModelDownloadError.networkError("no chunk result")
            }
            return first
        }
    }

    private static func excludeFromBackup(_ url: URL) {
        var v = URLResourceValues()
        v.isExcludedFromBackup = true
        var u = url
        try? u.setResourceValues(v)
    }
}

// A no-op-on-macOS wrapper around a UIKit finite-length background-task
// assertion. Holding one keeps the app from being suspended for a short grace
// period after it leaves the foreground, so a quick app-switch mid-download
// doesn't drop the connection.
private final class BackgroundAssertion {
    #if canImport(UIKit)
    private var id: UIBackgroundTaskIdentifier = .invalid
    @MainActor init(name: String) {
        // The expiration handler ends the task via `self` (not a captured
        // mutable local), so it always sees the assigned id and the OS can
        // reclaim the assertion if our grace period runs out.
        id = UIApplication.shared.beginBackgroundTask(withName: name) { [weak self] in
            self?.end()
        }
    }
    @MainActor func end() {
        guard id != .invalid else { return }
        UIApplication.shared.endBackgroundTask(id)
        id = .invalid
    }
    #else
    init(name: String) {}
    func end() {}
    #endif
}

// Carries the `Range` header onto the redirected request. HF `resolve/...`
// 302-redirects to the CDN; URLSession usually copies headers across a redirect,
// but if it ever dropped Range the CDN would send the full file (200) and the
// chunk's data(for:) would buffer the whole ~594 MB into RAM. This guarantees
// the redirected GET stays a 206 partial.
private final class RangePreservingRedirector: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping (URLRequest?) -> Void) {
        var req = request
        if let range = task.originalRequest?.value(forHTTPHeaderField: "Range") {
            req.setValue(range, forHTTPHeaderField: "Range")
        }
        completionHandler(req)
    }
}
