import CSeaG2P
import Foundation

/// Vỏ Swift mỏng quanh C ABI của sea-g2p. Một instance không thread-safe.
public final class SeaG2P: @unchecked Sendable {
    public struct Failure: Error { public let message: String }

    private let handle: OpaquePointer

    public init(dictionaryPath: String) throws {
        guard sea_g2p_abi_version() == SEA_G2P_ABI_VERSION else {
            throw Failure(message: "sea-g2p ABI \(sea_g2p_abi_version()) != \(SEA_G2P_ABI_VERSION)")
        }
        guard let h = sea_g2p_open(dictionaryPath) else { throw Failure(message: Self.lastError()) }
        handle = h
    }

    deinit { sea_g2p_close(handle) }

    public func phonemize(_ text: String, puncNorm: Bool = false) throws -> String {
        try Self.onLargeStack { try Self.take(sea_g2p_phonemize(self.handle, text, puncNorm ? 1 : 0)) }
    }

    /// Quy tắc dấu câu cuối của sea-g2p (câu < 5 từ ép một "."; câu dài thêm "." nếu chưa kết thúc). Không cần handle.
    public static func puncNorm(_ text: String) throws -> String {
        try onLargeStack { try take(sea_g2p_punc_norm(text)) }
    }

    public func normalize(_ text: String, puncNorm: Bool = false) throws -> String {
        try Self.onLargeStack { try Self.take(sea_g2p_normalize(self.handle, text, puncNorm ? 1 : 0)) }
    }

    /// Phần Rust dùng nhiều stack; thread của dispatch/Swift Concurrency chỉ có ~512 KB và bị SIGBUS.
    private static func onLargeStack<T>(_ body: @escaping () throws -> T) throws -> T {
        let box = ResultBox<T>()
        let done = DispatchSemaphore(value: 0)
        let thread = Thread {
            box.result = Result { try body() }
            done.signal()
        }
        thread.stackSize = 16 << 20
        thread.start()
        done.wait()
        return try box.result!.get()
    }

    private static func take(_ p: UnsafeMutablePointer<CChar>?) throws -> String {
        guard let p else { throw Failure(message: Self.lastError()) }
        defer { sea_g2p_string_free(p) }
        return String(cString: p)
    }

    private static func lastError() -> String { sea_g2p_last_error().map { String(cString: $0) } ?? "unknown" }
}

private final class ResultBox<T>: @unchecked Sendable { var result: Result<T, Error>? }
