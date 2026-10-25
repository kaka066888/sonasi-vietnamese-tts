import Foundation
import OnnxRuntimeBindings

/// Helper mỏng quanh ORT Objective-C API.
enum ORT {
    nonisolated(unsafe) static let env: ORTEnv = try! ORTEnv(loggingLevel: .warning)

    /// Graph acoustic (vieneu_acoustic_cached) chứa MatMulInteger/DynamicQuantizeLinear (lượng tử hóa int8 động,
    /// kể cả trong thư mục fp32) nên kết quả phụ thuộc cách ORT fuse + kernel nền tảng. Đã đo (ORT 1.24.2/macOS):
    /// `.all`/`.extended` lệch ~4% ngay bước đầu, `.basic` lệch ~10% từ kênh 4; `.none` gần nhất với golden
    /// (13/20 câu trùng từng mã). Ngay trong Python/Linux `.none` cũng lệch `.all` ở một số kênh, nên không thể
    /// đòi trùng bit — chỉ kiểm hành vi. Prefill, decode_step và codec khớp ~1e-6 ở `.all`.
    /// CPU EP, 1 luồng inter-op, không busy-wait — giống bản Python.
    static func session(_ url: URL, threads: Int, level: ORTGraphOptimizationLevel = .all) throws -> ORTSession {
        let opts = try ORTSessionOptions()
        try opts.setGraphOptimizationLevel(level)
        try opts.setIntraOpNumThreads(Int32(threads))
        try opts.addConfigEntry(withKey: "session.intra_op.allow_spinning", value: "0")
        return try ORTSession(env: env, modelPath: url.path, sessionOptions: opts)
    }

    /// Vòng lặp dài chạy trên thread không có runloop (task nền/CLI) không bao giờ xả autorelease pool, nên mọi
    /// ORTValue/NSMutableData bị autorelease sẽ tích tụ (đo: ~75 MB/s → app bị iOS kill). Mọi gọi ORT đều bọc pool.
    static func run(_ s: ORTSession, _ inputs: [String: ORTValue], _ names: [String]) throws -> [String: ORTValue] {
        try autoreleasepool { try s.run(withInputs: inputs, outputNames: Set(names), runOptions: nil) }
    }

    private static func make<T>(_ a: [T], _ shape: [Int], _ type: ORTTensorElementDataType) throws -> ORTValue {
        try autoreleasepool { try makeUnpooled(a, shape, type) }
    }

    private static func makeUnpooled<T>(_ a: [T], _ shape: [Int], _ type: ORTTensorElementDataType) throws -> ORTValue {
        // NSMutableData(bytes:length:) rò bộ nhớ khi bọc vào ORTValue (đo: 0,5 MB/tensor không bao giờ trả lại);
        // cấp phát rồi memcpy thì không.
        let byteCount = a.count * MemoryLayout<T>.stride
        let data = NSMutableData(length: byteCount)!
        if byteCount > 0 { a.withUnsafeBytes { _ = memcpy(data.mutableBytes, $0.baseAddress!, byteCount) } }
        return try ORTValue(tensorData: data, elementType: type, shape: shape.map { NSNumber(value: $0) })
    }
    static func tensor(_ a: [Float], _ shape: [Int]) throws -> ORTValue { try make(a, shape, .float) }
    static func tensor(_ a: [Int32], _ shape: [Int]) throws -> ORTValue { try make(a, shape, .int32) }
    static func tensor(_ a: [Int64], _ shape: [Int]) throws -> ORTValue { try make(a, shape, .int64) }

    static func floats(_ v: ORTValue) throws -> [Float] {
        let d = try v.tensorData() as Data
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    static func int32s(_ v: ORTValue) throws -> [Int32] {
        let d = try v.tensorData() as Data
        return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }
    static func shape(_ v: ORTValue) throws -> [Int] {
        try v.tensorTypeAndShapeInfo().shape.map { $0.intValue }
    }
}
