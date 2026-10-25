import Compression
import Foundation

/// Mảng đọc từ .npy (little-endian, C-order).
public struct NPYArray: Sendable {
    public let shape: [Int]
    public let descr: String
    public let data: Data

    public var count: Int { shape.reduce(1, *) }

    public func floats() throws -> [Float] {
        switch descr {
        case "<f4": return load(Float.self)
        case "<f8": return load(Double.self).map(Float.init)
        default: throw NPZ.Failure.unsupported("floats from \(descr)")
        }
    }
    public func doubles() throws -> [Double] {
        guard descr == "<f8" else { throw NPZ.Failure.unsupported("doubles from \(descr)") }
        return load(Double.self)
    }
    public func ints() throws -> [Int] {
        switch descr {
        case "<i4": return load(Int32.self).map(Int.init)
        case "<i8": return load(Int64.self).map(Int.init)
        default: throw NPZ.Failure.unsupported("ints from \(descr)")
        }
    }

    private func load<T>(_: T.Type) -> [T] {
        data.withUnsafeBytes { raw in Array(raw.bindMemory(to: T.self).prefix(count)) }
    }
}

/// Đọc .npz (zip, stored hoặc deflate) — đủ cho heads và golden vectors.
public enum NPZ {
    public enum Failure: Error { case malformed(String), unsupported(String) }

    public static func read(_ url: URL) throws -> [String: NPYArray] {
        let zip = try Data(contentsOf: url, options: .alwaysMapped)
        func u16(_ o: Int) -> Int { Int(zip[zip.startIndex + o]) | Int(zip[zip.startIndex + o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        func u64(_ o: Int) -> Int { u32(o) | u32(o + 4) << 32 }

        // End of central directory
        var eocd = zip.count - 22
        while eocd >= 0, u32(eocd) != 0x0605_4b50 { eocd -= 1 }
        guard eocd >= 0 else { throw Failure.malformed("EOCD") }
        var entries = u16(eocd + 10)
        var pos = u32(eocd + 16)
        if pos == 0xFFFF_FFFF || entries == 0xFFFF {   // zip64
            let loc = eocd - 20
            guard loc >= 0, u32(loc) == 0x0706_4b50 else { throw Failure.malformed("zip64 locator") }
            let rec = u64(loc + 8)
            entries = u64(rec + 32)
            pos = u64(rec + 48)
        }

        var out: [String: NPYArray] = [:]
        for _ in 0..<entries {
            guard u32(pos) == 0x0201_4b50 else { throw Failure.malformed("central entry") }
            let method = u16(pos + 10)
            var csize = u32(pos + 20), usize = u32(pos + 24)
            let nameLen = u16(pos + 28), extraLen = u16(pos + 30), commentLen = u16(pos + 32)
            var lho = u32(pos + 42)
            let name = String(decoding: zip[(zip.startIndex + pos + 46)..<(zip.startIndex + pos + 46 + nameLen)], as: UTF8.self)
            // zip64 extra
            var e = pos + 46 + nameLen
            let eEnd = e + extraLen
            while e + 4 <= eEnd {
                let id = u16(e), sz = u16(e + 2)
                if id == 1 {
                    var q = e + 4
                    if usize == 0xFFFF_FFFF { usize = u64(q); q += 8 }
                    if csize == 0xFFFF_FFFF { csize = u64(q); q += 8 }
                    if lho == 0xFFFF_FFFF { lho = u64(q) }
                }
                e += 4 + sz
            }
            pos = eEnd + commentLen

            let dataStart = lho + 30 + u16(lho + 26) + u16(lho + 28)
            let raw = zip.subdata(in: (zip.startIndex + dataStart)..<(zip.startIndex + dataStart + csize))
            let npy: Data
            switch method {
            case 0: npy = raw
            case 8: npy = try inflate(raw, size: usize)
            default: throw Failure.unsupported("zip method \(method)")
            }
            out[name.hasSuffix(".npy") ? String(name.dropLast(4)) : name] = try parseNPY(npy)
        }
        return out
    }

    private static func inflate(_ src: Data, size: Int) throws -> Data {
        var dst = Data(count: size)
        let n = dst.withUnsafeMutableBytes { d in
            src.withUnsafeBytes { s in
                compression_decode_buffer(d.bindMemory(to: UInt8.self).baseAddress!, size,
                                          s.bindMemory(to: UInt8.self).baseAddress!, src.count,
                                          nil, COMPRESSION_ZLIB)
            }
        }
        guard n == size else { throw Failure.malformed("inflate \(n) != \(size)") }
        return dst
    }

    private static func parseNPY(_ d: Data) throws -> NPYArray {
        let b = [UInt8](d.prefix(12))
        guard b.count >= 10, b[0] == 0x93, b[1..<6].elementsEqual(Array("NUMPY".utf8)) else { throw Failure.malformed("npy magic") }
        let major = b[6]
        let (hlen, hstart) = major == 1 ? (Int(b[8]) | Int(b[9]) << 8, 10)
                                        : (Int(b[8]) | Int(b[9]) << 8 | Int(b[10]) << 16 | Int(b[11]) << 24, 12)
        let header = String(decoding: d[(d.startIndex + hstart)..<(d.startIndex + hstart + hlen)], as: UTF8.self)
        func field(_ key: String) -> String? {
            guard let r = header.range(of: "'\(key)':") else { return nil }
            return String(header[r.upperBound...]).trimmingCharacters(in: .whitespaces)
        }
        guard let descrRaw = field("descr"), descrRaw.hasPrefix("'"),
              let q = descrRaw.dropFirst().firstIndex(of: "'") else { throw Failure.malformed("descr") }
        let descr = String(descrRaw[descrRaw.index(after: descrRaw.startIndex)..<q])
        if field("fortran_order")?.hasPrefix("True") == true { throw Failure.unsupported("fortran_order") }
        guard let sh = field("shape"), let open = sh.firstIndex(of: "("), let close = sh.firstIndex(of: ")") else { throw Failure.malformed("shape") }
        let shape = sh[sh.index(after: open)..<close].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        return NPYArray(shape: shape, descr: descr, data: d.subdata(in: (d.startIndex + hstart + hlen)..<d.endIndex))
    }
}
