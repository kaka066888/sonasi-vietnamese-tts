// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SonasiVietnameseKit",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "SonasiVietnameseKit", targets: ["SonasiVietnameseKit"])],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", exact: "1.24.2"),
    ],
    targets: [
        .binaryTarget(name: "CSeaG2P", path: "Frameworks/CSeaG2P.xcframework"),
        .target(name: "SonasiVietnameseKit", dependencies: [
            "CSeaG2P",
            .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
        ], swiftSettings: [
            // Debug của app host sẽ không tối ưu Swift (RTF gấp đôi); thư viện này là vòng lặp số học nên luôn -O.
            .unsafeFlags(["-O"], .when(configuration: .debug)),
        ]),
        .executableTarget(name: "SonasiBench", dependencies: ["SonasiVietnameseKit"]),
        .testTarget(name: "SonasiVietnameseKitTests", dependencies: ["SonasiVietnameseKit"]),
    ]
)
