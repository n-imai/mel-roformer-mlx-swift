// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftSourceSeparation",
    platforms: [.macOS(.v15)],
    products: [
        .library(
            name: "SwiftRoFormer",
            targets: ["SwiftRoFormer"]
        ),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            from: "0.21.0"
        ),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            from: "1.1.6"
        ),
    ],
    targets: [
        .target(
            name: "SwiftRoFormer",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/SwiftRoFormer"
        ),
        .executableTarget(
            name: "roformer-validate",
            dependencies: [
                "SwiftRoFormer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/RoFormerValidate"
        ),
        .executableTarget(
            name: "roformer-dump",
            dependencies: [
                "SwiftRoFormer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/RoFormerDump"
        ),
        .executableTarget(
            name: "roformer-chunk",
            dependencies: [
                "SwiftRoFormer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/RoFormerChunk"
        ),
        .testTarget(
            name: "SwiftRoFormerTests",
            dependencies: [
                "SwiftRoFormer",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Tests/SwiftRoFormerTests"
        ),
    ]
)
