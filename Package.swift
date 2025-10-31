// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "NazorKit",
  platforms: [
    .iOS(.v16),
    .macOS(.v14),
    .visionOS(.v1),
  ],
  products: [
    .library(
      name: "NazorKit",
      targets: ["NazorKit"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", branch: "main")
  ],
  targets: [
    .target(
      name: "NazorKit",
      dependencies: [
        .product(name: "MLXVLM", package: "mlx-swift-examples"),
        .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
      ])
  ]
)
