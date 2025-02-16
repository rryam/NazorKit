// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "NazorKit",
  platforms: [
    .iOS(.v16),
    .macOS(.v13),
    .visionOS(.v1),
  ],
  products: [
    // Products define the executables and libraries a package produces, making them visible to other packages.
    .library(
      name: "NazorKit",
      targets: ["NazorKit"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", branch: "main")
  ],
  targets: [
    // Targets are the basic building blocks of a package, defining a module or a test suite.
    // Targets can depend on other targets in this package and products from dependencies.
    .target(
      name: "NazorKit",
      dependencies: [
        .product(name: "MLXVLM", package: "mlx-swift-examples"),
        .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
      ])
  ]
)
