# NazorKit
[![Star History Chart](https://api.star-history.com/svg?repos=rryam/NazorKit&type=Date)](https://star-history.com/#rryam/NazorKit&Date)


NazorKit is a library built on top of MLX-Swift to easily integrate on-device vision language models into your iOS app. 

The name "Nazor" is inspired by the Persian word "نظر" - "nazar" meaning vision/sight/gaze).

<p align="center">
  <img src="https://img.shields.io/badge/Swift-6.0+-fa7343?style=flat&logo=swift&logoColor=white" alt="Swift 6.0+">
  <br>
  <img src="https://img.shields.io/badge/iOS-16.0+-000000?style=flat&logo=apple&logoColor=white" alt="iOS 16.0+">
  <img src="https://img.shields.io/badge/macOS-14.0+-000000?style=flat&logo=apple&logoColor=white" alt="macOS 14.0+">
  <img src="https://img.shields.io/badge/visionOS-1.0+-000000?style=flat&logo=apple&logoColor=white" alt="visionOS 1.0+">
</p>

## Installation

Swift Package Manager handles the distribution of Swift code and comes built into the Swift compiler.

To add NazorKit to your project, simply include it in your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/rryam/NazorKit.git", .upToNextMajor(from: "0.1.0"))
]
```

Or add NazorKit to your project through Xcode's package manager:

1. In Xcode, go to File > Add Packages...
2. Enter the package URL: `https://github.com/rryam/NazorKit`
3. Select the version or branch you want to use (e.g. `main`)
4. Click Add Package

## Quick Start

Get up and running with NazorKit in minutes. Here is an example of analyzing an image:

```swift
import NazorKit
import SwiftUI

struct ContentView: View {
    @VLMServiceProvider private var vlmService
    @State private var image: UIImage?
    @State private var generatedDescription: String = ""
    
    var body: some View {
        VStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .analyzeMedia(
                        service: vlmService,
                        prompt: "Describe this image in detail",
                        image: image
                    ) { description in
                        generatedDescription = description
                    }
                
                Text(generatedDescription)
                    .padding()
            }
        }
    }
}
```

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Configuration](#advanced-configuration)
- [Video Analysis](#video-analysis)
- [Requirements](#requirements)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## Features

- SwiftUI-first API design
- Support for iOS 16.0+, macOS 14.0+, and visionOS 1.0+
- Image analysis capabilities
- Video analysis support
- Built on top of MLX for efficient model inference
- Customizable model configurations
- Easy-to-use property wrappers and view modifiers

## Basic Usage

Here's a simple example of how to analyze an image using NazorKit:

```swift
struct ContentView: View {
    @VLMServiceProvider private var vlmService
    @State private var image: UIImage?
    @State private var generatedDescription: String = ""
    
    var body: some View {
        VStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .analyzeMedia(
                        service: vlmService,
                        prompt: "Describe this image in detail",
                        image: image
                    ) { description in
                        generatedDescription = description
                    }
                
                Text(generatedDescription)
                    .padding()
            }
        }
    }
}
```

## Advanced Configuration

You can customize the VLM service with specific model configurations:

```swift
@VLMServiceProvider(
    configuration: .qwen2VL2BInstruct4Bit,
    generateParameters: .init(temperature: 0.8),
    maxTokens: 1000
) private var vlmService
```

### Custom Generation Parameters

You can fine-tune the generation process with custom parameters:

```swift
let generateParameters = GenerateParameters(
    temperature: 0.8,  // Controls randomness (0.0-1.0)
    topP: 0.9          // Nucleus sampling parameter
)

@VLMServiceProvider(
    configuration: .qwen2VL2BInstruct4Bit,
    generateParameters: generateParameters,
    maxTokens: 1000
) private var vlmService
```

## Video Analysis

NazorKit also supports video analysis:

```swift
import AVKit

struct VideoAnalysisView: View {
    @VLMServiceProvider private var vlmService
    @State private var analysis: String = ""
    let videoURL: URL
    
    var body: some View {
        VStack {
            VideoPlayer(player: AVPlayer(url: videoURL))
                .frame(height: 300)
                .analyzeMedia(
                    service: vlmService,
                    prompt: "What's happening in this video?",
                    video: videoURL
                ) { description in
                    analysis = description
                }
            
            Text(analysis)
                .padding()
        }
    }
}
```

## Contributing

I welcome contributions to NazorKit! Here is how you can help:

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the existing code style
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

### Development Setup

1. Clone the repository
2. Open `Package.swift` in Xcode or VS Code forks or CLIs
3. Run tests to ensure everything works
4. Make your changes and test them

### Code Style

- Follow SwiftLint rules (run `swiftlint lint`)
- Use Swift 6.0+ features where appropriate

## License

NazorKit is available under the MIT license. See [LICENSE](LICENSE) for more information.

## Support

- [Issues](https://github.com/rryam/NazorKit/issues)
- [Discussions](https://github.com/rryam/NazorKit/discussions)
- [Twitter](https://x.com/rudrankriyam)

## Acknowledgments

- Thanks to the MLX team for their excellent work on the MLX and the MLX Swift framework!