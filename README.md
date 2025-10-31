# NazorKit

NazorKit is a library built on top of MLX-Swift to easily integrate on-device vision language models into your iOS app. 

The name "Nazor" is inspired by the Persian word "نظر" - "nazar" meaning vision/sight/gaze).

## Features

- SwiftUI-first API design
- Support for iOS 16.0+, macOS 13.0+, and visionOS 1.0+
- Image analysis capabilities
- Video analysis support
- Built on top of MLX for efficient model inference
- Customizable model configurations
- Easy-to-use property wrappers and view modifiers

## Installation

### Swift Package Manager

Add NazorKit to your project through Xcode's package manager:

1. In Xcode, go to File > Add Packages...
2. Enter the package URL: `https://github.com/rryam/NazorKit`
3. Select the version or branch you want to use (e.g. `main`)
4. Click Add Package

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

## Video Analysis

NazorKit also supports video analysis:

```swift
VideoPlayer(player: AVPlayer(url: videoURL))
    .frame(height: 300)
    .analyzeMedia(
        service: vlmService,
        prompt: "What's happening in this video?",
        video: videoURL
    ) { description in
        analysis = description
    }
```

## Requirements

- iOS 16.0 or later
- macOS 13.0 or later
- visionOS 1.0 or later
- Swift 6.0 or later

## Dependencies

- [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples) - For efficient model inference

## License

[MIT License](LICENSE)

## Contributing

[Contributing Guidelines](CONTRIBUTING.md)

## Acknowledgments

- Thanks to the MLX team for their excellent work on the MLX and the MLX Swift framework
