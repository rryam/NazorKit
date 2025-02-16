# Getting Started with NazorKit

NazorKit makes it incredibly easy to add Vision Language Model (VLM) capabilities to your iOS, macOS, or visionOS app.

## Overview

NazorKit provides a simple SwiftUI-first API for using Vision Language Models in your app. With just a few lines of code, you can analyze images and videos using state-of-the-art models.

### Basic Usage

Here's a simple example of how to use NazorKit in your SwiftUI view:

```swift
struct ContentView: View {
    // Create a VLM service using the property wrapper
    @VLMServiceProvider private var vlmService
    
    // Your image state
    @State private var image: UIImage?
    @State private var generatedDescription: String = ""
    
    var body: some View {
        VStack {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    // Add VLM capabilities with a simple modifier
                    .vlm(
                        service: vlmService,
                        prompt: "Describe this image in detail",
                        image: image
                    ) { description in
                        generatedDescription = description
                    }
                
                Text(generatedDescription)
                    .padding()
            }
            
            // Your image picker UI
        }
    }
}
```

### Advanced Usage

You can customize the model and generation parameters:

```swift
struct AdvancedView: View {
    // Customize the VLM service
    @VLMServiceProvider(
        configuration: .qwen2VL2BInstruct4Bit,
        generateParameters: .init(temperature: 0.8),
        maxTokens: 1000
    ) private var vlmService
    
    var body: some View {
        // Your view content
    }
}
```

### Video Analysis

NazorKit also supports video analysis:

```swift
struct VideoAnalysisView: View {
    @VLMServiceProvider private var vlmService
    @State private var videoURL: URL?
    @State private var analysis: String = ""
    
    var body: some View {
        VStack {
            if let videoURL {
                VideoPlayer(player: AVPlayer(url: videoURL))
                    .frame(height: 300)
                    .vlm(
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
}
```

## Topics

### Essentials
- ``VLMServiceProvider``
- ``VLMService``
- ``ModelConfiguration``

### View Modifiers
- ``View/vlm(service:prompt:image:video:onCompletion:)`` 