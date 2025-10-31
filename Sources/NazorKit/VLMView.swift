import CoreImage
import MLXLMCommon
import MLXVLM
import SwiftUI

/// A property wrapper that provides access to a VLM service
@propertyWrapper
@MainActor
public struct VLMServiceProvider: DynamicProperty {
    @StateObject private var serviceHolder: ServiceHolder
    
    public var wrappedValue: VLMService {
        serviceHolder.service
    }
    
    public init(
        configuration: ModelConfiguration = VLMRegistry.qwen2VL2BInstruct4Bit,
        generateParameters: GenerateParameters = .init(temperature: 0.6),
        maxTokens: Int = 800
    ) {
        _serviceHolder = StateObject(
            wrappedValue: ServiceHolder(
                configuration: configuration,
                generateParameters: generateParameters,
                maxTokens: maxTokens
            )
        )
    }
    
    @MainActor
    private final class ServiceHolder: ObservableObject {
        let service: VLMService
        
        init(
            configuration: ModelConfiguration = VLMRegistry.qwen2VL2BInstruct4Bit,
            generateParameters: GenerateParameters = .init(temperature: 0.6),
            maxTokens: Int = 800
        ) {
            self.service = VLMService(
                configuration: configuration,
                generateParameters: generateParameters,
                maxTokens: maxTokens
            )
        }
    }
}

/// A view modifier that adds VLM capabilities to a view
@MainActor
public struct VLMViewModifier: ViewModifier {
    @State private var isGenerating = false
    @State private var generatedText = ""
    @State private var error: Error?
    @State private var showError = false
    
    private let service: VLMService
    private let prompt: String
    private let image: CIImage?
    private let video: URL?
    private let onCompletion: ((String) -> Void)?
    
    public init(
        service: VLMService,
        prompt: String,
        image: CIImage? = nil,
        video: URL? = nil,
        onCompletion: ((String) -> Void)? = nil
    ) {
        self.service = service
        self.prompt = prompt
        self.image = image
        self.video = video
        self.onCompletion = onCompletion
    }
    
    public func body(content: Content) -> some View {
        content
            .overlay {
                if isGenerating {
                    ProgressView()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(.ultraThinMaterial)
                }
            }
            .task {
                let localService = service
                
                // Create a copy of the update handler that's isolated to the main actor
                @MainActor func updateText(_ text: String) {
                    generatedText = text
                }
                
                // Convert image to Data before passing to non-isolated context
                let imageData: Data?
                if let image = image {
                    let context = CIContext()
                    if let cgImage = context.createCGImage(image, from: image.extent) {
                        // Create a temporary bitmap context to draw the image
                        let width = cgImage.width
                        let height = cgImage.height
                        let colorSpace = CGColorSpaceCreateDeviceRGB()
                        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
                        
                        if let bitmapContext = CGContext(
                            data: nil,
                            width: width,
                            height: height,
                            bitsPerComponent: 8,
                            bytesPerRow: width * 4,
                            space: colorSpace,
                            bitmapInfo: bitmapInfo
                        ) {
                            // Draw the image into the context
                            let rect = CGRect(x: 0, y: 0, width: width, height: height)
                            bitmapContext.draw(cgImage, in: rect)
                            
                            // Get the data from the context
                            if let data = bitmapContext.makeImage() {
                                // Convert to JPEG data
                                imageData = context.jpegRepresentation(of: CIImage(cgImage: data), colorSpace: colorSpace)
                            } else {
                                imageData = nil
                            }
                        } else {
                            imageData = nil
                        }
                    } else {
                        imageData = nil
                    }
                } else {
                    imageData = nil
                }
                
                do {
                    await MainActor.run { isGenerating = true }
                    
                    let result = try await localService.generate(
                        prompt: prompt,
                        imageData: imageData,  
                        video: video
                    ) { text in
                        Task { @MainActor in
                            updateText(text)
                        }
                    }
                    
                    
                    await MainActor.run {
                        generatedText = result.output
                        onCompletion?(result.output)
                    }
                } catch {
                    await MainActor.run {
                        self.error = error
                        self.showError = true
                    }
                }
                
                await MainActor.run { isGenerating = false }
            }
            .alert("Error", isPresented: $showError) {
                Button("OK") {
                    error = nil
                    showError = false
                }
            } message: {
                if let error {
                    Text(error.localizedDescription)
                }
            }
    }
}

extension View {
    
    /// Adds VLM capabilities to analyze an image or video with the given prompt
    /// - Parameters:
    ///   - service: The VLM service to use
    ///   - prompt: The prompt to use for analysis
    ///   - image: Optional image to analyze
    ///   - video: Optional video to analyze
    ///   - onCompletion: Optional completion handler called with the generated text
    /// - Returns: A view with VLM capabilities
    @MainActor
    public func vlm(
        service: VLMService,
        prompt: String,
        image: CIImage? = nil,
        video: URL? = nil,
        onCompletion: ((String) -> Void)? = nil
    ) -> some View {
        modifier(
            VLMViewModifier(
                service: service,
                prompt: prompt,
                image: image,
                video: video,
                onCompletion: onCompletion
            )
        )
    }
}

#if os(iOS) || os(visionOS)
extension View {
    
    /// Adds VLM capabilities to analyze a UIImage with the given prompt
    /// - Parameters:
    ///   - service: The VLM service to use
    ///   - prompt: The prompt to use for analysis
    ///   - image: The UIImage to analyze
    ///   - onCompletion: Optional completion handler called with the generated text
    /// - Returns: A view with VLM capabilities
    @ViewBuilder
    public func vlm(
        service: VLMService,
        prompt: String,
        image: UIImage,
        onCompletion: ((String) -> Void)? = nil
    ) -> some View {
        if let ciImage = CIImage(image: image) {
            vlm(
                service: service,
                prompt: prompt,
                image: ciImage,
                onCompletion: onCompletion
            )
        } else {
            EmptyView()
        }
    }
}
#endif

#if os(macOS)
extension View {
    
    /// Adds VLM capabilities to analyze an NSImage with the given prompt
    /// - Parameters:
    ///   - service: The VLM service to use
    ///   - prompt: The prompt to use for analysis
    ///   - image: The NSImage to analyze
    ///   - onCompletion: Optional completion handler called with the generated text
    /// - Returns: A view with VLM capabilities
    @ViewBuilder
    public func vlm(
        service: VLMService,
        prompt: String,
        image: NSImage,
        onCompletion: ((String) -> Void)? = nil
    ) -> some View {
        if let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
            let ciImage = CIImage(cgImage: cgImage)
            
            vlm(
                service: service,
                prompt: prompt,
                image: ciImage,
                onCompletion: onCompletion
            )
        }
    }
}
#endif
