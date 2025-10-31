import CoreImage
import MLXLMCommon
import MLXVLM
import SwiftUI

/// A property wrapper that provides access to a VLM service for use in SwiftUI views.
///
/// Use `@VLMServiceProvider` to create and manage a `VLMService` instance in your SwiftUI views.
/// The service is automatically created and managed by SwiftUI's lifecycle.
///
/// ```swift
/// struct ContentView: View {
///     @VLMServiceProvider private var vlmService
///     // Use vlmService.wrappedValue to access the service
/// }
/// ```
///
/// To customize the service configuration:
///
/// ```swift
/// @VLMServiceProvider(
///     configuration: .qwen2VL2BInstruct4Bit,
///     generateParameters: .init(temperature: 0.8),
///     maxTokens: 1000
/// ) private var vlmService
/// ```
@propertyWrapper
@MainActor
public struct VLMServiceProvider: DynamicProperty {
    @StateObject private var serviceHolder: ServiceHolder

    /// The underlying VLM service instance.
    ///
    /// Access this property to use the service for generating text or loading models.
    /// The service is an actor, so all methods must be called with `await`.
    public var wrappedValue: VLMService {
        serviceHolder.service
    }

    /// Creates a VLM service provider with the specified configuration.
    ///
    /// - Parameters:
    ///   - configuration: The model configuration to use. Defaults to `qwen2VL2BInstruct4Bit`.
    ///   - generateParameters: Parameters for text generation. Defaults to temperature 0.6.
    ///   - maxTokens: The maximum number of tokens to generate. Defaults to 800.
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

/// A view modifier that adds VLM capabilities to analyze images or videos.
///
/// This modifier automatically triggers analysis when the view appears and handles loading states,
/// errors, and completion callbacks. It displays a progress indicator while generating and shows
/// error alerts if generation fails.
///
/// - Note: This modifier is typically used through the `analyzeMedia` view extension rather than
///   directly.
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

    /// Creates a view modifier that analyzes media with the given service.
    ///
    /// - Parameters:
    ///   - service: The VLM service to use for analysis.
    ///   - prompt: The text prompt describing what to analyze or describe.
    ///   - image: Optional CIImage to analyze.
    ///   - video: Optional video URL to analyze. Only one of `image` or `video` should be provided.
    ///   - onCompletion: Optional closure called with the generated text when analysis completes.
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
                await performAnalysis()
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

    private func performAnalysis() async {
        let localService = service

        // Create a copy of the update handler that's isolated to the main actor
        @MainActor func updateText(_ text: String) {
            generatedText = text
        }

        // Convert image to Data before passing to non-isolated context
        let imageData = convertImageToData(image)

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

    private func convertImageToData(_ image: CIImage?) -> Data? {
        guard let image = image else { return nil }

        let context = CIContext()
        guard let cgImage = context.createCGImage(image, from: image.extent) else {
            return nil
        }

        // Create a temporary bitmap context to draw the image
        let width = cgImage.width
        let height = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue

        guard let bitmapContext = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            return nil
        }

        // Draw the image into the context
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        bitmapContext.draw(cgImage, in: rect)

        // Get the data from the context
        guard let data = bitmapContext.makeImage() else {
            return nil
        }

        // Convert to JPEG data
        return context.jpegRepresentation(
            of: CIImage(cgImage: data),
            colorSpace: colorSpace
        )
    }
}

extension View {

    /// Adds VLM capabilities to analyze an image or video with the given prompt.
    ///
    /// This modifier automatically triggers analysis when the view appears and handles the
    /// generation process, including loading states and error handling. The analysis runs asynchronously
    /// and displays a progress indicator while generating.
    ///
    /// - Parameters:
    ///   - service: The VLM service to use for analysis. Typically obtained from `@VLMServiceProvider`.
    ///   - prompt: The text prompt describing what you want the model to analyze or describe.
    ///   - image: Optional `CIImage` to analyze. If provided, the image will be processed and analyzed.
    ///   - video: Optional video URL to analyze. If provided, video frames will be processed.
    ///     Only one of `image` or `video` should be provided.
    ///   - onCompletion: Optional closure called on the main actor with the generated text when
    ///     analysis completes successfully.
    /// - Returns: A modified view that automatically analyzes the provided media.
    ///
    /// - Note: The analysis starts automatically when the view appears. If you need to trigger
    ///   analysis manually, use `VLMService.generate()` directly instead.
    ///
    /// ```swift
    /// Image(uiImage: image)
    ///     .analyzeMedia(
    ///         service: vlmService,
    ///         prompt: "Describe this image",
    ///         image: ciImage
    ///     ) { description in
    ///         print(description)
    ///     }
    /// ```
    @MainActor
    public func analyzeMedia(
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

    /// Adds VLM capabilities to analyze a UIImage with the given prompt.
    ///
    /// This is a convenience method for iOS and visionOS that converts a `UIImage` to a `CIImage`
    /// and analyzes it. If the conversion fails, an empty view is returned.
    ///
    /// - Parameters:
    ///   - service: The VLM service to use for analysis.
    ///   - prompt: The text prompt describing what to analyze.
    ///   - image: The `UIImage` to analyze.
    ///   - onCompletion: Optional closure called with the generated text when analysis completes.
    /// - Returns: A view with VLM capabilities, or `EmptyView` if the image conversion fails.
    ///
    /// - Note: Available on iOS and visionOS only.
    @ViewBuilder
    public func analyzeMedia(
        service: VLMService,
        prompt: String,
        image: UIImage,
        onCompletion: ((String) -> Void)? = nil
    ) -> some View {
        if let ciImage = CIImage(image: image) {
            analyzeMedia(
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

    /// Adds VLM capabilities to analyze an NSImage with the given prompt.
    ///
    /// This is a convenience method for macOS that converts an `NSImage` to a `CIImage`
    /// and analyzes it. If the conversion fails, an empty view is returned.
    ///
    /// - Parameters:
    ///   - service: The VLM service to use for analysis.
    ///   - prompt: The text prompt describing what to analyze.
    ///   - image: The `NSImage` to analyze.
    ///   - onCompletion: Optional closure called with the generated text when analysis completes.
    /// - Returns: A view with VLM capabilities, or `EmptyView` if the image conversion fails.
    ///
    /// - Note: Available on macOS only.
    @ViewBuilder
    public func analyzeMedia(
        service: VLMService,
        prompt: String,
        image: NSImage,
        onCompletion: ((String) -> Void)? = nil
    ) -> some View {
        if let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
            let ciImage = CIImage(cgImage: cgImage)

            analyzeMedia(
                service: service,
                prompt: prompt,
                image: ciImage,
                onCompletion: onCompletion
            )
        }
    }
}
#endif
