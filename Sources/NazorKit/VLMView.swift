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
    configuration: ModelConfiguration = ModelRegistry.qwen2VL2BInstruct4Bit,
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
      configuration: ModelConfiguration = ModelRegistry.qwen2VL2BInstruct4Bit,
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
        do {
          isGenerating = true
          let result = try await service.generate(
            prompt: prompt,
            image: image,
            video: video
          ) { text in
            generatedText = text
          }
          generatedText = result.output
          onCompletion?(result.output)
        } catch {
          self.error = error
          self.showError = true
        }
        isGenerating = false
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
    @MainActor
    public func vlm(
      service: VLMService,
      prompt: String,
      image: NSImage,
      onCompletion: ((String) -> Void)? = nil
    ) -> some View {
      if let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
        let ciImage = CIImage(cgImage: cgImage)
        return vlm(
          service: service,
          prompt: prompt,
          image: ciImage,
          onCompletion: onCompletion
        )
      }
      return self
    }
  }
#endif
