import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import MLXRandom

/// A service that handles Vision Language Model operations
public final class VLMService: @unchecked Sendable {
  /// The current state of the model
  private enum LoadState: @unchecked Sendable {
    case idle
    case loaded(ModelContainer)
    case failed(Error)
  }

  /// The current state of the model loading
  private var loadState = LoadState.idle

  /// The configuration for the model
  private let configuration: ModelConfiguration

  /// The parameters for generation
  private let generateParameters: GenerateParameters

  /// The maximum number of tokens to generate
  private let maxTokens: Int

  /// Creates a new VLM service
  /// - Parameters:
  ///   - configuration: The model configuration to use
  ///   - generateParameters: The parameters for generation
  ///   - maxTokens: The maximum number of tokens to generate
  public init(
    configuration: ModelConfiguration = ModelRegistry.qwen2VL2BInstruct4Bit,
    generateParameters: GenerateParameters = .init(temperature: 0.6),
    maxTokens: Int = 800
  ) {
    self.configuration = configuration
    self.generateParameters = generateParameters
    self.maxTokens = maxTokens

    // Set a reasonable cache limit
    MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
  }

  /// Loads the model if it hasn't been loaded yet
  /// - Parameter progressHandler: A closure that receives loading progress updates
  /// - Returns: The loaded model container
  public func loadModelIfNeeded(
    progressHandler: ((Progress) -> Void)? = nil
  ) async throws -> ModelContainer {
    switch loadState {
    case .idle:
      do {
        let modelContainer = try await VLMModelFactory.shared.loadContainer(
          configuration: configuration,
          progressHandler: progressHandler
        )
        loadState = .loaded(modelContainer)
        return modelContainer
      } catch {
        loadState = .failed(error)
        throw error
      }

    case .loaded(let container):
      return container

    case .failed(let error):
      throw error
    }
  }

  /// Generates a response for the given prompt and image
  /// - Parameters:
  ///   - prompt: The text prompt
  ///   - image: Optional image to analyze
  ///   - video: Optional video URL to analyze
  ///   - updateHandler: Optional handler for receiving token updates
  /// - Returns: The generated response and performance metrics
  public func generate(
    prompt: String,
    image: CIImage? = nil,
    video: URL? = nil,
    updateHandler: ((String) -> Void)? = nil
  ) async throws -> (output: String, tokensPerSecond: Double) {
    let modelContainer = try await loadModelIfNeeded()

    // Seed for reproducibility while allowing variation between generations
    MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

    return try await modelContainer.perform { context in
      let images: [UserInput.Image] = image.map { [.ciImage($0)] } ?? []
      let videos: [UserInput.Video] = video.map { [.url($0)] } ?? []

      var userInput = UserInput(
        messages: [
          [
            "role": "user",
            "content": [
              ["type": "text", "text": prompt]
            ]
              + images.map { _ in ["type": "image"] }
              + videos.map { _ in ["type": "video"] },
          ]
        ],
        images: images,
        videos: videos
      )

      // Set reasonable processing parameters
      userInput.processing.resize = .init(width: 448, height: 448)

      let input = try await context.processor.prepare(input: userInput)

      return try MLXLMCommon.generate(
        input: input,
        parameters: generateParameters,
        context: context
      ) { tokens in
        if let updateHandler = updateHandler {
          let text = context.tokenizer.decode(tokens: tokens)
          Task { @MainActor in
            updateHandler(text)
          }
        }

        return tokens.count >= maxTokens ? .stop : .more
      }
    }
  }
}
