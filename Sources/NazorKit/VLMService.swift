import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import MLXRandom

/// A service that handles Vision Language Model operations.
///
/// `VLMService` is an actor that provides thread-safe access to vision language models.
/// It manages model loading, caching, and generation operations. The service lazily loads
/// models on first use and reuses them for subsequent operations.
///
/// ```swift
/// let service = VLMService(
///     configuration: .qwen2VL2BInstruct4Bit,
///     generateParameters: .init(temperature: 0.8),
///     maxTokens: 1000
/// )
/// ```
public actor VLMService {
    /// The current state of the model
    private enum LoadState: Sendable {
        case idle
        case loaded(ModelContainer)
        case failed(Error)
    }

    // ModelStateManager and properties remain the same

    // Remove 'private actor' since VLMService itself is now an actor
    private final class ModelStateManager {
        private var state = LoadState.idle

        // ModelStateManager methods remain the same but don't need async/await anymore
        // since they're protected by the parent actor
        func getContainer() throws -> ModelContainer {
            switch state {
            case .idle:
                throw VLMError.modelNotLoaded
            case .loaded(let container):
                return container
            case .failed(let error):
                throw error
            }
        }

        func setLoaded(_ container: ModelContainer) {
            state = .loaded(container)
        }

        func setFailed(_ error: Error) {
            state = .failed(error)
        }

        func isIdle() -> Bool {
            if case .idle = state {
                return true
            }
            return false
        }
    }

    // Add VLMError enum
    private enum VLMError: Error {
        case modelNotLoaded
    }

    /// The configuration for the model
    private let configuration: ModelConfiguration

    /// The parameters for generation
    private let generateParameters: GenerateParameters

    /// The maximum number of tokens to generate
    private let maxTokens: Int

    /// The state manager
    private let stateManager = ModelStateManager()

    /// Creates a new VLM service with the specified configuration.
    ///
    /// - Parameters:
    ///   - configuration: The model configuration to use. Defaults to `qwen2VL2BInstruct4Bit`.
    ///   - generateParameters: Parameters for text generation, including temperature and sampling settings.
    ///     Defaults to temperature 0.6.
    ///   - maxTokens: The maximum number of tokens to generate in a single response. Defaults to 800.
    ///
    /// The service will automatically download and load the model on first use. The GPU cache limit
    /// is set to 20MB to optimize memory usage.
    public init(
        configuration: ModelConfiguration = VLMRegistry.qwen2VL2BInstruct4Bit,
        generateParameters: GenerateParameters = .init(temperature: 0.6),
        maxTokens: Int = 800
    ) {
        self.configuration = configuration
        self.generateParameters = generateParameters
        self.maxTokens = maxTokens

        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
    }

    /// Loads the model if it hasn't been loaded yet.
    ///
    /// This method is called automatically when generating text, so you typically don't need
    /// to call it directly. However, you can use it to preload the model and track download progress.
    ///
    /// - Parameter progressHandler: An optional closure called periodically with download progress.
    ///   The progress value ranges from 0.0 to 1.0.
    /// - Returns: The loaded model container.
    /// - Throws: An error if the model fails to load or download.
    ///
    /// - Note: This method is thread-safe and idempotent. If the model is already loaded,
    ///   it returns the existing container immediately without reloading.
    public func loadModelIfNeeded(
        progressHandler: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> ModelContainer {
        if stateManager.isIdle() {
            do {
                let modelContainer = try await VLMModelFactory.shared.loadContainer(
                    configuration: configuration,
                    progressHandler: progressHandler ?? { _ in }
                )
                stateManager.setLoaded(modelContainer)
                return modelContainer
            } catch {
                stateManager.setFailed(error)
                throw error
            }
        }
        return try stateManager.getContainer()
    }

    /// Generates a response for the given prompt and optional media.
    ///
    /// This method analyzes the provided image or video using the vision language model and generates
    /// a text response based on the prompt. The model is automatically loaded on first use.
    ///
    /// - Parameters:
    ///   - prompt: The text prompt describing what you want the model to analyze or describe.
    ///   - imageData: Optional JPEG/PNG image data to analyze. If provided, the image will be
    ///     processed and included in the model's context.
    ///   - video: Optional video URL to analyze. If provided, video frames will be processed
    ///     and included in the model's context. Note: Only one of `imageData` or `video` should be provided.
    ///   - updateHandler: An optional closure called on the main actor as tokens are generated.
    ///     This allows you to display streaming text updates in your UI.
    /// - Returns: A tuple containing:
    ///   - `output`: The complete generated text response.
    ///   - `tokensPerSecond`: The generation speed in tokens per second.
    /// - Throws: An error if the model fails to load, the media fails to process, or generation fails.
    ///
    /// - Note: This method is thread-safe and can be called concurrently from multiple tasks.
    ///   The service will manage model loading and state automatically.
    ///
    /// - Warning: Video processing is computationally intensive and may take longer than image analysis.
    public func generate(
        prompt: String,
        imageData: Data? = nil,
        video: URL? = nil,
        updateHandler: (@Sendable @MainActor (String) -> Void)? = nil
    ) async throws -> (output: String, tokensPerSecond: Double) {
        let modelContainer = try await loadModelIfNeeded()

        // Seed for reproducibility while allowing variation between generations
        MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

        // Capture necessary properties before the closure to avoid capturing self
        let params = generateParameters
        let maxTokenLimit = maxTokens

        let generateResult = try await modelContainer.perform { context in
            let images: [UserInput.Image]
            if let imageData = imageData,
               let ciImage = CIImage(data: imageData) {
                images = [.ciImage(ciImage)]
            } else {
                images = []
            }

            let videos: [UserInput.Video] = video.map { [.url($0)] } ?? []

            var userInput = UserInput(
                messages: [
                    [
                        "role": "user",
                        "content": [
                            ["type": "text", "text": prompt]
                        ]
                        + images.map { _ in ["type": "image"] }
                        + videos.map { _ in ["type": "video"] }
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
                parameters: params,
                context: context
            ) { tokens in
                if let handler = updateHandler {
                    let text = context.tokenizer.decode(tokens: tokens)
                    Task { @MainActor in
                        handler(text)
                    }
                }

                return tokens.count >= maxTokenLimit ? .stop : .more
            }
        }

        return (output: generateResult.output, tokensPerSecond: generateResult.tokensPerSecond)
    }
}
