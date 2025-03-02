import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import MLXRandom

/// A service that handles Vision Language Model operations
public actor VLMService {
    // ModelStateManager and properties remain the same
    
    // Remove 'private actor' since VLMService itself is now an actor
    private final class ModelStateManager {
        /// The current state of the model
        private enum LoadState: Sendable {
            case idle
            case loaded(ModelContainer)
            case failed(Error)
        }
        
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
    
    // Keep the init the same
    public init(
        configuration: ModelConfiguration = ModelRegistry.qwen2VL2BInstruct4Bit,
        generateParameters: GenerateParameters = .init(temperature: 0.6),
        maxTokens: Int = 800
    ) {
        self.configuration = configuration
        self.generateParameters = generateParameters
        self.maxTokens = maxTokens
        
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
    }
    
    /// Loads the model if it hasn't been loaded yet
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
    
    /// Generates a response for the given prompt and image
    /// - Parameters:
    ///   - prompt: The text prompt
    ///   - imageData: Optional image data to analyze
    ///   - video: Optional video URL to analyze
    ///   - updateHandler: Optional handler for receiving token updates
    /// - Returns: The generated response and performance metrics
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
