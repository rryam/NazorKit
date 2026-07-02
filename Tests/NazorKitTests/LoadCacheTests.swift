import Testing
@testable import NazorKit

private actor LoadHarness {
    private let cache = VLMService.LoadCache<Int>()
    private var loadCount = 0

    func load() async throws -> Int {
        try await cache.loadIfNeeded {
            await self.recordLoad()
            try await Task.sleep(for: .milliseconds(50))
            return 42
        }
    }

    func currentLoadCount() -> Int {
        loadCount
    }

    private func recordLoad() {
        loadCount += 1
    }
}

@Suite
struct LoadCacheTests {
    @Test
    func concurrentCallsShareInFlightLoad() async throws {
        let harness = LoadHarness()
        let results = try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0..<20 {
                group.addTask {
                    try await harness.load()
                }
            }

            var values: [Int] = []
            for try await value in group {
                values.append(value)
            }
            return values
        }

        #expect(Set(results) == Set([42]))
        #expect(await harness.currentLoadCount() == 1)

        let cachedValue = try await harness.load()

        #expect(cachedValue == 42)
        #expect(await harness.currentLoadCount() == 1)
    }
}
