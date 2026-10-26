// CUDA regression for per-matrix metadata: small allocations, growth,
// asynchronous upload completion and complete release of pinned data/events.
// Link with --wrap for cudaMallocHost/cudaFreeHost and event create/destroy.
#include "h2_gpu/gpu_runtime.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
std::unordered_map<void*, size_t> pinned;
size_t pinned_bytes = 0, peak_bytes = 0, calls = 0, events = 0;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}
extern "C" cudaError_t __real_cudaMallocHost(void**, size_t);
extern "C" cudaError_t __real_cudaFreeHost(void*);
extern "C" cudaError_t __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
extern "C" cudaError_t __real_cudaEventDestroy(cudaEvent_t);
extern "C" cudaError_t __wrap_cudaMallocHost(void** ptr, size_t bytes) {
    const auto status = __real_cudaMallocHost(ptr, bytes);
    if (status == cudaSuccess) {
        pinned[*ptr] = bytes;
        pinned_bytes += bytes;
        peak_bytes = std::max(peak_bytes, pinned_bytes);
        ++calls;
    }
    return status;
}
extern "C" cudaError_t __wrap_cudaFreeHost(void* ptr) {
    const auto status = __real_cudaFreeHost(ptr);
    const auto found = pinned.find(ptr);
    if (status == cudaSuccess && found != pinned.end()) {
        pinned_bytes -= found->second;
        pinned.erase(found);
    }
    return status;
}
extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    const auto status = __real_cudaEventCreateWithFlags(event, flags);
    if (status == cudaSuccess) ++events;
    return status;
}
extern "C" cudaError_t __wrap_cudaEventDestroy(cudaEvent_t event) {
    const auto status = __real_cudaEventDestroy(event);
    if (status == cudaSuccess) --events;
    return status;
}
int main() {
    using namespace fmm::gpu;
    try {
        check_cuda(cudaSetDevice(0), "device");
        cudaStream_t stream;
        check_cuda(cudaStreamCreate(&stream), "stream");
        for (int pass = 0; pass < 16; ++pass) {
            DeviceBuffer device;
            std::vector<std::uint64_t> expected(257), actual(expected.size());
            for (size_t i = 0; i < expected.size(); ++i) expected[i] = 1009 * pass + i;
            size_t offset = 0;
            char* uploaded = nullptr;
            {
                MetaBuilder image;
                offset = image.append(expected);
                require(pinned_bytes <= (size_t{1} << 20), "small image pins more than 1 MiB");
                image.upload(device, stream);
                image.alloc(256 * 1024); // grow while the previous upload may still read the host image
                uploaded = image.upload(device, stream);
            } // destructor must fence and release pinned storage and its event
            require(pinned.empty() && pinned_bytes == 0, "metadata destructor leaked pinned memory");
            require(events == 0, "metadata destructor leaked an upload event");
            check_cuda(cudaMemcpy(actual.data(), uploaded + offset, expected.size() * sizeof(expected[0]),
                                  cudaMemcpyDeviceToHost), "verify metadata after destruction");
            require(actual == expected, "growth/upload/destruction corrupted metadata");
        }
        check_cuda(cudaStreamDestroy(stream), "destroy stream");
        std::cout << "Metadata lifetime: PASS calls=" << calls << " peak_pinned_bytes=" << peak_bytes
                  << " live_pinned_bytes=" << pinned_bytes << " live_events=" << events << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Metadata lifetime: FAIL: " << error.what() << '\n';
        return 1;
    }
}
