#pragma once
// Bounded cuSOLVER ownership for the HODLR backend's process-local queue.

#ifdef H2_HAVE_GPU

#include "h2_gpu/gpu_runtime.hpp"
#include <cusolverDn.h>
#include <mutex>
#include <stdexcept>
#include <string>

namespace bpack {
namespace gpu {

// Construction objects may live as long as their fronts, but their QR and SVD
// calls all use Context's single device/stream. Share one handle and default
// parameter object across matrices, scalar types and construction stages.
// A lease serializes host access through the complete buffer-size/execute
// sequence. Device work remains ordered on the same queue; releasing a lease
// requires no additional device synchronization. This does not make the rest
// of the backend (including its heap) safe for concurrent OpenMP construction.
class CuSolverLease {
public:
    CuSolverLease() : state_(shared_state()), lock_(state_.mutex) {
        auto& context = fmm::gpu::Context::instance();
        context.activate();
        if (state_.handle == nullptr) {
            cusolverDnHandle_t handle = nullptr;
            cusolverDnParams_t params = nullptr;
            check(cusolverDnCreate(&handle), "cusolverDnCreate");
            const auto status = cusolverDnCreateParams(&params);
            if (status != CUSOLVER_STATUS_SUCCESS) {
                cusolverDnDestroy(handle);
                check(status, "cusolverDnCreateParams");
            }
            state_.handle = handle;
            state_.params = params;
        }
        check(cusolverDnSetStream(state_.handle, context.stream()), "cusolverDnSetStream");
    }

    CuSolverLease(const CuSolverLease&) = delete;
    CuSolverLease& operator=(const CuSolverLease&) = delete;
    cusolverDnHandle_t handle() const { return state_.handle; }
    cusolverDnParams_t params() const { return state_.params; }

private:
    struct State {
        std::mutex mutex;
        cusolverDnHandle_t handle = nullptr;
        cusolverDnParams_t params = nullptr;
    };
    static State& shared_state() {
        // Like Context's MAGMA queue, keep the single resource until process
        // exit: static CUDA destruction can race runtime teardown. Front
        // destructors must neither destroy it nor change another front's state.
        static auto* state = new State;
        return *state;
    }
    static void check(cusolverStatus_t status, const char* what) {
        if (status != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error(std::string(what) + " failed with status " +
                                     std::to_string(static_cast<int>(status)));
    }
    State& state_;
    std::unique_lock<std::mutex> lock_;
};

}  // namespace gpu
}  // namespace bpack

#endif  // H2_HAVE_GPU
