#pragma once
// SVD of one dense device matrix (cuSOLVER QR-based gesvd):
// the recompression of the HODLR blocks of large rank and the root
// of the merges' TSQR.

#ifdef H2_HAVE_GPU

#include "h2_gpu/device_heap.hpp"
#include "gpu_common/device_scalar.hpp"
#include "h2_gpu/gpu_runtime.hpp"

#include "hodlr_solver.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace bpack {
namespace gpu {

// An m x n identity (ones on the diagonal, leading dimension m) from the
// heap: the operand of the warm-ups
template<typename T>
T* heap_identity(int m, int n) {
    std::vector<T> host(static_cast<size_t>(m) * n, T(0.0));
    for (int i = 0; i < std::min(m, n); ++i) host[static_cast<size_t>(i) * m + i] = T(1.0);
    T* d = fmm::gpu::DeviceHeap::instance().alloc<T>(host.size() * sizeof(T));
    fmm::gpu::check_cuda(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "warm-up");
    return d;
}

template<typename T>
class DeviceSvd {
public:
    DeviceSvd() = default;
    DeviceSvd(const DeviceSvd&) = delete;
    DeviceSvd& operator=(const DeviceSvd&) = delete;

    // a (m x n, leading dimension m) = u diag(s) v^H with u (m x mn), s (mn,
    // descending), v (n x mn), mn = min(m, n), all on the device (a
    // destroyed). QR-based gesvd is the default: polar gesvdp can perturb
    // rank-deficient frontal blocks substantially despite info == 0.
    // BPACK_GPU_HODLR_SVD=polar retains the old path for controlled diagnosis;
    // its perturbation estimate is returned, QR returns zero.
    double svd_device(int m, int n, T* da, double* ds, T* du, T* dv) {
        const char* driver = std::getenv("BPACK_GPU_HODLR_SVD");
        if (driver && std::string(driver) == "polar") return svd_polar(m, n, da, ds, du, dv);
        if (driver && std::string(driver) != "qr")
            throw std::invalid_argument("BPACK_GPU_HODLR_SVD must be qr or polar");
        return svd_qr(m, n, da, ds, du, dv);
    }

private:
    static void transpose(int m, int n, const T* a, T* at, magma_queue_t queue) {
        if constexpr (std::is_same_v<T, double>) magmablas_dtranspose(m, n, a, m, at, n, queue);
        else magmablas_ztranspose_conj(m, n, reinterpret_cast<const magmaDoubleComplex*>(a), m,
                                      reinterpret_cast<magmaDoubleComplex*>(at), n, queue);
    }

    double svd_qr(int m, int n, T* da, double* ds, T* du, T* dv) {
        using fmm::gpu::check_cuda;
        auto& ctx = fmm::gpu::Context::instance();
        ctx.activate();
        auto& heap = fmm::gpu::DeviceHeap::instance();
        const auto stream = ctx.stream();
        if (m < n) {
            // gesvd requires a tall matrix. A^H = V S U^H swaps the two
            // output factors and preserves the public U,S,V convention.
            T* adjoint = heap.alloc<T>(static_cast<size_t>(m) * n * sizeof(T));
            transpose(m, n, da, adjoint, ctx.queue());
            svd_qr(n, m, adjoint, ds, dv, du);
            heap.free(adjoint);
            return 0.0;
        }
        CuSolverLease solver;
        const cudaDataType type = fmm::gpu::is_complex_scalar<T> ? CUDA_C_64F : CUDA_R_64F;
        // gesvd returns V^H, whereas the backend consumes V.
        T* vt = heap.alloc<T>(static_cast<size_t>(n) * n * sizeof(T));
        int* dinfo = heap.alloc<int>(sizeof(int));
        size_t dbytes = 0, hbytes = 0;
        check(cusolverDnXgesvd_bufferSize(solver.handle(), solver.params(), 'S', 'S', m, n, type, da, m,
                                          CUDA_R_64F, ds, type, du, m, type, vt, n, type, &dbytes, &hbytes),
              "cusolverDnXgesvd_bufferSize");
        char* dwork = heap.alloc<char>(std::max<size_t>(dbytes, 1));
        std::vector<char> hwork(std::max<size_t>(hbytes, 1));
        check(cusolverDnXgesvd(solver.handle(), solver.params(), 'S', 'S', m, n, type, da, m, CUDA_R_64F, ds,
                               type, du, m, type, vt, n, type, dwork, dbytes, hwork.data(), hbytes, dinfo),
              "cusolverDnXgesvd");
        transpose(n, n, vt, dv, ctx.queue());
        int info = 0;
        check_cuda(cudaMemcpyAsync(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost, stream), "SVD info");
        check_cuda(cudaStreamSynchronize(stream), "QR-based SVD");
        heap.free(dwork);
        heap.free(dinfo);
        heap.free(vt);
        if (info != 0) throw std::runtime_error("cusolverDnXgesvd failed, info " + std::to_string(info));
        return 0.0;
    }

    double svd_polar(int m, int n, T* da, double* ds, T* du, T* dv) {
        using fmm::gpu::check_cuda;
        fmm::gpu::Context& ctx = fmm::gpu::Context::instance();
        ctx.activate();
        const cudaStream_t stream = ctx.stream();
        CuSolverLease solver;
        const cudaDataType type = fmm::gpu::is_complex_scalar<T> ? CUDA_C_64F : CUDA_R_64F;
        fmm::gpu::DeviceHeap& heap = fmm::gpu::DeviceHeap::instance();
        int* dinfo = heap.alloc<int>(sizeof(int));
        size_t dbytes = 0, hbytes = 0;
        check(cusolverDnXgesvdp_bufferSize(solver.handle(), solver.params(), CUSOLVER_EIG_MODE_VECTOR, 1, m, n, type, da, m,
                                           CUDA_R_64F, ds, type, du, m, type, dv, n, type, &dbytes, &hbytes),
              "cusolverDnXgesvdp_bufferSize");
        char* dwork = heap.alloc<char>(std::max<size_t>(dbytes, 1));
        std::vector<char> hwork(std::max<size_t>(hbytes, 1));
        double err_sigma = 0.0;
        check(cusolverDnXgesvdp(solver.handle(), solver.params(), CUSOLVER_EIG_MODE_VECTOR, 1, m, n, type, da, m, CUDA_R_64F, ds, type,
                                du, m, type, dv, n, type, dwork, dbytes, hwork.data(), hbytes, dinfo, &err_sigma),
              "cusolverDnXgesvdp");
        int info = 0;
        check_cuda(cudaMemcpyAsync(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost, stream), "SVD info");
        check_cuda(cudaStreamSynchronize(stream), "SVD");
        heap.free(dwork);
        heap.free(dinfo);
        if (info != 0) throw std::runtime_error("cusolverDnXgesvdp failed, info " + std::to_string(info));
        return err_sigma;
    }

public:
    // Before the first SVD (GpuState::warm_up): the cuSOLVER handle and one
    // small SVD, so that no level pays their first use
    void warm_up() {
        constexpr int n = 8;
        fmm::gpu::DeviceHeap& heap = fmm::gpu::DeviceHeap::instance();
        T* a = heap_identity<T>(n, n);
        double* s = heap.alloc<double>(n * sizeof(double));
        T* u = heap.alloc<T>(n * n * sizeof(T));
        T* v = heap.alloc<T>(n * n * sizeof(T));
        svd_device(n, n, a, s, u, v);  // (synchronizes)
        for (void* p : {static_cast<void*>(v), static_cast<void*>(u), static_cast<void*>(s), static_cast<void*>(a)}) {
            heap.free(p);
        }
    }

private:
    static void check(cusolverStatus_t status, const char* what) {
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(what) + " failed with status " + std::to_string(static_cast<int>(status)));
        }
    }

};

}  // namespace gpu
}  // namespace bpack

#endif  // H2_HAVE_GPU
