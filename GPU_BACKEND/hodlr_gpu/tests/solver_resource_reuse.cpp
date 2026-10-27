// Keep many real and complex front construction states alive simultaneously.
// Their cuSOLVER resources must remain bounded outside the reserved GPU heap.
// Run with two MPI ranks sharing a GPU and BPACK_GPU_HEAP_FRACTION=0.85.
#include <cuda_runtime.h>
#include <mpi.h>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

// Native C bridge entry points, as declared in SRC/BPACK_gpu_wrapper.cpp.
extern "C" {
void d_c_bpack_gpu_share_device(const int*, int*);
void d_c_bpack_gpu_create(void**);
void z_c_bpack_gpu_create(void**);
void d_c_bpack_gpu_delete(void*);
void z_c_bpack_gpu_delete(void*);
void d_c_bpack_hodlr_gpu_warm_up(void*, const int*, const int*, double*);
void z_c_bpack_hodlr_gpu_warm_up(void*, const int*, const int*, double*);
}

struct State {
    explicit State(bool complex) : complex(complex) {
        (complex ? z_c_bpack_gpu_create : d_c_bpack_gpu_create)(&value);
    }
    ~State() { (complex ? z_c_bpack_gpu_delete : d_c_bpack_gpu_delete)(value); }
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    void warm_up() {
        const int construct = 1, evaluate = 0;
        double seconds[2];
        (complex ? z_c_bpack_hodlr_gpu_warm_up : d_c_bpack_hodlr_gpu_warm_up)
            (value, &construct, &evaluate, seconds);
    }
    bool complex;
    void* value = nullptr;
};

size_t free_memory() {
    size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess)
        throw std::runtime_error("Cannot query GPU memory");
    return free;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    try {
        int rank = 0, sharing = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const int comm = MPI_Comm_c2f(MPI_COMM_WORLD);
        d_c_bpack_gpu_share_device(&comm, &sharing);
        State first_real(false), first_complex(true);
        first_real.warm_up(); first_complex.warm_up();
        MPI_Barrier(MPI_COMM_WORLD);
        const size_t before = free_memory();
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<std::unique_ptr<State>> real, complex;
        for (int i = 0; i < 64; ++i) {
            real.push_back(std::make_unique<State>(false)); real.back()->warm_up();
            complex.push_back(std::make_unique<State>(true)); complex.back()->warm_up();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const size_t after = free_memory();
        MPI_Barrier(MPI_COMM_WORLD);
        unsigned long long growth = before > after ? before-after : 0, maximum = 0;
        MPI_Allreduce(&growth, &maximum, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        std::printf("rank=%d sharing=%d live_front_states=130 external_memory_growth=%llu\n",
                    rank, sharing, growth);
        if (maximum > (256ull << 20))
            throw std::runtime_error("Per-front cuSOLVER resource growth exceeds 256 MiB");
        real.clear(); complex.clear();
        // Destroying other matrices must not invalidate the shared resource.
        first_complex.warm_up(); first_real.warm_up();
        if (!rank) std::puts("HODLR cuSOLVER resource reuse: PASS");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
