Pinned metadata lifetime regression
===================================

metadata_lifetime.cpp checks small-image allocation size, geometric growth
while a CUDA upload may be pending, the uploaded data after builder destruction,
and complete release of per-instance pinned buffers and upload events. It uses
real CUDA calls and GNU linker wrapping for resource accounting.

With CUDA and MAGMA headers available, compile from the repository root
using an MPI C++ compiler wrapper (CC on Perlmutter)::

  mpicxx -std=c++17 -O2 -DH2_HAVE_GPU \
    -I GPU_BACKEND -I "$CUDA_ROOT/include" -I "$MAGMA_ROOT/include" \
    GPU_BACKEND/h2_gpu/tests/metadata_lifetime.cpp \
    -L "$CUDA_ROOT/lib64" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcudart \
    -Wl,--wrap=cudaMallocHost -Wl,--wrap=cudaFreeHost \
    -Wl,--wrap=cudaEventCreateWithFlags -Wl,--wrap=cudaEventDestroy \
    -o metadata_lifetime

Run metadata_lifetime on an allocated GPU, one process. It does not enter
MPI collectives. The fixed helper reports PASS with 393216 peak pinned bytes,
zero live pinned bytes and zero live upload events. Rebuilding the same test
against the original GPU_BACKEND include directory reproduces the excessive
initial allocation failure. The include path deliberately selects which
runtime header is tested; the test needs no ButterflyPACK shared-library link.

Integration evidence: STRUMPACK profiling job 60338444 also passes repeated
CPU/GPU sparse solves, real/complex multi-RHS inverse applications and fixed
two-ranks-per-GPU sharing with this metadata fix. Its successful CUDA trace
confirms that per-matrix pinned images are now released. Process-lifetime
pinned staging pools intentionally remain available for reuse.
