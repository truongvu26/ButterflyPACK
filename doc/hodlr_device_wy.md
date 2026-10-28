# Device compact-WY recompression

GPU HODLR recompression now applies the compact Householder representation
on the device, including blocks whose QR has zero Householder coefficients.
The previous zero-coefficient path downloaded Gram matrices, coefficients
and intermediate right-hand sides, formed the triangular factor with a
cubic host recurrence, applied it and uploaded the result. It appeared in
3.7--20.1% of leaf CPU samples in the validated 150-cubed STRUMPACK profile
(job 60425000). Those sample fractions are not wall-time fractions.

For `Q = I - Y T Y^H`, let `D = diag(tau)` and
`G = strict_upper(Y^H Y)`. Instead of forming `T`, solve

```
(I + D G) Z = D (Y1^H X).
```

For nonzero coefficients this is the row-scaled form of
`T^-1 = G + D^-1`. The polynomial identity `(I + D G) T = D` also holds
when coefficients are zero, so the new unit-upper solve covers those
cases without dividing by zero. Complex coefficients are used directly;
the Gram matrix already contains the conjugation.

One batched setup kernel builds the unit-upper matrices and scales the
right-hand sides in place after the existing GEMM. The existing MAGMA
batched TRSM applies the result. CUDA and MAGMA use the same context
stream; the existing completion fence precedes freeing QR and workspace
buffers. The implementation reuses the prior triangular-matrix workspace,
removes the host coefficient scan and preserves compression tolerances,
SVD selection, factor layout and the C/Fortran interfaces.

`hodlr_gpu_recompression_test --wy-only` compares against an independent
host `larft` recurrence using LAPACK-generated real/complex QR data, zero
columns, identity factors, complex phases, ill-conditioned inputs, tiny
coefficients, mixed batches and padded leading dimensions. The default
regression also runs these checks and the existing independent factor
reconstruction tests.

`--wy-benchmark` measures the isolated former host-fallback operation at
ranks 256, 512 and 1024 with two blocks and half-rank right-hand-side
counts. Both blocks contain zero coefficients. Four warmed trials use
alternating order. Legacy timing includes its required device/host
transfers; device timing includes setup, TRSM and the completion fence.
Both exclude input creation, shared metadata upload and reset copies.
These component ratios do not predict complete sparse-solver speedup.

STRUMPACK's `test/benchmark_hodlr_device_wy_perlmutter.sbatch` provides
the end-to-end check: two old-library and two new-library runs in
before/after/after/before order, using the same quiet executable, GPU
topology, tolerances and allocation. It reports speed ratios only after
all four solves pass numerical validation. Frozen binary hashes and the
exact native patch are retained with the run manifests.
