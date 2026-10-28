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

Validation completed with 168 independent real/complex WY checks, 288
recompression comparisons, 12 direct SVD checks and 24 captured-factor
comparisons (component job 60465148). Its later sparse launch failed
before solver startup; corrected sparse job 60466018 passed 40 and
100 cubed with three RHS. Component success is recorded separately from
the original job's failed overall status.

The complete 150-cubed sparse comparison, job 60467720, passed all four
before/after/after/before trials on 32 ranks and 16 A100 40-GB GPUs.
Median factor time decreased from 380.0015 to 347.048 seconds; median
factor-plus-solve decreased from 384.256025 to 351.038845 seconds,
a 1.09463x total speedup. The executable, tolerances, GPU allocation and
sequential BLAS were identical; only the native library changed.
Source revision `ab40497` corresponds to native library SHA256
`66ebd85ba461d64bc840711661ad71ae400ec93fcea2677dcaa360b6be569bae`.
The two new factor times, 346.975 and 347.121 seconds, agree closely,
but this remains a small sample with randomized compression work.

STRUMPACK's `doc/HODLR_GPU_DEVICE_WY.{json,rst}` records the full evidence,
separate BLAS-worker/image-budget tests and the supplied external timing
comparison. No full device frontal-entry evaluator or input-image lifetime
change is included here.
