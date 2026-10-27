# Experimental rectangular low-rank GPU interface

The `feature/gpu-rectangular-schur` branch adds an opt-in GPU path for
standalone level-zero `c_bf_*` handles. It reuses the HODLR BACA, split,
TSQR and factor redistribution implementation. Activate with
`[dz]_c_bf_enable_gpu` after `c_bf_construct_init`; optionally register a
block evaluator before `c_bf_construct_element_compute`. Without an evaluator,
CPU compression is retained and factors are uploaded for GPU multiplies.
The normal BF API is unchanged unless explicitly activated.

The evaluator receives merged tree slots (rows `0..M-1`, columns
`M..M+N-1`), rather than the application IDs of a square HODLR evaluator.
Host copies of U/V remain valid for existing extraction. GPU-built device
factors are borrowed directly from their matrix-owned mirrors, avoiding a
second upload and factor allocation. GPU multiplies
support real/complex N/T/C and sum rank-by-RHS coefficients on the front's
borrowed communicator, using host staging initially.

`[dz]_c_bf_gpu_schur_mult(F12,F11,F21,...)` applies
`F21 inverse(F11) F12` or its plain transpose. Only its outer RHS/result
arrays cross the host API; intermediates stay in the shared backend heap.
It requires unsymmetric HODLR GPU factors and unit scaling, as used by
STRUMPACK. It does not implement nonzero butterfly levels, concurrent GPU
callers or a device-native randomized compressor.

The entry-image upload functions lease resident memory from the matrix's
backend heap. The lease is immutable and lives until matrix destruction.
Uploads complete before evaluator use. Applications must keep the callback
owner alive through native deletion and finish queued callbacks before
releasing borrowed data. Image memory competes with native factors; a
per-matrix byte bound is not an aggregate solver memory budget.

The paired STRUMPACK branch `feature/hodlr-gpu-rectangular-schur` provides
independent real/complex dense-reference tests, stride and ownership checks,
and one-/four-rank multiple-front sparse solves. Validation is pending;
results belong in STRUMPACK's HODLR_GPU_RECTANGULAR.json/rst record.
