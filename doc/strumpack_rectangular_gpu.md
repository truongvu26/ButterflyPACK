# Experimental rectangular low-rank GPU interface

The `feature/gpu-rectangular-schur` branch adds an opt-in GPU path for
standalone level-zero `c_bf_*` handles. Initialize device sharing/exchange
on the full solver communicator before activation or image upload. It reuses the HODLR BACA, split,
TSQR and factor redistribution implementation. Set `HODLR_use_gpu` to 1 or 2 on the BF options, then activate with
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
`F21 inverse(F11) F12` or its transpose/conjugate transpose. Only its outer RHS/result
arrays cross the host API; intermediates stay in the shared backend heap.
It requires unsymmetric HODLR GPU factors and unit scaling, as used by
STRUMPACK. Conjugation around the transposed pipeline stays on the device. It does
not implement nonzero butterfly levels, concurrent GPU
callers or a device-native randomized compressor.

The entry-image upload functions lease resident memory from the matrix's
backend heap. The lease is immutable and lives until matrix destruction.
Uploads complete before evaluator use. Applications must keep the callback
owner alive through native deletion and finish queued callbacks before
releasing borrowed data. Image memory competes with native factors; a
per-matrix byte bound is not an aggregate solver memory budget.

The paired STRUMPACK branch `feature/hodlr-gpu-rectangular-schur` validates
independent real/complex dense references, N/T/C products and Schur
applications, padded one/three RHS, moved owners, three-/four-rank process
groups and MPI_SELF work while other solver ranks wait. Jobs 60390291 and
60392134 pass the direct tests and six 40-cubed multiple-front sparse
variants. The evaluator kernel contract test reports zero Compute Sanitizer
memory errors (the native sparse solver is not run under the sanitizer).

Job 60390292 passes twelve 60-cubed, three-RHS trials on one allocation:
four ranks sharing two GPUs, three trials per variant. Median factor-plus-
solve is 46.523 s for the original hybrid and 31.190 s with bounded GPU
construction, versus 46.202 s with rectangular multiplies/device Schur alone
and 45.095 s on CPU. All trials use six refinement iterations. This is
one controlled regime; host preparation, per-block residency bounds and
large-front fallbacks remain. Exact source/binary provenance and results
belong in STRUMPACK's HODLR_GPU_RECTANGULAR.json/rst record.
