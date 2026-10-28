# GPU HODLR SVD accuracy and experimental BACA certification

The assembled 1368-by-1368 STRUMPACK front captured in job 60420335 has
localized off-diagonal Schur couplings. Its prepared image agrees exactly
with STRUMPACK's child-inclusive extraction callback. Replaying that image
in job 60420593 reproduces intermittent approximation errors on both CPU
and GPU with BACA variants 4 and 5. Multiplication and extraction of the
compressed representation agree to about 1e-15. Independent QR/SVD/WY
recompression tests passed (job 60419636).

This evidence identifies an inaccurate BACA approximation, consistent with
false convergence when sampled panels miss a localized residual. It does
not establish that every previously observed large-solver failure has the
same cause. The replay uses `knn=0`; the original sparse capture used graph
hints.

`BPACK_GPU_BACA_RESIDUAL_GUARD=1` enables an experimental correction in the
native GPU construction path, including rectangular low-rank blocks:

1. At apparent convergence, evaluate every block in column stripes and
   compare its complete relative Frobenius residual against
   `sqrt(min(m,n))*tol_comp`, the Frobenius budget corresponding to the
   relative singular-value cutoff. `tol_comp` itself is unchanged.
2. Restart blocks exceeding the bound from the column with the largest
   residual norm, retaining their existing factors (at most two repairs).
3. Require rank progress between repairs and enforce the dimension/rank
   limits. If BACA stalls or saturates, replace its factors by an exact
   representation, `I*A` or `A*I` according to the smaller dimension.
   Reevaluate that representation and report a runtime error if it fails
   certification; this check remains enabled in Release builds.
4. Recompress with the existing QR/SVD path, adopt its actual output, and
   certify again. The post-recompression bound is
   `2*sqrt(min(m,n))*tol_comp`, allowing for the Frobenius error of a
   relative singular-value cutoff. Bad output gets one exact-factor
   fallback and recompression attempt. A second failure aborts explicitly.

Stripe width is at most 16 and at most the smallest local panel width.
The check reuses existing device panel buffers and a host buffer of at
most `max_local_rows * stripe_width` scalars. It allocates no additional
full matrix image. The exact fallback can grow the factors to full rank
and therefore use O(m*n) storage. For collective host-panel evaluators, the front
communicator agrees on stripe width and maximum column count; ranks
without local blocks still join each evaluation. The evaluator remains
responsible for the original matrix entries and all child contributions.

This is a correctness experiment, disabled by default pending validation
and timing. `BPACK_GPU_BACA_RESIDUAL_LOG=1` reports each certification
sweep's maximum error and wall time; its time is included in panel
construction statistics. A complete residual sweep evaluates O(m*n) entries and
performs O(m*n*rank) arithmetic per block. Host-panel callbacks can make
that expensive. CPU BACA is unchanged. This option is not a claim that
the general BACA sampling problem or production GPU accuracy is solved.

The first experimental sparse run (60421036) timed out because a no-progress
assertion was compiled out in Release. Stripe logging in 60421207 exposed
repeated certification of a saturated 200-by-800 block with error 2.90e-4.
Those runs are failures, not evidence of a CUDA or MPI deadlock. Runtime
handling and the exact-factor fallback address this experiment's loop.

Job 60421594 passed 176 independent recompression comparisons and the
captured-front replays, but the 40-cubed sparse case failed after some
experimental BACA repairs: pre-recompression residuals passed while final
front checks failed (sample errors 1.00519 and 0.0101637). This motivates
checking the final factors as well. It is not a claim that the independent
recompression tests reproduced those actual factors or that every prior
failure has the same cause.


A direct SVD reproducer (60422299) isolated a concrete polar-SVD failure on
an exactly rank-deficient 684-square frontal block. The reference norm was
2.26377982924; `cusolverDnXgesvdp` returned info=0 and perturbation estimate
0.50266009782, while independent reconstruction had relative error
5.80721996296. Callers discarded that estimate. NVIDIA documents that
polar SVD perturbs inputs near zero singular values:
<https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgesvdp>.

The large-block and TSQR-root SVD now defaults to QR-based
`cusolverDnXgesvd`. It converts the returned V^H into the backend's V
convention, including complex and rectangular inputs. The legacy path is
available only for controlled comparison with `BPACK_GPU_HODLR_SVD=polar`;
`qr` is the default. Small batched MAGMA SVD is unchanged. New direct-SVD
and recompression tests cover exactly rank-deficient square input and both
orientations of real/complex rectangular input. Captured-factor job 60422461 passed: direct SVD reconstruction errors
were 4.60e-15 and 3.90e-15 on the 684- and 580-square inputs. Job 60422460
passed 288 independent recompression comparisons, 12 direct SVD checks,
the captured-front replays, and a full 40-cubed three-RHS sparse solve
(error 5.29e-12, scaled residual 1.44e-11). Its strict-guard 100-cubed run
timed out at six minutes; that case is not a PASS. The norm-budget
revision is being checked in job 60422977; its 40-cubed solve passed
(error 1.01e-11, scaled residual 1.51e-10), with 100-cubed pending.
The final Frobenius acceptance bound is unchanged. No updated 150-cubed
accuracy or speed claim is made yet.
