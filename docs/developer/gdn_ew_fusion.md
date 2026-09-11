# Experimental GDN Elementwise Fusion

`MCORE_GDN_FUSION=1` enables a shape-specialized GatedDeltaNet optimization
bundle. It defaults to off. The selected attention-core call is unchanged;
no cuDNN, FLA, Transformer Engine, PyTorch, or Triton package sources are modified.

## Scope

- Fuse width-four convolution, SiLU, native-head Q/K L2 normalization,
  grouped-head expansion, layout preparation, and gate preparation.
- Fold repeated-head gradients before normalization backward, reuse FLA's
  existing linear convolution backward with tuned tiles, and write the packed
  projection gradient directly.
- Fuse the external RMSNorm and SiLU output gate in both directions, accepting
  a strided projection view instead of a separate gate copy.
- Cache packed-metadata validation using weak references and tensor mutation
  versions. Inference tensors are always revalidated; changed metadata misses
  the cache. A single complete sequence bypasses chunk-index preparation.

Preparation supports CUDA BF16, batch one, projection width 5152, key/value
head dimension 128, four native Q/K heads and sixteen value heads per rank,
CP1, width-four SiLU convolution, and nondeterministic mode. Other shapes use
the existing path. Output fusion also requires the supported RMSNorm layout.
The FLA backward kernel entry point must be available for preparation fusion.
Higher-order gradients are not supported by the custom autograd functions.

`MCORE_GDN_COMMON_OPT=1` independently enables the same metadata cache,
single-complete-sequence indexing shortcut, and unfused convolution backward
launch settings. For a matched comparison, keep this set in both modes and
toggle only `MCORE_GDN_FUSION`. Both selectors default to zero. The existing
FLA backward kernel uses a time tile of 128 positions, a channel tile of 32,
and four warps per program. These settings are held equal across modes, not
claimed to be optimal for every path: an unfused SiLU-only backward sweep
preferred a time tile of 64. No FLA package source is modified.

Revision 4 keeps `exp(A_log)` and its gradient reduction in FP32, casting only
the final parameter gradient. This matches the compiled GDN baseline and
fixes a newly covered all-zero-input case that exceeded the original gradient
tolerance in revision 3. The fix does not change convolution launch settings.

Revision 5 retains that correction and uses two warps for the new fusion
kernels, with time tiles 8/16 for preparation forward/backward and 4/16 for
output-normalization forward/backward. Compiler FMA is enabled in those four
kernels. The existing FLA backward launch and the separate projection-gate
backward kernel are unchanged. The original correctness tolerances still apply.

## Measurement and Attribution

The tested image used a four-GB300, four-layer, 65536-token policy-training
proxy, TP4/CP1/PP1/EP1, micro/global batch one, and seed 1234. The denominator
was the complete `policy.train()` call, including forward, backward,
communication, and optimizer work. Logprob refresh was outside the timer in
both modes. End-to-end measurements were not profiled.

### Matched Shared-Optimization Control

Revision 4's completed same-process comparison used 100 steps: 20 warmup and
80 measured, counterbalanced in twenty ABBA/BAAB blocks. Keeping common
optimizations on in both modes measured 1.366342979 s control versus
1.356713612 s fused: **0.704755% lower complete-step time**. The block-bootstrap
95% interval was [0.501353%, 0.893634%], so its lower bound only narrowly clears
0.5%. Fresh-process ABBA measured 1.387404302 s control versus 1.383501628 s
fused, only 0.281293% lower, including all four trials and the predetermined
steady steps 11-30. Evidence is therefore mixed across protocols; the 0.5%
target is not reliably established by both tests. Independent paired
confirmation on another node measured 1.368127667 s control versus
1.359384321 s fused, a 0.639074% reduction with a 95% block-bootstrap interval
of [0.480549%, 0.804522%]. This supports a positive gain but does not settle the
exact >0.5% threshold.

Installed-image correctness passed all five cases on all four GPUs, including
the full 65536-token shape and the formerly failing all-zero case, without
relaxing tolerances. In that run, preparation microbenchmarks measured
623.133004/2631.301999 us control forward/backward versus
334.882006/1172.696009 us fused (46.258%/55.433% reductions). These are kernel
boundary timings, not complete training-phase times. The microbenchmark
compiled its reference across correctness shapes before timing, so its
absolute numbers should not be compared directly with the earlier suite.

The packaged branch sources also passed all 21 focused unit tests in a second
nightly-derived GB300 image, including the CUDA preparation and output-norm
tests. The five-case preparation suite passed on all four GPUs in that image.
The revision-4 full-model GRPO run completed ten steps with real rollout
generation. Against an unpaired stock nightly, policy-training time was
0.151715% higher over steps 1-10 and 0.399927% lower over the predeclared
steps 2-10 window. Full GRPO step time was 0.264540% and 0.010623% higher,
respectively. The stock comparison includes the common optimizations and
stochastic asynchronous rollout differences; it does not isolate fusion.
This run does not establish a greater-than-0.5% full-model improvement.

### Revision 5 Kernel Tuning

A four-GPU sweep against installed revision 4 measured mean per-rank median
preparation forward/backward times of 335.013/1172.387 us versus
299.056/1147.586 us for the selected revision-5 settings. Output normalization
forward/backward measured 146.084/390.618 us versus 136.410/376.060 us.
Reference launches bracketed each sweep; all four GPUs improved for each
selected region. These are exploratory kernel measurements, not complete
policy-step speedups. Selection from a sweep requires independent confirmation.

The combined revision-5 candidate subsequently passed all 21 focused tests
and the five-case preparation suite on all four GPUs in the nightly-derived
image, without changing tolerances. Revision-5 proxy and full-workload timing
are not yet available.

### Historical Bundle Result

Fresh-process ABBA, thirty steps per run, using all predetermined steady steps
11-30, measured 1.391328200 s baseline versus 1.383018440 s enabled:
**0.597254% lower step time**. The first A/B pair alone improved only 0.124939%;
run-level variability is material. Two independent counterbalanced
same-process tests measured 0.987297% and 0.795397% reductions.

These percentages describe the **whole optimization bundle**, not GPU fusion
alone. Metadata caching, the single-sequence fast path, and convolution launch
tuning were enabled with the fused kernels. No cache-disabled or factorial
ablation has isolated their individual contributions. The ABBA launcher was
configured to change only the selector, using the same pinned image, allocation,
workload, seed, driver, and timer. A retrospective comparison of all four saved
environment snapshots confirmed 150 of 156 exports identical. Only the fusion
selector and five timestamp/logging variables differed. This excludes an
additional workload-setting difference in those snapshots; it does not turn
the historical bundle result into a fusion-only result.

Installed-image microbenchmarks measured preparation forward/backward
reductions of 45.576%/47.195%, and output-normalization forward/backward
reductions of 44.425%/12.750%. Separate training profiles measured summed
preparation-kernel reductions of 44.524% forward and 47.009% backward; these
kernel sums are not end-to-end timing estimates.

The branch contains the revision-5 kernel settings, corrected gate math, and
shared control. Validation is tolerance-based, not bitwise identity. No
revision-5 convergence, full-model speedup, or MLPerf accuracy claim is made.

## Tests

Focused tests are in `tests/unit_tests/ssm/test_gdn_fusion.py` and
`tests/unit_tests/ssm/test_gdn_packed_sequence.py`. Run in the normal MCore
GPU test environment:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/ssm/test_gdn_fusion.py \
  tests/unit_tests/ssm/test_gdn_packed_sequence.py
```

The preparation tests compare both optimized control and fusion to the compiled
stock reference, covering packed boundaries, strided projections, zero and
near-zero values, optional bias, BF16/FP32 gate parameters, and first-order
gradients. Output tests cover strided gates and zero-centered normalization.
Metadata tests cover mutation, aliases, cache lifetime/bounds, and inference
tensors. The original installed-image suite also validated the full 65536-token
shape on all four GPUs.
