# GDN Elementwise Fusion

`MCORE_GDN_FUSION=1` opts into fused Gated DeltaNet preparation and output
gating. Preparation combines causal convolution, SiLU, Q/K L2 normalization,
head expansion, layouts, and decay/write gates. Output gating combines GDN's
output RMSNorm with SiLU gating. The attention core and GQA layers are unchanged.

Preparation supports CUDA BF16 projections with batch size 1, 5,152 local
features, four key heads, sixteen value heads, head dimension 128, convolution
width 4, and context-parallel size 1 or 4. These are the local dimensions after
GDN's context-to-head all-to-all. For a model with sixteen key heads and
sixty-four value heads, TP=1/CP=4 has the same local dimensions as TP=4/CP=1.
TP=4/CP=4 has smaller local head counts and retains the unfused path.
The existing CP collectives and parameter-gradient reductions are unchanged.
Preparation requires FLA's convolution backward kernel. Unsupported shapes
and deterministic mode retain the existing path.

The option defaults to disabled. The original packed-sequence checks and
unfused convolution path are unchanged.

The kernels preserve intermediate BF16 rounding and additive Q/K L2 epsilon
of `1e-6`. They support first-order autograd only. Floating-point operation
ordering can differ from the unfused path; correctness tests do not establish
training convergence or bitwise equivalence to that path.
