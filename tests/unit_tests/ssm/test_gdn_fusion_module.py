# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDN module integration with optional output-norm recomputation."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import gdn_fusion
from megatron.core.ssm.gated_delta_net import gdn as gdn_module
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.fixture(params=[1, 4], ids=["cp1", "cp4"])
def model_parallel(request):
    cp_size = request.param
    if Utils.world_size % cp_size:
        pytest.skip(f"CP={cp_size} requires a world size divisible by {cp_size}")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp_size
    )
    try:
        yield cp_size
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
def test_gdn_fused_module(monkeypatch, model_parallel, packed, recompute, record_property):
    pytest.importorskip("transformer_engine.pytorch")
    tex = pytest.importorskip("transformer_engine_torch")
    if gdn_fusion._LINEAR_BWD is None:
        pytest.skip("FLA convolution backward is unavailable")
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(321)
    cp_size = model_parallel
    config = TransformerConfig(
        hidden_size=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=4 * cp_size,
        linear_num_value_heads=16 * cp_size,
        context_parallel_size=cp_size,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        num_attention_heads=4,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["gdn_norm_out"] if recompute else [],
    )
    spec = get_experimental_attention_variant_module_spec(config=config)
    groups = ProcessGroupCollection(
        tp=parallel_state.get_tensor_model_parallel_group(),
        cp=parallel_state.get_context_parallel_group(),
    )
    model = spec.module(
        config,
        submodules=spec.submodules,
        layer_number=1,
        conv_bias=cp_size > 1,
        pg_collection=groups,
    )
    model = model.cuda().bfloat16()
    assert model.feat_dim_split == (3072, 2048, 16, 16)
    if cp_size > 1:
        with torch.no_grad():
            for parameter in model.parameters():
                torch.distributed.broadcast(
                    parameter, src=torch.distributed.get_global_rank(groups.cp, 0), group=groups.cp
                )
        reference_model = (
            spec.module(
                replace(config, context_parallel_size=1),
                submodules=spec.submodules,
                layer_number=1,
                conv_bias=True,
                pg_collection=ProcessGroupCollection(tp=groups.tp, cp=groups.tp),
            )
            .cuda()
            .bfloat16()
        )
        reference_model.load_state_dict(model.state_dict())

    length = 257 if cp_size == 1 else 264
    torch.manual_seed(4321)
    hidden_full = torch.randn(length, 1, 256, device="cuda", dtype=torch.bfloat16)
    dy_full = torch.randn_like(hidden_full)
    boundaries = [0, 1, 128, 257] if cp_size == 1 else [0, 8, 32, 264]
    cu = torch.tensor(boundaries if packed else [0, length], device="cuda", dtype=torch.int32)
    indices = (
        tex.thd_get_partitioned_indices(cu, length, cp_size, groups.cp.rank())
        if cp_size > 1
        else torch.arange(length, device="cuda")
    )
    hidden = hidden_full.index_select(0, indices)
    dy = dy_full.index_select(0, indices)
    max_seqlen = max(b - a for a, b in zip(boundaries, boundaries[1:]))
    metadata = (
        PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        if packed
        else None
    )

    def run(module, inputs, output_grad, fused):
        monkeypatch.setenv("MCORE_GDN_FUSION", str(int(fused)))
        module.zero_grad(set_to_none=True)
        x = inputs.detach().clone().requires_grad_()
        out, _ = module(x, None, packed_seq_params=metadata)
        out.backward(output_grad)
        grads = {name: p.grad.detach().clone() for name, p in module.named_parameters()}
        grads["input"] = x.grad.detach().clone()
        return out.detach(), grads

    expected, reference_grads = run(model, hidden, dy, False)
    with (
        patch.object(gdn_module, "fused_prepare", wraps=gdn_module.fused_prepare) as prepare,
        patch.object(gdn_module, "fused_gated_norm", wraps=gdn_module.fused_gated_norm) as norm,
    ):
        actual, grads = run(model, hidden, dy, True)
        assert prepare.call_count == 1
        assert norm.call_count == (2 if recompute else 1)
    assert actual.shape == expected.shape
    assert grads.keys() == reference_grads.keys()
    comparisons = [
        ("output", actual, expected, 0.02),
        *((name, grads[name], reference_grads[name], 0.04) for name in grads),
    ]
    if cp_size > 1:
        full_output, full_grads = run(reference_model, hidden_full, dy_full, False)
        comparisons.extend(
            [
                ("full_output", actual, full_output.index_select(0, indices), 0.02),
                ("full_input", grads["input"], full_grads["input"].index_select(0, indices), 0.04),
            ]
        )
        # Replicated parameters receive the usual CP gradient sum outside the GDN module.
        for name in dict(model.named_parameters()):
            reduced = grads[name].float()
            torch.distributed.all_reduce(reduced, group=groups.cp)
            comparisons.append((f"full_{name}", reduced, full_grads[name], 0.04))

    for name, got, ref, tolerance in comparisons:
        assert torch.isfinite(got).all(), name
        assert torch.isfinite(ref).all(), name
        relative_l2 = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        record_property(f"relative_l2_{name}", relative_l2.item())
        assert relative_l2 < tolerance, name
