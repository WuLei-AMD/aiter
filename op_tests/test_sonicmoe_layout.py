# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.sonicmoe import (
    SonicMoEActivationType,
    moe_general_routing_inputs,
    moe_pre_routed_inputs,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def test_general_routing_grouped_weights_match_legacy_layout():
    torch.manual_seed(43)
    device = torch.device("cuda")
    tokens, hidden, intermediate, experts = 64, 64, 64, 2
    token_indices = torch.arange(tokens, dtype=torch.int32, device=device)
    expert_indices = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32, device=device),
        torch.tensor([32, 32], dtype=torch.int32, device=device),
    )

    x_grouped = torch.randn(
        tokens, hidden, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    x_legacy = x_grouped.detach().clone().requires_grad_(True)
    scores_grouped = torch.rand(
        tokens, dtype=torch.float32, device=device, requires_grad=True
    )
    scores_legacy = scores_grouped.detach().clone().requires_grad_(True)
    w1_grouped = torch.randn(
        experts,
        hidden,
        2 * intermediate,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w2_grouped = torch.randn(
        experts,
        intermediate,
        hidden,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w1_legacy = w1_grouped.detach().permute(2, 1, 0).contiguous().requires_grad_(True)
    w2_legacy = w2_grouped.detach().permute(2, 1, 0).contiguous().requires_grad_(True)

    common = (
        token_indices,
        expert_indices,
        None,
        experts,
        torch.cuda.current_stream().cuda_stream,
        SonicMoEActivationType.SWIGLU,
        False,
        True,
    )
    output_grouped, _ = moe_general_routing_inputs(
        x_grouped,
        scores_grouped,
        common[0],
        common[1],
        w1_grouped,
        common[2],
        w2_grouped,
        common[2],
        *common[3:],
        grouped_weight_layout=True,
    )
    output_legacy, _ = moe_general_routing_inputs(
        x_legacy,
        scores_legacy,
        common[0],
        common[1],
        w1_legacy,
        common[2],
        w2_legacy,
        common[2],
        *common[3:],
    )

    torch.testing.assert_close(output_grouped, output_legacy)
    grad = torch.randn_like(output_grouped)
    output_grouped.backward(grad)
    output_legacy.backward(grad)
    torch.testing.assert_close(x_grouped.grad, x_legacy.grad)
    torch.testing.assert_close(scores_grouped.grad, scores_legacy.grad)
    torch.testing.assert_close(w1_grouped.grad, w1_legacy.grad.permute(2, 1, 0))
    torch.testing.assert_close(w2_grouped.grad, w2_legacy.grad.permute(2, 1, 0))


def test_pre_routed_fused_swiglu_matches_separate_kernel(monkeypatch):
    torch.manual_seed(47)
    device = torch.device("cuda")
    counts = torch.tensor([17, 0, 31], dtype=torch.int32, device=device)
    tokens, hidden, intermediate, experts = int(counts.sum()), 64, 64, 3

    x_ref = torch.randn(
        tokens, hidden, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    x_fused = x_ref.detach().clone().requires_grad_(True)
    scores = torch.ones(tokens, dtype=torch.float32, device=device)
    w1_ref = torch.randn(
        experts,
        hidden,
        2 * intermediate,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w2_ref = torch.randn(
        experts,
        intermediate,
        hidden,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
    )
    w1_fused = w1_ref.detach().clone().requires_grad_(True)
    w2_fused = w2_ref.detach().clone().requires_grad_(True)

    def run(x, w1, w2, enabled):
        monkeypatch.setenv("LUMEN_EXPERT_FUSED_SWIGLU", enabled)
        return moe_pre_routed_inputs(
            x,
            scores,
            counts,
            w1,
            None,
            w2,
            None,
            torch.cuda.current_stream().cuda_stream,
            SonicMoEActivationType.SWIGLU,
        )[0]

    output_ref = run(x_ref, w1_ref, w2_ref, "0")
    output_fused = run(x_fused, w1_fused, w2_fused, "1")
    torch.testing.assert_close(output_fused, output_ref, rtol=0, atol=0)

    grad = torch.randn_like(output_ref)
    output_ref.backward(grad)
    output_fused.backward(grad)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(w1_fused.grad, w1_ref.grad, rtol=0, atol=0)
    torch.testing.assert_close(w2_fused.grad, w2_ref.grad, rtol=0, atol=0)
