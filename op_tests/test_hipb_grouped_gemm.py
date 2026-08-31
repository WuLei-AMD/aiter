# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.gradlib import hipb_grouped_mm, hipb_multistream_mm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _offsets(counts, device="cuda"):
    return torch.tensor(
        [0, *torch.tensor(counts).cumsum(0).tolist()],
        dtype=torch.int64,
        device=device,
    )


@pytest.mark.parametrize("counts", ([7, 0, 13], [1, 19, 4]))
def test_hipb_grouped_forward_matches_torch(counts):
    torch.manual_seed(17)
    dtype = torch.float16
    total, experts, k, n = sum(counts), len(counts), 64, 96
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b = torch.randn(experts, k, n, device="cuda", dtype=dtype)
    out = torch.empty(total, n, device="cuda", dtype=dtype)

    hipb_grouped_mm(a, b.transpose(1, 2).contiguous(), _offsets(counts), out)

    expected = torch.cat(
        [chunk @ b[index] for index, chunk in enumerate(torch.split(a, counts))]
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("offsets_device", ("cpu", "cuda"))
def test_hipb_multistream_wgrad_matches_torch(offsets_device):
    torch.manual_seed(23)
    dtype = torch.bfloat16
    counts = [5, 0, 11]
    total, experts, k, n = sum(counts), len(counts), 64, 80
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b = torch.randn(total, n, device="cuda", dtype=dtype)
    out = torch.full((experts, k, n), 7.0, device="cuda", dtype=dtype)

    hipb_multistream_mm(a, b, _offsets(counts, offsets_device), out, True)

    expected = torch.stack(
        [
            a_chunk.T @ b_chunk
            if count
            else torch.zeros(k, n, device="cuda", dtype=dtype)
            for count, a_chunk, b_chunk in zip(
                counts, torch.split(a, counts), torch.split(b, counts)
            )
        ]
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("input_dtype", "output_dtype"),
    (
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ),
)
@pytest.mark.parametrize("offsets_device", ("cpu", "cuda"))
def test_hipb_multistream_forward_matches_torch(
    input_dtype, output_dtype, offsets_device
):
    torch.manual_seed(29)
    counts = [7, 0, 13]
    total, experts, k, n = sum(counts), len(counts), 64, 96
    a = torch.randn(total, k, device="cuda", dtype=input_dtype)
    b = torch.randn(experts, k, n, device="cuda", dtype=input_dtype)
    out = torch.empty(total, n, device="cuda", dtype=output_dtype)

    hipb_multistream_mm(a, b, _offsets(counts, offsets_device), out)

    expected = torch.cat([
        chunk.to(output_dtype) @ b[index].to(output_dtype)
        for index, chunk in enumerate(torch.split(a, counts))
    ])
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("offsets_device", ("cpu", "cuda"))
def test_hipb_multistream_transposed_weight_matches_torch(offsets_device):
    torch.manual_seed(31)
    dtype = torch.bfloat16
    counts = [9, 0, 15]
    total, experts, k, n = sum(counts), len(counts), 64, 80
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b_transposed = torch.randn(experts, n, k, device="cuda", dtype=dtype)
    out = torch.empty(total, n, device="cuda", dtype=dtype)

    hipb_multistream_mm(
        a,
        b_transposed,
        _offsets(counts, offsets_device),
        out,
        False,
        None,
        True,
    )

    expected = torch.cat(
        [
            chunk @ b_transposed[index].T
            for index, chunk in enumerate(torch.split(a, counts))
        ]
    )
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
