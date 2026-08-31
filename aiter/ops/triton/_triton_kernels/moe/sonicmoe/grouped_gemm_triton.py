import logging
import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_grouped_gemm_dw_config,
    get_grouped_gemm_fwd_config,
    split_launch_config,
)


logger = logging.getLogger(__name__)
_LOGGED_BACKENDS: set[str] = set()
_MULTISTREAM_CALLS = 0
_HOST_CU_SEQLENS_CACHE_MAX_ENTRIES = 4096
_HOST_CU_SEQLENS_CACHE: OrderedDict[tuple[int, int], torch.Tensor] = OrderedDict()


def _cu_seqlens_cache_key(cu_seqlens: torch.Tensor) -> tuple[int, int]:
    device_index = cu_seqlens.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return device_index, cu_seqlens.data_ptr()


def register_host_cu_seqlens(
    cu_seqlens: torch.Tensor, host_cu_seqlens: torch.Tensor
) -> None:
    """Associate GPU offsets with dispatcher-produced CPU offsets.

    The multi-stream backend launches one hipBLASLt GEMM per expert and therefore
    needs offsets on the host. Keeping the host copy produced at the Megatron
    dispatcher boundary avoids synchronously copying the same GPU tensor before
    every forward and backward GEMM.
    """
    if cu_seqlens.device.type != "cuda":
        raise ValueError("cu_seqlens cache keys must be GPU tensors")
    host_cu_seqlens = host_cu_seqlens.to(
        device="cpu", dtype=torch.int64, copy=False
    ).contiguous()
    key = _cu_seqlens_cache_key(cu_seqlens)
    _HOST_CU_SEQLENS_CACHE[key] = host_cu_seqlens
    _HOST_CU_SEQLENS_CACHE.move_to_end(key)
    while len(_HOST_CU_SEQLENS_CACHE) > _HOST_CU_SEQLENS_CACHE_MAX_ENTRIES:
        _HOST_CU_SEQLENS_CACHE.popitem(last=False)


def _registered_host_cu_seqlens(
    cu_seqlens: torch.Tensor,
) -> torch.Tensor | None:
    if cu_seqlens.device.type != "cuda":
        return cu_seqlens
    key = _cu_seqlens_cache_key(cu_seqlens)
    host_cu_seqlens = _HOST_CU_SEQLENS_CACHE.get(key)
    if host_cu_seqlens is not None:
        _HOST_CU_SEQLENS_CACHE.move_to_end(key)
    return host_cu_seqlens


def clear_registered_host_cu_seqlens(cu_seqlens: torch.Tensor) -> None:
    """Discard a stale host-offset entry for a newly allocated GPU tensor."""
    if cu_seqlens.device.type == "cuda":
        _HOST_CU_SEQLENS_CACHE.pop(_cu_seqlens_cache_key(cu_seqlens), None)


def _log_backend_once(backend: str) -> None:
    if os.environ.get("SONIC_MOE_LOG_BACKEND", "0") != "1":
        return
    if backend not in _LOGGED_BACKENDS:
        logger.warning("Sonic grouped GEMM active backend: %s", backend)
        _LOGGED_BACKENDS.add(backend)


def _local_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


_QWEN3_FWD_CONFIGS = {
    (1536, 2048, 16, True): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    (2048, 768, 16, False): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    (768, 2048, 16, False): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    (2048, 1536, 16, False): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
}

_QWEN3_DW_CONFIGS = {
    (1536, 2048, 16, True): {
        "BLOCK_K": 128,
        "BLOCK_N": 128,
        "BLOCK_T": 64,
        "num_warps": 4,
        "num_stages": 2,
    },
    (2048, 768, 16, False): {
        "BLOCK_K": 128,
        "BLOCK_N": 128,
        "BLOCK_T": 32,
        "num_warps": 4,
        "num_stages": 2,
    },
}


def _use_qwen3_tuned_configs() -> bool:
    return os.environ.get("SONIC_MOE_USE_QWEN3_TUNED_GEMM", "0") == "1"


def _get_fwd_autotune_configs():
    configs = []
    for BLOCK_M in [32, 64, 128]:
        for BLOCK_N in [32, 64, 128]:
            for BLOCK_K in [32, 64]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 4]:
                        if BLOCK_M * BLOCK_N <= 16384 and BLOCK_M * BLOCK_K <= 8192:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "BLOCK_K": BLOCK_K,
                                    },
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                )
                            )
    return configs


def _prune_fwd_configs(configs, nargs, **kw):
    K = kw.get("K", nargs.get("K", 9999))
    N = kw.get("N", nargs.get("N", 9999))
    pruned = []
    for c in configs:
        bk = c.kwargs["BLOCK_K"]
        bn = c.kwargs["BLOCK_N"]
        if bk <= triton.next_power_of_2(K) and bn <= triton.next_power_of_2(N):
            pruned.append(c)
    return pruned if pruned else configs


@triton.jit
def _grouped_gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    cu_seqlens_ptr,
    bias_ptr,
    A_idx_ptr,
    scatter_idx_ptr,
    stride_ak,
    stride_am,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias_e,
    stride_bias_n,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_GATHER_IDX: tl.constexpr,
    HAS_SCATTER_IDX: tl.constexpr,
):
    pid = tl.program_id(0)

    cumulative_blocks = 0
    expert_id = 0
    expert_start = 0
    expert_end = 0

    for e in range(E):
        s = tl.load(cu_seqlens_ptr + e).to(tl.int32)
        f = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
        m_e = f - s
        blocks_m_e = tl.cdiv(m_e, BLOCK_M)
        blocks_this_expert = blocks_m_e * tl.cdiv(N, BLOCK_N)
        if pid >= cumulative_blocks and pid < cumulative_blocks + blocks_this_expert:
            expert_id = e
            expert_start = s
            expert_end = f
        cumulative_blocks += blocks_this_expert

    # Launching an upper bound avoids copying cu_seqlens to the CPU just to
    # calculate the exact grid size.
    if pid >= cumulative_blocks:
        return

    local_pid = pid
    for e in range(E):
        if e < expert_id:
            s = tl.load(cu_seqlens_ptr + e).to(tl.int32)
            f = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
            m_e = f - s
            local_pid -= tl.cdiv(m_e, BLOCK_M) * tl.cdiv(N, BLOCK_N)

    M_expert = expert_end - expert_start
    num_pid_m = tl.cdiv(M_expert, BLOCK_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = local_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (local_pid % num_pid_in_group) % group_size_m
    pid_n = (local_pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M_expert
    global_m = expert_start + offs_m

    if HAS_GATHER_IDX:
        a_row_idx = tl.load(A_idx_ptr + global_m, mask=m_mask, other=0).to(tl.int64)
    else:
        a_row_idx = global_m.to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    expert_id_i64 = expert_id.to(tl.int64)
    a_dtype = A_ptr.dtype.element_ty

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K
        a = tl.load(
            A_ptr
            + a_row_idx[:, None] * stride_ak
            + k_offs[None, :].to(tl.int64) * stride_am,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(a_dtype)
        b = tl.load(
            B_ptr
            + expert_id_i64 * stride_be
            + k_offs[:, None].to(tl.int64) * stride_bk
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        ).to(a_dtype)
        acc += tl.dot(a, b)

    if HAS_BIAS:
        bias_vals = tl.load(
            bias_ptr
            + expert_id_i64 * stride_bias_e
            + offs_n.to(tl.int64) * stride_bias_n,
            mask=offs_n < N,
            other=0.0,
        )
        acc += bias_vals[None, :]

    c = acc.to(C_ptr.dtype.element_ty)

    if HAS_SCATTER_IDX:
        c_row_idx = tl.load(scatter_idx_ptr + global_m, mask=m_mask, other=0).to(
            tl.int64
        )
    else:
        c_row_idx = global_m.to(tl.int64)

    c_ptrs = (
        C_ptr
        + c_row_idx[:, None] * stride_cm
        + offs_n[None, :].to(tl.int64) * stride_cn
    )
    c_mask = m_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


_grouped_gemm_kernel_autotuned = triton.autotune(
    configs=_get_fwd_autotune_configs(),
    key=["N", "K", "E"],
    prune_configs_by={"early_config_prune": _prune_fwd_configs},
)(_grouped_gemm_kernel)


def _get_dw_autotune_configs():
    configs = []
    for BLOCK_K in [32, 64, 128]:
        for BLOCK_N in [32, 64, 128]:
            for BLOCK_T in [16, 32, 64]:
                for num_warps in [4, 8]:
                    if BLOCK_K * BLOCK_N <= 16384 and BLOCK_T * BLOCK_K <= 8192:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_K": BLOCK_K,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_T": BLOCK_T,
                                },
                                num_warps=num_warps,
                                num_stages=2,
                            )
                        )
    return configs


def _prune_dw_configs(configs, nargs, **kw):
    K = kw.get("K", nargs.get("K", 9999))
    N = kw.get("N", nargs.get("N", 9999))
    pruned = []
    for c in configs:
        bk = c.kwargs["BLOCK_K"]
        bn = c.kwargs["BLOCK_N"]
        if bk <= triton.next_power_of_2(K) and bn <= triton.next_power_of_2(N):
            pruned.append(c)
    return pruned if pruned else configs


@triton.jit
def _grouped_gemm_dw_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    cu_seqlens_ptr,
    A_idx_ptr,
    stride_ak,
    stride_am,
    stride_bm,
    stride_bn,
    stride_ce,
    stride_ck,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_GATHER_IDX: tl.constexpr,
):
    pid = tl.program_id(0)
    num_k_blocks: tl.constexpr = tl.cdiv(K, BLOCK_K)
    num_n_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)
    blocks_per_expert: tl.constexpr = num_k_blocks * num_n_blocks

    expert_id = pid // blocks_per_expert
    local_pid = pid % blocks_per_expert
    pid_k = local_pid // num_n_blocks
    pid_n = local_pid % num_n_blocks

    expert_start = tl.load(cu_seqlens_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(cu_seqlens_ptr + expert_id + 1).to(tl.int32)
    M_expert = expert_end - expert_start

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)

    k_mask = offs_k < K
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    a_dtype = A_ptr.dtype.element_ty

    for t_start in range(0, M_expert, BLOCK_T):
        t_offs = t_start + offs_t
        t_mask = t_offs < M_expert
        global_t = expert_start + t_offs

        if HAS_GATHER_IDX:
            a_row_idx = tl.load(A_idx_ptr + global_t, mask=t_mask, other=0).to(tl.int64)
        else:
            a_row_idx = global_t.to(tl.int64)

        a = tl.load(
            A_ptr
            + a_row_idx[:, None] * stride_ak
            + offs_k[None, :].to(tl.int64) * stride_am,
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(a_dtype)

        b = tl.load(
            B_ptr
            + global_t[:, None].to(tl.int64) * stride_bm
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=t_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(a_dtype)

        acc += tl.dot(tl.trans(a), b)

    c = acc.to(C_ptr.dtype.element_ty)
    expert_id_i64 = expert_id.to(tl.int64)
    c_ptrs = (
        C_ptr
        + expert_id_i64 * stride_ce
        + offs_k[:, None].to(tl.int64) * stride_ck
        + offs_n[None, :].to(tl.int64) * stride_cn
    )
    c_mask = k_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


_grouped_gemm_dw_kernel_autotuned = triton.autotune(
    configs=_get_dw_autotune_configs(),
    key=["N", "K", "E"],
    prune_configs_by={"early_config_prune": _prune_dw_configs},
)(_grouped_gemm_dw_kernel)


def _compute_grid_fwd(cu_seqlens_cpu, N, E, BLOCK_M, BLOCK_N):
    total_blocks = 0
    for e in range(E):
        m_e = cu_seqlens_cpu[e + 1].item() - cu_seqlens_cpu[e].item()
        total_blocks += triton.cdiv(m_e, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    return total_blocks


def grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
    B_is_transposed: bool = False,
):
    backend = os.environ.get("SONIC_MOE_GROUPED_GEMM_BACKEND", "triton").lower()
    if backend not in {"triton", "hipblaslt", "multistream", "auto"}:
        raise ValueError(
            "SONIC_MOE_GROUPED_GEMM_BACKEND must be triton, hipblaslt, "
            "multistream, or auto"
        )
    if backend == "triton":
        triton_b = B.transpose(1, 2) if B_is_transposed else B
        return _grouped_gemm_triton(
            A, triton_b, cu_seqlens, out, bias, A_idx, scatter_idx, A_is_transposed
        )
    if backend == "multistream":
        return _grouped_gemm_multistream(
            A,
            B,
            cu_seqlens,
            out,
            bias,
            A_idx,
            scatter_idx,
            A_is_transposed,
            B_is_transposed,
        )

    try:
        return _grouped_gemm_hipblaslt(
            A,
            B,
            cu_seqlens,
            out,
            bias,
            A_idx,
            scatter_idx,
            A_is_transposed,
            B_is_transposed,
        )
    except (RuntimeError, ValueError):
        if backend == "hipblaslt":
            raise
        try:
            return _grouped_gemm_multistream(
                A,
                B,
                cu_seqlens,
                out,
                bias,
                A_idx,
                scatter_idx,
                A_is_transposed,
                B_is_transposed,
            )
        except (RuntimeError, ValueError):
            triton_b = B.transpose(1, 2) if B_is_transposed else B
            return _grouped_gemm_triton(
                A,
                triton_b,
                cu_seqlens,
                out,
                bias,
                A_idx,
                scatter_idx,
                A_is_transposed,
            )


def _grouped_gemm_hipblaslt(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    bias: torch.Tensor | None,
    A_idx: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
    A_is_transposed: bool,
    B_is_transposed: bool,
):
    from aiter.ops.gradlib import hipb_grouped_mm

    A = _local_tensor(A)
    B = _local_tensor(B)
    bias = _local_tensor(bias)
    out = _local_tensor(out)
    work_a = A.index_select(0, A_idx) if A_idx is not None else A
    work_a = work_a.contiguous()
    work_b = (
        B.contiguous()
        if A_is_transposed or B_is_transposed
        else B.transpose(1, 2).contiguous()
    )
    counts = cu_seqlens.contiguous()

    if A_is_transposed:
        if scatter_idx is not None:
            raise ValueError("scatter_idx is invalid for a grouped wgrad")
        E = counts.numel() - 1
        shape = (E, work_a.shape[1], work_b.shape[1])
    else:
        shape = (work_a.shape[0], work_b.shape[1])

    direct_out = (
        out is not None
        and out.is_contiguous()
        and scatter_idx is None
        and tuple(out.shape) == shape
    )
    work_out = out if direct_out else torch.empty(shape, dtype=A.dtype, device=A.device)
    if A_is_transposed:
        work_out.zero_()

    hipb_grouped_mm(
        work_a,
        work_b,
        counts,
        work_out,
        A_is_transposed,
        bias.contiguous() if bias is not None else None,
    )

    if out is None:
        if scatter_idx is None:
            return work_out
        out = torch.empty_like(work_out)
    if scatter_idx is not None:
        out.index_copy_(0, scatter_idx, work_out)
    elif work_out is not out:
        out.copy_(work_out)
    return out


def _grouped_gemm_multistream(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    bias: torch.Tensor | None,
    A_idx: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
    A_is_transposed: bool,
    B_is_transposed: bool,
):
    global _MULTISTREAM_CALLS

    from aiter.ops.gradlib import hipb_multistream_mm

    _log_backend_once("hipblaslt_multistream")
    A = _local_tensor(A)
    B = _local_tensor(B)
    bias = _local_tensor(bias)
    out = _local_tensor(out)
    work_a = A.index_select(0, A_idx) if A_idx is not None else A
    work_a = work_a.contiguous()
    work_b = B.contiguous()
    host_cu_seqlens = _registered_host_cu_seqlens(cu_seqlens)
    counts = (
        host_cu_seqlens
        if host_cu_seqlens is not None
        else cu_seqlens.contiguous()
    )
    if A_is_transposed:
        if scatter_idx is not None:
            raise ValueError("scatter_idx is invalid for a grouped wgrad")
        shape = (counts.numel() - 1, work_a.shape[1], work_b.shape[1])
    else:
        shape = (
            work_a.shape[0],
            work_b.shape[1] if B_is_transposed else work_b.shape[2],
        )

    input_dtype = torch.promote_types(work_a.dtype, work_b.dtype)
    if A_is_transposed and out is not None:
        input_dtype = torch.promote_types(input_dtype, out.dtype)
    work_a = work_a.to(dtype=input_dtype)
    work_b = work_b.to(dtype=input_dtype)
    direct_out = (
        out is not None
        and out.is_contiguous()
        and out.dtype == input_dtype
        and scatter_idx is None
        and tuple(out.shape) == shape
    )
    work_out = (
        out
        if direct_out
        else torch.empty(shape, dtype=input_dtype, device=A.device)
    )
    _MULTISTREAM_CALLS += 1
    local_rank = os.environ.get("LOCAL_RANK", "0")
    trace = os.environ.get("SONIC_MOE_TRACE_GEMM", "0") == "1" and (
        local_rank == "0" or os.environ.get("SONIC_MOE_TRACE_ALL_RANKS", "0") == "1"
    )
    if trace:
        expert_rows = (counts[1:] - counts[:-1]).cpu().tolist()
        logger.warning(
            "Sonic multi-stream rank=%s call %d start: A=%s B=%s out=%s "
            "rows=%s wgrad=%s",
            local_rank,
            _MULTISTREAM_CALLS,
            tuple(work_a.shape),
            tuple(work_b.shape),
            tuple(work_out.shape),
            expert_rows,
            A_is_transposed,
        )
    hipb_multistream_mm(
        work_a,
        work_b,
        counts,
        work_out,
        A_is_transposed,
        bias.to(dtype=input_dtype).contiguous() if bias is not None else None,
        B_is_transposed,
    )
    if trace:
        logger.warning(
            "Sonic multi-stream rank=%s call %d done", local_rank, _MULTISTREAM_CALLS
        )

    if out is None:
        if scatter_idx is None:
            return work_out
        out = torch.empty_like(work_out)
    if scatter_idx is not None:
        out.index_copy_(0, scatter_idx, work_out)
    elif work_out is not out:
        out.copy_(work_out)
    return out


def _grouped_gemm_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
):
    if A_is_transposed and B.dim() == 2:
        return _grouped_gemm_dw(A, B, cu_seqlens, out, A_idx)

    E = B.shape[0]
    K_dim = B.shape[1]
    N = B.shape[2]

    TK = A.shape[0] if A_idx is None else A_idx.numel()

    if out is None:
        out = torch.empty(TK, N, dtype=A.dtype, device=A.device)

    def grid(META):
        max_m_blocks = triton.cdiv(TK, META["BLOCK_M"]) + E - 1
        return (max_m_blocks * triton.cdiv(N, META["BLOCK_N"]),)

    launch_args = (
        A, B, out,
        cu_seqlens,
        bias if bias is not None else A,
        A_idx if A_idx is not None else cu_seqlens,
        scatter_idx if scatter_idx is not None else cu_seqlens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        out.stride(0),
        out.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
    )
    launch_meta = {
        "N": N,
        "K": K_dim,
        "E": E,
        "GROUP_SIZE_M": 8,
        "HAS_BIAS": (bias is not None),
        "HAS_GATHER_IDX": (A_idx is not None),
        "HAS_SCATTER_IDX": (scatter_idx is not None),
    }
    fixed = _QWEN3_FWD_CONFIGS.get((N, K_dim, E, A_idx is not None))
    fwd_cfg = get_grouped_gemm_fwd_config(N, K_dim, E)
    if _use_qwen3_tuned_configs() and fixed is not None:
        _grouped_gemm_kernel[grid](*launch_args, **launch_meta, **fixed)
    elif fwd_cfg is not None:
        constexprs, launch = split_launch_config(fwd_cfg)
        _grouped_gemm_kernel[grid](
            *launch_args, **launch_meta, **constexprs, **launch
        )
    else:
        _grouped_gemm_kernel_autotuned[grid](*launch_args, **launch_meta)
    return out


def _grouped_gemm_dw(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    A_idx: torch.Tensor | None,
):
    K_dim = A.shape[1]
    N = B.shape[1]
    E = cu_seqlens.shape[0] - 1

    if out is None:
        out = torch.empty(E, K_dim, N, dtype=A.dtype, device=A.device)

    def grid(META):
        num_k_blocks = triton.cdiv(K_dim, META["BLOCK_K"])
        num_n_blocks = triton.cdiv(N, META["BLOCK_N"])
        return (E * num_k_blocks * num_n_blocks,)

    launch_args = (
        A, B, out,
        cu_seqlens,
        A_idx if A_idx is not None else cu_seqlens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
    )
    launch_meta = {
        "N": N,
        "K": K_dim,
        "E": E,
        "HAS_GATHER_IDX": A_idx is not None,
    }
    fixed = _QWEN3_DW_CONFIGS.get((N, K_dim, E, A_idx is not None))
    dw_cfg = get_grouped_gemm_dw_config(N, K_dim, E)
    if _use_qwen3_tuned_configs() and fixed is not None:
        _grouped_gemm_dw_kernel[grid](*launch_args, **launch_meta, **fixed)
    elif dw_cfg is not None:
        constexprs, launch = split_launch_config(dw_cfg)
        _grouped_gemm_dw_kernel[grid](
            *launch_args, **launch_meta, **constexprs, **launch
        )
    else:
        _grouped_gemm_dw_kernel_autotuned[grid](*launch_args, **launch_meta)
    return out
