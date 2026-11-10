# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Runtime implementation registry — who is actually serving each op.

# The catalog (hcu_config.BACKEND_CATALOG) declares what CAN be selected;
# this module records what IS installed right now, per catalog op:
# which call-site got rebound, into what, and by which install mechanism.
# It is the data side of the runtime dump (dump()/log_summary()) and the
# arbiter for the L0 static rules that forbid duplicate / unregistered
# patches -- see docs/governance.md.

# Roles (mirrors governance §2):
#   active  — serves production traffic; exactly one per catalog entry
#   ref     — framework-free reference implementation for numerics triage;
#             NEVER routed into production by a bridge
# Bridges may only bind `active` implementations; binding a `ref` impl is
# an L0 violation (tests/tools has _test_ scripts that exercise refs
# directly instead).

from __future__ import annotations

import json
import logging
import os
import sys
import types

LOG = logging.getLogger("hcu.adapter")

# mechanism -- how the binding landed (governance §3 ladder, top = preferred)
MECH_SEAM = "seam"          # official upstream hook (--quantization, backend registry)
MECH_REBIND = "rebind"      # module-attribute swap of a pure-python entry point
MECH_TEXT = "text-patch"    # source text replace + recompile (assert relax) — allowlist

# roles
ROLE_ACTIVE = "active"
ROLE_REF = "ref"

# op -> record; op keys ARE catalog keys (l0_gate cross-checks this).
_RECORDS = {}
# target symbol -> op, for duplicate-bind detection at install time
_BOUND = {}

# text-patch allowlist: exact (op, needle) pairs permitted. Anything else
# attempting MECH_TEXT must raise at install time; l0_gate also counts
# occurrences in sitecustomize.py against this table.
# JIT entries below govern transformations of adapter-owned compiler inputs.
# They do not authorize writing to the community source tree.
TEXT_PATCH_ALLOWLIST = {
    ("per_token_quant", "#elif defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)"),
    ("per_token_quant", "reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf)"),
    ("per_token_quant", "__nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3)"),
    ("fused_moe", "assert _is_cuda or _is_xpu, ("),
    ("kda", "t < t_slot"),  # prefill-CP rope fused guard, same pattern
    # torch2.11 cannot infer dtype from numpy scalars (np.prod int64) when
    # building the KV-pool stride tensor (dlpack route); int() restores the
    # pre-dlpack behavior "python int infers an integer dtype".
    ("kv_pool_strides", "np.prod(x.shape[1:]) * x.dtype.itemsize"),
    # gfx936 DTK hipcc (clang) rejects nvcc-isms in upstream JIT rope:
    # bare __syncwarp() (needs mask arg via portability shim) and structured
    # bindings into HIP_vector_type (anonymous union). Patch both to
    # portable equivalents at JIT-source-read time.
    ("rotary_embedding", "  __syncwarp();  // to avoid warp divergence"),
    ("rotary_embedding", "      const auto [x0, x1] = cast<fp32x2_t>(input_vec_x[j]);"),
    # JIT per-token-group-quant fp8: the CUDA DTypeTrait<fp8>::kFloatMax
    # member is compiled out under USE_ROCM; the HIP-side packed convert is
    # not the CUDA intrinsic. Repoint the clamp constant at the literal
    # 448.0f and the packed convert at the DTK __hip_cvt family (OCP e4m3
    # == CUDA e4m3_fn; deep_gemm EP runner needs this JIT kernel).
    ("per_token_quant", "static constexpr float kMaxValue = DTypeTrait<fp8_e4m3_t>::kFloatMax;"),
    ("per_token_quant", "reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf)"),
    ("per_token_quant", "__nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3)"),
    # v0.5.22 per-token-group quant JIT: DTK hipcc declares plain __all but
    # not the mask-form __all_sync, killing the whole module family at name
    # lookup (glm53f decode graph capture, 2026-10-20). Full-mask vote from
    # kernel entry == __all(pred); see hcu_adapter/jit_ptgq.py.
    ("per_token_quant", "if (__all_sync(warp::kFullMask, skip)) return;"),
    (
        "per_token_quant",
        "  SGL_DEVICE static packed2_t quant(const float2 v) {\n"
        "    return packed2_t{float2{fminf(v.x, kMaxValue), fminf(v.y, kMaxValue)}};\n"
        "  }",
    ),
    (
        "per_token_quant",
        "#else\n"
        "      value = Trait::reduce(value, __shfl_xor(value, mask, kStart));\n"
        "#endif",
    ),
    (
        "per_token_quant",
        "#else\n"
        "      value = Trait::reduce(value, __shfl_xor(value, mask, kFinish));\n"
        "#endif",
    ),
    # --- GLM-5.2-DSA (gfx938, box16) launch chain: three more hipcc
    # --- portability text patches, same evidence class as the rope ones.
    # DSA indexer k-cache store: HIP fp8x2 alias is a bare uint16_t (no
    # brace ctor) -> delegate pack_fp8 to the proven deepseek_v4 software
    # packer (E4M3FN 448 clip == this file's kFP8E4M3Max assumption).
    (
        "per_token_quant",
        "#include <sgl_kernel/vec.cuh>\n#include <sgl_kernel/warp.cuh>",
    ),
    (
        "per_token_quant",
        "[[maybe_unused]]\n"
        "SGL_DEVICE fp8x2_e4m3_t pack_fp8(float x, float y) {\n"
        "  return fp8x2_e4m3_t{fp32x2_t{fp8_e4m3_clip(x), fp8_e4m3_clip(y)}};\n"
        "}",
    ),
    (
        "per_token_quant",
        "  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);",
    ),
    (
        "per_token_quant",
        "  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);",
    ),
    # warp.cuh reduce<T>: HIP arm hardcodes a __hip_bfloat162 shuffle
    # receiver -> reduce<T> un-compilable for every non-bf16 T (fp32 warp
    # max in the store kernel). Receiver type widened to T.
    ("per_token_quant", "__hip_bfloat162 _v;"),
    # warp.cuh inclusive scan / broadcast helpers use the *_sync shuffle
    # variants; DTK device headers only declare the non-sync family. First
    # hit: deepseek_v4/topk_impl.cuh radix tie-select -> warp::inclusive_sum
    # -> '__shfl_up_sync' use-of-undeclared-identifier, which failed every
    # dsa_mha_one_shot_* subcase of upstream
    # test/registered/attention/unittests/dsa/test_dsa.py
    # (sgl_kernel_jit_dpsk_v4_topk_v2_false build). HIP shuffles are
    # wave-collective and ignore the mask, so the plain form is the
    # drop-in equivalent.
    ("per_token_quant", "__shfl_up_sync(active_mask, val, offset, kWidth);"),
    ("per_token_quant", "__shfl_down_sync(active_mask, val, offset, kWidth);"),
    ("per_token_quant", "__shfl_sync(active_mask, value, src_lane, kWidth);"),
    # concat_mla.cuh: hand-written PTX (laneid register, L1::no_allocate
    # st/ld with "l" constraint, prefetch.global.L2) -> HIP-portable
    # equivalents (utils.cuh lane identity / plain volatile derefs).
    (
        "platform_detect",
        "SGL_DEVICE int get_lane_id() {\n"
        "  int lane_id;\n"
        '  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));\n'
        "  return lane_id;\n"
        "}",
    ),
    (
        "platform_detect",
        "SGL_DEVICE void st_na_global_v1(const int* ptr, int v) {\n"
        '  asm volatile("st.global.L1::no_allocate.s32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");\n'
        "}",
    ),
    (
        "platform_detect",
        "SGL_DEVICE void st_na_global_v2(const int2* ptr, const int2& v) {\n"
        '  asm volatile("st.global.L1::no_allocate.v2.s32 [%0], {%1, %2};" ::"l"(ptr), "r"(v.x), "r"(v.y) : "memory");\n'
        "}",
    ),
    (
        "platform_detect",
        "SGL_DEVICE int ld_na_global_v1(const int* ptr) {\n"
        "  int r;\n"
        '  asm volatile("ld.global.nc.L1::no_allocate.s32 %0, [%1];" : "=r"(r) : "l"(ptr));\n'
        "  return r;\n"
        "}",
    ),
    (
        "platform_detect",
        "SGL_DEVICE int2 ld_na_global_v2(const int2* ptr) {\n"
        "  int2 r;\n"
        '  asm volatile("ld.global.nc.L1::no_allocate.v2.s32 {%0, %1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(ptr));\n'
        "  return r;\n"
        "}",
    ),
    # deepseek_v4/paged_mqa_metadata.cuh includes NVIDIA cub for BlockScan;
    # DTK ships hipCUB only (namespace cub mirrors the API, umbrella header
    # per the tree's own lora JIT convention) — DSV4-Flash decode graph
    # capture, nmz16 gfx938 2026-10-20. See hcu_adapter/jit_dpskmeta.py.
    ("platform_detect", "#include <cub/block/block_scan.cuh>"),
    # same file, Phase-1/2 hand scans: mask-form shuffles undeclared on DTK;
    # width=32 HIP forms keep the exact CUDA 32-lane warp segmentation.
    ("platform_detect", "uint32_t y = __shfl_up_sync(0xffffffff, v, o);"),
    ("platform_detect", "running = __shfl_sync(0xffffffff, v, 31);"),
    # c_plan.cuh compress-plan: structured bind into HIP uint2 (anonymous
    # union) — 4th site of the ABI fact; unpack_w returns a real uint2 so
    # member reads are type-identical (DSV4-Flash nmz16 capture chain).
    (
        "platform_detect",
        "    const auto [ragged_id, batch_id] = unpack_w(plan_w);",
    ),
    # main_norm_rope / topk_bf16_small / flashmla_sched_meta / fused_norm_rope:
    # bare __syncwarp, %laneid PTX, mask-form shuffles, float2 binds — all
    # established needle classes (jit_rope / jit_concat_mla / warp.cuh
    # precedents); DSV4-Flash capture chain 2026-10-20.
    ("platform_detect", "  __syncwarp();"),
    ("platform_detect", "  const auto [freq_real, freq_imag] = freq;"),
    (
        "platform_detect",
        "        const float other = __shfl_xor_sync(0xFFFFFFFFu, data[i], mask, kWarpThreads);",
    ),
    (
        "platform_detect",
        '  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));',
    ),
    (
        "platform_detect",
        "    const auto t = __shfl_down_sync(device::warp::kFullMask, inc, offset);",
    ),
    (
        "platform_detect",
        "  warp_base = __shfl_sync(warp::kFullMask, warp_base, kWarpThreads - 1);",
    ),
    (
        "platform_detect",
        "    partial += __shfl_xor_sync(uint32_t(-1), partial, offset);",
    ),
    (
        "platform_detect",
        "    const auto [freq_real, freq_imag] = freq;",
    ),
    ("platform_detect", "    __syncwarp();"),
    # --- DSV4 family batch #3 (syntax-preflight driver, nmz16 2026-10-20):
    # --- float2 binds (rope/fp4_indexer_rope/c1/c2/mega), mask shuffles +
    # --- PTX scalar asm (mhc_post_combine_norm_prefill), attribute
    # --- placement + cudaMemcpy (c128_online*), CUDA bit intrinsics
    # --- (topk_bf16_small). Details in hcu_adapter/jit_dpskmeta.py.
    ("platform_detect", "  const auto [f_real, f_imag] = freq_ptr[lane_id];"),
    ("platform_detect", "  const auto [x_real, x_imag] = cast<fp32x2_t>(data);"),
    ("platform_detect", "    const auto [scale, inv_scale] = fp4::block_scale(amax);"),
    ("platform_detect", "    const auto [h0, h1] = cast<fp32x2_t>(head_in[0]);"),
    ("platform_detect", "    const auto [t0, t1] = cast<fp32x2_t>(tail_in[0]);"),
    ("platform_detect", "    const auto [wh0, wh1] = cast<fp32x2_t>(head_w[0]);"),
    ("platform_detect", "    const auto [wt0, wt1] = cast<fp32x2_t>(tail_w[0]);"),
    ("platform_detect", "      const auto [x, y] = cast<fp32x2_t>(input[j]);"),
    ("platform_detect", "      const auto [wx, wy] = cast<fp32x2_t>(weight[j]);"),
    ("platform_detect", "    const auto [x, y] = cast<fp32x2_t>(latent[j]);"),
    ("platform_detect", "      const auto [r0, r1] = cast<fp32x2_t>(rotated);"),
    (
        "platform_detect",
        "      const auto [x, y] = deepseek_v4::fp4::fake_quant_compressed_kv_x2({data[i * 2 + 0], data[i * 2 + 1]}, scale);",
    ),
    ("platform_detect", "    const auto [x, y] = cast<fp32x2_t>(cast<bf16x2_t>(packed));"),
    ("platform_detect", "    const auto [wx, wy] = cast<fp32x2_t>(weight[i]);"),
    ("platform_detect", "    const auto [x, y] = cast<fp32x2_t>(out[i]);"),
    (
        "platform_detect",
        "      const auto [x, y] = deepseek_v4::fp4::fake_quant_compressed_kv_x2({staged[i * 2 + 0], staged[i * 2 + 1]}, scale);",
    ),
    ("platform_detect", "        const auto [v0, v1] = cast<fp32x2_t>(in_vec[i]);"),
    ("platform_detect", "      const float post = __shfl_sync(0xffffffff, coeff, 16 + i);"),
    ("platform_detect", "      const float pre = __shfl_sync(0xffffffff, coeff, 20 + i);"),
    ("platform_detect", "      const float c0 = __shfl_sync(0xffffffff, coeff, i);"),
    ("platform_detect", "      const float c1 = __shfl_sync(0xffffffff, coeff, 4 + i);"),
    ("platform_detect", "      const float c2 = __shfl_sync(0xffffffff, coeff, 8 + i);"),
    ("platform_detect", "      const float c3 = __shfl_sync(0xffffffff, coeff, 12 + i);"),
    (
        "platform_detect",
        "      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));",
    ),
    (
        "platform_detect",
        "    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 2));",
    ),
    (
        "platform_detect",
        "    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, 1));",
    ),
    (
        "platform_detect",
        '      asm("div.full.f32 %0, %1, %2;" : "=f"(mean) : "f"(sum), "f"(5120.f));\n'
        "      // Match Triton's unqualified PTX add: ptxas may contract the\n"
        "      // constant division's reciprocal multiply with this epsilon add.\n"
        '      asm("add.f32 %0, %1, %2;" : "=f"(mean) : "f"(mean), "f"(p.eps));\n'
        '      asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(mean));',
    ),
    (
        "platform_detect",
        "  __shared__ alignas(16) float seg_kv[kTileDim];\n"
        "  __shared__ alignas(16) float seg_max[kTileDim];\n"
        "  __shared__ alignas(16) float seg_sum[kTileDim];",
    ),
    (
        "platform_detect",
        "        RuntimeDeviceCheck(::cudaMemcpy(dst, src, bytes, ::cudaMemcpyHostToDevice));",
    ),
    (
        "platform_detect",
        "  const auto lo = __byte_perm(m[0], m[1], 0x7531);\n"
        "  const auto hi = __byte_perm(m[2], m[3], 0x7531);\n"
        "  const auto nib = __dp4a(static_cast<int>(lo), static_cast<int>(0xF8FCFEFFu), 0);  // -1,-2,-4,-8\n"
        "  return __dp4a(static_cast<int>(hi), static_cast<int>(0x80C0E0F0u), nib);          // -16..-128",
    ),
    ("platform_detect", "      gt[j] = __hgt2_mask(vecs[i][j], pivot_x2);"),
    ("platform_detect", "      eq[j] = __heq2_mask(vecs[i][j], pivot_x2);"),
    # --- DSV4 batch #4: template-body float2 binds (invisible to the
    # --- syntax preflight; detonate at instantiation). Same ABI class.
    ("platform_detect", "  const auto [x_real, x_imag] = cast<fp32x2_t>(elem);"),
    ("platform_detect", "    const auto [x_real, x_imag] = cast<fp32x2_t>(elem);"),
    ("platform_detect", "    const auto [x, y] = cast<fp32x2_t>(elems);"),
    ("platform_detect", "    const auto [x, y] = cast<fp32x2_t>(elems[i]);"),
    ("platform_detect", "  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);"),
    ("platform_detect", "  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);"),
    ("platform_detect", "  const auto [g0, g1] = cast<fp32x2_t>(gate);"),
    ("platform_detect", "  const auto [u0, u1] = cast<fp32x2_t>(up);"),
    (
        "platform_detect",
        "    const auto [x, y] = silu_and_mul<kApplySwigluLimit>(gate_vec[i], up_vec[i], params.swiglu_limit);",
    ),
    # concat_mla.cuh host side: TensorMatcher pins the dldevice enum to
    # kDLCUDA; ROCm tensors arrive as kDLROCM (upstream's own kDLGPU
    # indirection in utils.cuh exists for this reason). Six identical sites.
    ("platform_detect", "        .with_device<kDLCUDA>(device)"),
    # gemm/dot_product.cuh: bf16 dot-product accumulation decomposes
    # fp32x2_t (HIP float2, anonymous union) — third site of the same ABI
    # fact after rope.cuh / fused_store_index_cache.cuh. .x/.y member reads.
    (
        "platform_detect",
        "    const auto [a0, a1] = cast<fp32x2_t>(a[i]);",
    ),
    (
        "platform_detect",
        "    const auto [b0, b1] = cast<fp32x2_t>(b[i]);",
    ),
}


def register(op, impl, target, mechanism, role=ROLE_ACTIVE, source=None,
             detail=None):
    """Record that `op` is now served by `impl` via `mechanism`.

    target  — dotted symbol that got rebound/patched (what a debugger would
              set a breakpoint in, e.g. "sglang...fused_moe.fused_experts")
    source  — module that performed the install; defaults to caller's file
    Raises on: unknown op (not in catalog), role violation, duplicate
    binding of a target, non-allowlisted text patch.
    """
    from . import hcu_config

    if op not in hcu_config.BACKEND_CATALOG:
        raise ValueError(
            f"registry: op {op!r} not in BACKEND_CATALOG; a binding without"
            " a catalog entry does not exist (see docs/governance.md)"
        )
    if mechanism == MECH_TEXT:
        needle = detail if isinstance(detail, str) else None
        if (op, needle) not in TEXT_PATCH_ALLOWLIST:
            raise ValueError(
                f"registry: text-patch for {op!r} ({needle!r}) is not in"
                " TEXT_PATCH_ALLOWLIST -- new patches need a governance"
                " review (docs/governance.md §3)"
            )
    if mechanism not in (MECH_SEAM, MECH_REBIND, MECH_TEXT):
        raise ValueError("Unknown installation mechanism: " + mechanism)
    if role not in (ROLE_ACTIVE, ROLE_REF):
        raise ValueError("Unknown implementation role: " + role)
    if op in hcu_config._DEEPINFER_CATALOG:
        from .ops.selector import guard
        guard(op)
        candidates = hcu_config._DEEPINFER_CATALOG[op]
        if impl not in {item["implementation"] for item in candidates.values()}:
            raise ValueError("Implementation is absent from catalog: " + str(impl))
        configured = hcu_config.load()["ops"].get(op)
        if configured is not None and candidates[configured]["implementation"] != impl:
            raise ValueError("Selected backend differs from installed provider for " + op)
        previous = _RECORDS.get(op)
        if previous is not None and (previous["implementation"], previous["target"], previous["mechanism"]) != (impl, target, mechanism):
            raise ValueError("Conflicting active binding: " + op)
    src = source or _caller()
    prev = _BOUND.get(target)
    if prev is not None and prev != op:
        raise ValueError(
            f"registry: target {target!r} already bound to op {prev!r};"
            " duplicate patches are forbidden -- extend the existing entry"
        )
    _BOUND[target] = op
    if impl is None:
        impl_name = "None"
    elif isinstance(impl, str):
        impl_name = str.__repr__(impl)
    elif type(impl) in (types.FunctionType, types.BuiltinFunctionType, types.MethodType, type):
        impl_name = impl.__qualname__
    else:
        impl_name = type(impl).__qualname__
    _RECORDS[op] = {
        "op": op,
        "impl": impl_name,
        "implementation": impl if isinstance(impl, str) else impl_name,
        "target": target,
        "mechanism": mechanism,
        "role": role,
        "source": src,
    }
    if role == ROLE_REF:
        # refs never serve traffic; nag loudly if registered from a bridge
        LOG.debug("registry: ref impl for %s recorded from %s", op, src)


def _caller():
    stack = sys._getframe(2)
    return f"{stack.f_globals.get('__name__', '?')}:{stack.f_lineno}"


def get(op):
    return _RECORDS.get(op)


def records():
    return dict(_RECORDS)


TEXT_PATCH_TARGET_PREFIX = "text:"  # target convention for text patches


def dump(path=None):
    """Machine-readable dump. dict if path is None, else JSON written to path.

    Structure: {"ops": {op: record}, "unbound": [catalog ops with no
    active record], "env": every resolved HCU_* value+source (from
    hcu_env.log_summary's data)}. CI and the config-level bisection
    scripts consume this; humans read log_summary().
    """
    from . import hcu_config
    from . import hcu_env

    unbound = [op for op in hcu_config.BACKEND_CATALOG if op not in _RECORDS and op not in {r["op"] for r in _OVERRIDES.values()}]
    doc = {
        "ops": _RECORDS,
        "overrides": overrides(),
        "unbound": unbound,
        "env": {
            name: dict(zip(("value", "source"), hcu_env.resolve(name)))
            for name in hcu_env.REGISTRY
        },
    }
    if path is None:
        return doc
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return doc


def log_summary(logger=None):
    """One line per installed binding — the 'who is running' startup log."""
    lg = logger or LOG
    if not _RECORDS:
        lg.warning("HCU adapter runtime: no ops installed (pure upstream)")
        return
    for op in sorted(_RECORDS):
        r = _RECORDS[op]
        lg.warning(
            "HCU adapter runtime: %-18s <- %-40s [%s/%s] via %s",
            op, r["impl"], r["role"], r["mechanism"], r["target"],
        )
    from . import hcu_config

    unbound = [op for op in hcu_config.BACKEND_CATALOG if op not in _RECORDS and op not in {r["op"] for r in _OVERRIDES.values()}]
    if unbound:
        lg.warning(
            "HCU adapter runtime: catalog ops on upstream path (no patch):"
            " %s", ", ".join(sorted(unbound)),
        )


def snapshot():
    from . import hcu_config as config
    return {
        "overrides": overrides(),
        "bindings": {op: {"implementation": record["implementation"],
                          "target": record["target"], "mechanism": record["mechanism"]}
                     for op, record in _RECORDS.items()},
        "unbound": sorted(set(config.BACKEND_CATALOG) - _RECORDS.keys() - {r["op"] for r in _OVERRIDES.values()}),
    }


_OVERRIDES = {}


def replace(owner, name, implementation, *, namespace, expected):
    from . import hcu_config as config
    from .ops.selector import guard
    module_name = getattr(owner, "__name__", type(owner).__name__)
    if not hasattr(owner, "__path__") and hasattr(owner, "__module__"):
        module_name = owner.__module__ + "." + getattr(owner, "__qualname__", module_name)
    target = module_name + "." + name
    if getattr(owner, name, None) is not expected:
        raise RuntimeError("Binding changed before replacement: " + target)
    previous = _OVERRIDES.get(target)
    if previous is not None:
        if previous["namespace"] != namespace or previous["object"] is not expected:
            raise RuntimeError("Conflicting overlay owner: " + target)
        if implementation is expected:
            return expected

    guard(target)
    parent = _BOUND.get(target)
    if parent is not None:
        guard(parent)
    op = namespace + ":" + target
    provider = getattr(implementation, "__module__", "") + "." + getattr(implementation, "__qualname__", type(implementation).__name__)
    if op in config.BACKEND_CATALOG and previous is None:
        raise RuntimeError("Duplicate plugin catalog entry: " + op)
    setattr(owner, name, implementation)
    config.BACKEND_CATALOG[op] = {
        "var": None, "summary": target,
        "backends": {"plugin": {"status": "candidate", "note": "Explicit overlay of the effective adapter baseline", "implementation": provider}},
    }
    baseline = (previous["baseline"] if previous is not None else
                getattr(expected, "__module__", "") + "." + getattr(expected, "__qualname__", type(expected).__name__))
    _OVERRIDES[target] = {
        "namespace": namespace, "op": op, "implementation": provider,
        "baseline_op": parent, "baseline": baseline,
        "history": ([] if previous is None else previous["history"] + [previous["implementation"]]),
        "object": getattr(owner, name),
    }
    return expected


def overrides():
    return {target: {key: value for key, value in record.items() if key != "object"}
            for target, record in _OVERRIDES.items()}
