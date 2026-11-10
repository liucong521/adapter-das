# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JIT per_token_group_quant.{cuh} HIP fp8 portability patch.

Compiler-input transformation: _apply operates only on private adapter staging
assets via jit_sources. install() delegates to that complete bundle pipeline;
no community source or module origin is rewritten.


The quant kernel's fp8 `WeightTrait` references CUDA-only traits
(`DTypeTrait<fp8_e4m3_t>::kFloatMax` — that specialization is compiled
out under USE_ROCM in type.cuh, where HIP aliases fp8 to raw uint8
storage). The DTK HIP runtime *does* carry the full conversion family
(`__hip_cvt_float2_to_fp8x2` with `__HIP_E4M3` OCP interpretation), so
the patch re-points the fp8 clamp constant at the literal 448.0f and the
packed convert at the DTK API. Bit-parity note: SATFINITE + pre-clamp
matches the CUDA trait comment in-place (non-finite -> 448/-448 codes).
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

# E4M3 max as a literal (the trait member does not exist under USE_ROCM)
_NEEDLE_MAX = "static constexpr float kMaxValue = DTypeTrait<fp8_e4m3_t>::kFloatMax;"
_REPL_MAX = "static constexpr float kMaxValue = 448.0f;  // HCU: literal (HIP trait is compiled out); OCP e4m3 fn semantics"

_NEEDLE_QUANT = (
    "  SGL_DEVICE static packed2_t quant(const float2 v) {\n"
    "    return packed2_t{float2{fminf(v.x, kMaxValue), fminf(v.y, kMaxValue)}};\n"
    "  }"
)
_REPL_QUANT = (
    "  SGL_DEVICE static packed2_t quant(const float2 v) {\n"
    "    // HCU: HIP fp8 is storage-typed (uint16 packed codes); use the\n"
    "    // DTK conversion family with OCP e4m3 interpretation (== CUDA\n"
    "    // e4m3_fn). Pre-clamp keeps the SATFINITE comment above true.\n"
    "    float2 c = float2{fminf(v.x, kMaxValue), fminf(v.y, kMaxValue)};\n"
    "    return (packed2_t)__hip_cvt_float2_to_fp8x2(c, __HIP_SATFINITE, __HIP_E4M3);\n"
    "  }"
)

_PATCHES = (
    ("gemm/per_token_group_quant.cuh", _NEEDLE_MAX, _REPL_MAX),
    ("gemm/per_token_group_quant.cuh", _NEEDLE_QUANT, _REPL_QUANT),
)

# v0.5.22 added a second-generation kernel file whose fp8 store path calls
# the CUDA conversion family by its __nv spelling (per_token_group_quant_
# 8bit_v2.cuh:325/332 — gfx938 hipcc: unknown type name
# '__nv_fp8x2_storage_t'; undeclared __NV_SATFINITE/__NV_E4M3). DTK's
# amd_hip_fp8.h carries the identical-shape __hip family; the mapping is
# purely nominal (both SATFINITE + OCP e4m3).
_NEEDLE_V2_PACKED = "reinterpret_cast<__nv_fp8x2_storage_t*>(&output_buf)"
_REPL_V2_PACKED = "reinterpret_cast<__hip_fp8x2_storage_t*>(&output_buf)"
_NEEDLE_V2_CVT = "__nv_cvt_float2_to_fp8x2(outputx2, __NV_SATFINITE, __NV_E4M3)"
_REPL_V2_CVT = "__hip_cvt_float2_to_fp8x2(outputx2, __HIP_SATFINITE, __HIP_E4M3)"

_PATCHES_V2 = (
    ("gemm/per_token_group_quant_8bit_v2.cuh", _NEEDLE_V2_PACKED, _REPL_V2_PACKED),
    ("gemm/per_token_group_quant_8bit_v2.cuh", _NEEDLE_V2_CVT, _REPL_V2_CVT),
)

# The community DSV4 software converter saturates exp8 >= 15, collapsing
# the valid OCP top segment [256, 448] to 448. On verified gfx938, use
# DTK's OCP conversion branch, as already used by generic/v2 quantization.
# Keep other architectures (especially gfx942/FNUZ) on their original path.
_NEEDLE_DSV4_CVT = "#elif defined(__gfx950__) || defined(__gfx1200__) || defined(__gfx1201__)"
_REPL_DSV4_CVT = _NEEDLE_DSV4_CVT + " || defined(__gfx938__)"
_PATCHES_INCLUDE = (
    ("deepseek_v4/fp8_utils.cuh", _NEEDLE_DSV4_CVT, _REPL_DSV4_CVT),
)

# HIP: __hip_bfloat162 has no __shfl_xor overload (reduce<double/bf162> in
# the quant kernel goes through warp.cuh) — route it through the uint16
# bit pattern and shuffle that.
_NEEDLE_SHFL = (
    "#else\n"
    "      value = Trait::reduce(value, __shfl_xor(value, mask, kStart));\n"
    "#endif"
)
_REPL_SHFL = (
    "#else\n"
    "      T _v;\n"
    "      if constexpr (std::is_same_v<T, __hip_bfloat162>) {\n"
    "        unsigned int _b = *reinterpret_cast<unsigned int*>(&value);\n"
    "        _b = __shfl_xor(_b, mask, kStart);\n"
    "        _v = *reinterpret_cast<__hip_bfloat162*>(&_b);\n"
    "      } else {\n"
    "        _v = __shfl_xor(value, mask, kStart);\n"
    "      }\n"
    "      value = Trait::reduce(value, _v);\n"
    "#endif"
)

# same for the kFinish branch
_NEEDLE_SHFL2 = (
    "#else\n"
    "      value = Trait::reduce(value, __shfl_xor(value, mask, kFinish));\n"
    "#endif"
)
_REPL_SHFL2 = _REPL_SHFL.replace("mask, kStart", "mask, kFinish")

_PATCHES_WARP = (
    ("warp.cuh", _NEEDLE_SHFL, _REPL_SHFL),
    ("warp.cuh", _NEEDLE_SHFL2, _REPL_SHFL2),
)


def _apply(root, register):
    """Transform adapter-owned staging files; never pass a community root."""

    csrc = root / "csrc"
    inc = root / "include" / "sgl_kernel"
    applied = 0
    missing = []
    for base, rel, needle, repl in (
        [(csrc, r, n, rp) for (r, n, rp) in _PATCHES + _PATCHES_V2]
        + [(inc, r, n, rp) for (r, n, rp) in _PATCHES_WARP + _PATCHES_INCLUDE]
    ):
        # _PATCHES_WARP rel paths are already relative to inc
        p = base / rel
        text = p.read_text(encoding="utf-8")
        if repl in text:
            applied += 1
            continue
        if needle not in text:
            missing.append(rel)
            continue
        from hcu_adapter.atomic_io import atomic_write

        atomic_write(p, text.replace(needle, repl))
        applied += 1

    if missing:
        raise RuntimeError("jit_quant_fp8: upstream needles changed: " + ",".join(missing))

    import hcu_adapter.registry as _reg

    # TEXT_PATCH_ALLOWLIST keys are (op, exact needle); the registry gate
    # fingerprints a text patch by its detail, so each needle registers as
    # its own record (same op on the same file target is not a conflict).
    for r, n, _rp in _PATCHES + _PATCHES_V2 + _PATCHES_WARP + _PATCHES_INCLUDE:
        register(
            "per_token_quant",
            "gfx936-hipcc-fp8-cvt",
            target=f"text:sglang/kernels/jit/{'csrc/' + r if r.startswith('gemm/') else 'include/sgl_kernel/' + r}",
            mechanism=_reg.MECH_TEXT,
            detail=n,
        )
    LOG.warning(
        "HCU adapter: JIT per-token-group-quant fp8 patch installed "
        "(%d edits)", applied,
    )


def install():
    """Compatibility entry: prepare the complete immutable JIT input bundle."""
    from .jit_sources import install as install_inputs

    return install_inputs()
