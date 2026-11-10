# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Alias the DTK ``deepgemm`` wheel as upstream ``deep_gemm`` (EP bridge).

The DTK stack ships Hygon's deepgemm fork (2.1.0+dtk2604) under the
one-word package name ``deepgemm`` with upstream-compatible entry points
(``m_grouped_bf16_gemm_nt_masked`` / ``m_grouped_fp8_gemm_nt_masked[_ll]``
/ ``..._contiguous`` etc.). sglang only ever imports ``deep_gemm``, so on
this stack the DeepGEMM-backed MoE runners (the only deepep_ll-capable
runners besides aiter/npu ones) are unreachable.

``install()`` (called from sitecustomize under ``HCU_DEEPGEMM_BRIDGE=1``)
builds a ``deep_gemm`` module in ``sys.modules`` backed by the DTK package
plus the ``deep_gemm.utils.layout`` helper module the wrapper imports at
module load. The utils.layout scale transforms have no verified DTK
equivalent and raise loudly. The adapter's existing Triton grouped FP8
route instead reads logical scales and installs an identity layout
transform; its single-device numerical/graph contracts do not validate
distributed DeepEP. Disabling that route retains the layout refusal.

fp8 contiguous verdict (R-04 probe campaign, 2026-10-18, nmz26 GPU4,
wheel 2.1.0+dtk2604): NOT unlockable from this wheel, for three
independent reasons. (1) The ``m_grouped_fp8_gemm_nt_contiguous``
dispatcher has no gfx936 arm at all (only i8/w6 variants) — the EP
target card never gets an fp8 path. (2) The gfx938 arm
(``..._wasp_pair32``) takes ``a_scale`` as per-token whole-K 1-D
[M]/[M,1] only (per-128-group [M, K/128] is rejected), so even folding
sglang's DeepEP recipe onto it is lossy re-quantisation; and its signed
numerics are broken: unit-probe vs dequant golden gives rel 0.88, a
negatives-zeroed golden does NOT fit (rel 1.0017 — unlike the
paged-MQA clamp pathology), positives still hold ~0.035 residual.
(3) The tensor-wise marlin variant is also broken on signed data with
unit scales (rel 1.22). Keep the stubs raising; revisit only with a new
wheel or a hand-rolled kernel (probe: tmp_fetch/probe_r04_contig.py).

Ragged MQA entry (2026-10-18 zero-env gate): the wheel also lacks
``fp8_mqa_logits`` (kpool long-prefill plan, dsa_indexer_kpool.py:933);
the shim routes it to the exact torch emu in mqa_logits_emu.
"""

from __future__ import annotations

import sys
import types

import torch


def _make_layout_module() -> types.ModuleType:
    mod = types.ModuleType("deep_gemm.utils.layout")

    def _unimplemented(name):
        def _raise(*args, **kwargs):
            raise NotImplementedError(
                f"deep_gemm.utils.layout.{name}: no verified DTK deepgemm "
                "equivalent (fp8 DeepEP scale layouts unvalidated on HCU); "
                "bf16 grouped GEMM paths must not reach here"
            )

        return _raise

    mod.get_mn_major_tma_aligned_tensor = _unimplemented(
        "get_mn_major_tma_aligned_tensor"
    )
    mod.get_col_major_tma_aligned_tensor = _unimplemented(
        "get_col_major_tma_aligned_tensor"
    )
    mod.transform_sf_into_required_layout = _unimplemented(
        "transform_sf_into_required_layout"
    )
    return mod


def fp8_gemm_nt_dequant(lhs, rhs, out):
    """Dense fp8 NT GEMM with 2-D block scales via explicit dequant.

    sglang hands (q, scales) pairs: lhs = (A [M, K] fp8, As), rhs = (B [N, K]
    fp8, Bs). As is per-token-group-128 [M, K/128] — row-major or the
    column-major TMA-aligned VIEW (stride (1, aligned_M), sliced on dim 0;
    ``reshape`` restores the logical [M, K/128] either way). Bs is per-block
    [N/128, K/128] (128x128 recipe) or per-row [N, K/128] (1x128 recipe);
    both reduce to the same broadcast after expanding the N dim to blocks.

    out is written in place and returned (entrypoint contract).
    """
    a, a_s = lhs[0], lhs[1]
    b, b_s = rhs[0], rhs[1]
    m, k = a.shape
    a_s_groups = a_s.reshape(m, -1).shape[-1] if a_s.numel() and m else 0
    if a_s_groups == 0 or k % a_s_groups != 0:
        # Degenerate warmup/compile probe (empty scales, m=0, non-divisible
        # k): no valid block semantics exist — produce zeros rather than
        # SIGFPE-ing in the view math (rank dies mid-warmup otherwise).
        out.zero_()
        return out
    block_k = k // a_s_groups
    n_block = b_s.shape[0] if b_s.numel() else 0
    if n_block == 0 or b.shape[0] % n_block != 0:
        out.zero_()
        return out
    block_n = b.shape[0] // n_block
    a_dq = (
        a.to(torch.float32).view(m, -1, block_k)
        * a_s.reshape(m, -1).unsqueeze(-1)
    ).view(m, k)
    # b rows are physically (n_block, block_n, k); scale per (n_block, k/128)
    # broadcasts over the inner-row dim: [n_block, 1, k_groups, 1].
    b_dq = (
        b.to(torch.float32).view(n_block, block_n, -1, block_k)
        * b_s.reshape(n_block, -1)[:, None, :, None]
    ).view(b.shape[0], k)
    torch.matmul(a_dq.to(out.dtype), b_dq.to(out.dtype).t(), out=out)
    return out


def install():
    """Bind ``deep_gemm`` (+ ``deep_gemm.utils.layout``) to the DTK wheel.

    No-op when a real ``deep_gemm`` is importable or an alias already
    exists, so a future upstream-wheel install wins automatically.
    """
    if "deep_gemm" in sys.modules:
        return sys.modules["deep_gemm"]
    try:
        import importlib.util

        if importlib.util.find_spec("deep_gemm") is not None:
            return None  # real deep_gemm present; do not shadow it
    except Exception:
        pass
    import deepgemm

    shim = types.ModuleType("deep_gemm")
    shim.__dict__.update(
        {k: v for k, v in vars(deepgemm).items() if not k.startswith("__")}
    )
    shim.__package__ = "deep_gemm"
    shim.__file__ = getattr(deepgemm, "__file__", None)
    shim.dtk_deepgemm = deepgemm  # escape hatch to the raw fork

    _dtk_masked = getattr(deepgemm, "m_grouped_bf16_gemm_nt_masked", None)
    _dtk_pack_bf16 = getattr(deepgemm, "pack_bf16_masked_weight", None)
    if _dtk_masked is not None:

        _bf16_packed = {}  # (data_ptr, shape, dtype) -> packed weight

        def _bf16_masked(a, b, d, masked_m=None, expected_m=None,
                         expected_m_per_group=None, config=None, **unknown):
            # sglang names the 5th arg `expected_m` (entrypoint positional,
            # compile_utils warmup passes it as a kwarg); the DTK fork calls
            # it `expected_m_per_group` and puts `config` last. The default
            # masked mode (2000, BM64/BN256/BK128 HIPC) reads the weight in a
            # packed layout — raw (G,N,K) weights give garbage numbers, so
            # weights are packed once and cached by storage identity.
            if unknown:
                raise TypeError(
                    f"m_grouped_bf16_gemm_nt_masked: unexpected kwargs "
                    f"{sorted(unknown)}"
                )
            if expected_m is None:
                expected_m = expected_m_per_group
            if masked_m is None or expected_m is None:
                raise TypeError(
                    "m_grouped_bf16_gemm_nt_masked: masked_m and "
                    "expected_m are required"
                )
            if a.dtype != torch.bfloat16:
                # dispatch can hand us fp8/fp32 activations depending on the
                # runner dtype config; the DTK masked bf16 kernel is strictly
                # bf16-in — upcast a (b is handled by the pack path below).
                a = a.to(torch.bfloat16)
            if b.dtype != torch.bfloat16:
                b = b.to(torch.bfloat16)
            if _dtk_pack_bf16 is not None and b.dim() == 3:
                key = (b.data_ptr(), tuple(b.shape), b.dtype)
                packed = _bf16_packed.get(key)
                if packed is None:
                    packed = _dtk_pack_bf16(b.contiguous())
                    _bf16_packed[key] = packed
                b = packed
            if config is not None:
                return _dtk_masked(a, b, d, masked_m, expected_m, config)
            return _dtk_masked(a, b, d, masked_m, expected_m)

        shim.m_grouped_bf16_gemm_nt_masked = _bf16_masked

    utils = types.ModuleType("deep_gemm.utils")
    layout = _make_layout_module()
    utils.layout = layout
    shim.utils = utils

    # DSA indexer (v0.5.21) resolves the paged-MQA trio from deep_gemm on the
    # is_cuda semantic branch (the aiter arm is is_hip-gated and HCU never
    # takes it). The DTK wheel lacks all three; route to the torch
    # implementation (same numerics as the proven MQA emu).
    _missing_paged_mqa = (
        getattr(deepgemm, "fp8_paged_mqa_logits", None) is None
        or getattr(deepgemm, "get_paged_mqa_logits_metadata", None) is None
        or getattr(deepgemm, "get_num_sms", None) is None
    )
    if _missing_paged_mqa:
        from . import dsa_mqa_logits_emu as _dpm

        if getattr(shim, "get_num_sms", None) is None:
            shim.get_num_sms = _dpm.get_num_sms
        if getattr(shim, "get_paged_mqa_logits_metadata", None) is None:
            shim.get_paged_mqa_logits_metadata = _dpm.get_paged_mqa_logits_metadata
        if getattr(shim, "fp8_paged_mqa_logits", None) is None:
            shim.fp8_paged_mqa_logits = _dpm.fp8_paged_mqa_logits

    # Same story for the RAGGED entry (kpool long-prefill plan,
    # dsa_indexer_kpool.py:933): `deep_gemm.fp8_mqa_logits(q, (k, scale), w,
    # starts, ends, clean_logits=...)` — the is_cuda branch reaches it and the
    # DTK wheel only ships paged_mqa_logits, so every long-context prefill
    # died with AttributeError until 2026-10-18 (zero-env gate caught it).
    # Route to the exact ragged torch emu; it always returns the -inf-cleaned
    # variant, which satisfies the stricter contract (callers reading
    # clean_logits=False only expect stale data outside [ks,ke), never use
    # it).
    if getattr(deepgemm, "fp8_mqa_logits", None) is None:
        from . import mqa_logits_emu as _mle

        def _fp8_mqa_logits(q_fp8, kv, weights, cu_starts, cu_ends,
                            clean_logits=True, **_kw):
            # deep_gemm contract: kv arrives as the (k_fp8, k_scale) tuple
            if not isinstance(kv, (tuple, list)) or len(kv) != 2:
                raise TypeError(
                    "fp8_mqa_logits shim expects kv as the (k, scale) tuple"
                )
            kv, kv_scales = kv[0], kv[1]
            # R-20: the C4 nonpaged plan (dsv4/indexer.py:732) hits this
            # entry 44x per decode step; the torch emu is ~58 aten kernels
            # per call (the dominant wave6 elementwise family, fingerprinted
            # 2026-10-24). Route to the homegrown triton sibling under the
            # same impl switch as the paged entry (R-13 #4 default = triton);
            # the emu stays the golden and the explicit fallback.
            impl = "loop"
            try:
                from .hcu_env import resolve

                impl, _src = resolve("HCU_MQA_LOGITS_IMPL")
            except Exception:
                pass
            if impl == "triton":
                from .mqa_logits_triton import triton_ragged_mqa_logits as _tri_r

                return _tri_r(
                    q_fp8, kv, kv_scales, weights, cu_starts, cu_ends
                )
            return _mle._ragged_fp8_mqa_logits(
                q_fp8, kv, kv_scales, weights, cu_starts, cu_ends, **_kw
            )

        shim.fp8_mqa_logits = _fp8_mqa_logits

    # sm-count scheduler hooks (configure_deep_gemm_num_sms context) and the
    # PDL toggle: no DTK equivalents — setters are stateful no-ops.
    if getattr(shim, "set_num_sms", None) is None:
        def _set_num_sms(n):
            return None

        shim.set_num_sms = _set_num_sms
    if getattr(shim, "set_pdl", None) is None:
        def _set_pdl(enable):
            return None

        shim.set_pdl = _set_pdl

    # The DTK pernorm alias crashes on the current 0.5.22 mHC split-K call
    # (K=20480, N=24). Serve the existing checked public contract instead;
    # no platform, fusion, or model-default flags change.
    if (
        getattr(shim, "tf32_hc_prenorm_gemm", None) is None
        and getattr(deepgemm, "tf32_hc_pernorm_gemm", None) is not None
    ):
        from .kernels.mhc_emu import tf32_hc_prenorm_gemm
        from . import registry

        shim.tf32_hc_prenorm_gemm = tf32_hc_prenorm_gemm
        registry.register(
            "deep_gemm.tf32_hc_prenorm_gemm",
            "hcu_adapter.kernels.mhc_emu.tf32_hc_prenorm_gemm",
            "deep_gemm.tf32_hc_prenorm_gemm",
            registry.MECH_SEAM,
        )

    # v0.5.21 dense fp8 w8a8 block-linear (Glm5Next FP8 checkpoints): sglang
    # entrypoint.gemm_nt_f8f8bf16 calls deep_gemm.fp8_gemm_nt(lhs, rhs, out).
    # The DTK fork spells the dense fp8 GEMM `fp8_gemm(a, b, d, ...)` — but a
    # name-only alias is numerically WRONG: nmz26 GPU probe (probe10/12) shows
    # the fork's kernel only honors per-row scales [rows, 1]; given sglang's
    # per-token-group-128 As [M, K/128] (col-major TMA-aligned) and per-block
    # Bs [N/128, K/128] it either ignores As or reads them OOB (cos=0.004,
    # maxrel=4e5 vs dequant reference — the T1 word-soup root cause). Instead
    # dequantize both operands with their fp32 block scales and run one bf16
    # matmul: the fork's own dense-fp8 arm has no correct 128x128-block mode
    # to forward to, and the bf16 matmul it degenerates to was ~60x faster
    # than the triton w8a8 fallback on the same tensors (0.14ms vs 7.1ms at
    # 512³ cold; bf16 core 0.02ms).
    if (
        getattr(shim, "fp8_gemm_nt", None) is None
        and getattr(deepgemm, "fp8_gemm", None) is not None
    ):

        def _fp8_gemm_nt(lhs, rhs, out, **_unknown):
            if _unknown:
                raise TypeError(
                    f"fp8_gemm_nt dequant wrapper: unexpected kwargs "
                    f"{sorted(_unknown)}"
                )
            return fp8_gemm_nt_dequant(lhs, rhs, out)

        shim.fp8_gemm_nt = _fp8_gemm_nt

    sys.modules["deep_gemm"] = shim
    sys.modules["deep_gemm.utils"] = utils
    sys.modules["deep_gemm.utils.layout"] = layout

    # An earlier import in THIS process (adapter import chain) may have
    # evaluated deep_gemm_wrapper.configurer against the missing real
    # ``deep_gemm`` package: ENABLE_JIT_DEEPGEMM then froze False and
    # entrypoint skipped its ``if ENABLE_JIT_DEEPGEMM:`` definitions, so
    # the fp8 arms later die on deep_gemm_wrapper.get_mn_major_tma_aligned_
    # tensor (DSV4F fp8 dp8ep8 boot, nmz16 2026-10-21). Drop the package
    # from sys.modules; the next import re-evaluates against this shim
    # (fresh probe: ENABLE_JIT True, sm 93).
    for _n in [
        n
        for n in sys.modules
        if n == "sglang.srt.layers.deep_gemm_wrapper"
        or n.startswith("sglang.srt.layers.deep_gemm_wrapper.")
    ]:
        del sys.modules[_n]

    # One-shot imports that ran before this install leave the sglang
    # `try: import deep_gemm except ImportError as e: deep_gemm = e` pattern
    # bound to the exception object (dsa_indexer_kpool.py does exactly this
    # at module level) — rebind those names to the shim.
    for _m in list(sys.modules.values()):
        if not isinstance(_m, types.ModuleType):
            continue
        _v = getattr(_m, "deep_gemm", None)
        if isinstance(_v, BaseException):
            _m.deep_gemm = shim

    # masked-layout grouped GEMM (DeepEP low-latency decode, bf16): the
    # fork only ships the packed-layout masked kernel and the bridge's
    # pack cache DOUBLES expert memory (DSV4-Flash EP: 69 GB raw + 69 GB
    # packed > 144 GB card -> OOM at decode graph capture, nmz16
    # 2026-10-21). Triton reimplementation on NATIVE [E, N, K] weights:
    # single copy, shared with the contiguous path. HCU_TRITON_MASKED=0
    # keeps the fork pack path.
    try:
        from . import hcu_env as _env

        if _env.enabled("HCU_TRITON_MASKED"):
            from . import deep_gemm_masked_triton as _mt

            shim.m_grouped_bf16_gemm_nt_masked = (
                _mt.m_grouped_bf16_gemm_nt_masked
            )
    except Exception:
        import logging

        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: triton masked grouped GEMM not installed",
            exc_info=True,
        )

    # contiguous-layout grouped GEMM (DeepEP normal / auto prefill): the
    # DTK w16a16 contiguous kernel is broken (2D input VMFault, 3D input
    # wrong numbers) and the fork lacks the mk-alignment get/set pair the
    # wrapper's contiguous_layout_alignment_scope needs. A triton
    # reimplementation (native [E, N, K] weights, alignment-padded row
    # blocks) replaces both unless HCU_TRITON_CONTIG=0 keeps the raw DTK
    # escape hatch.
    try:
        from . import hcu_env as _env

        if _env.enabled("HCU_TRITON_CONTIG"):
            from . import deep_gemm_contig_triton as _ct

            shim.m_grouped_bf16_gemm_nt_contiguous = (
                _ct.m_grouped_bf16_gemm_nt_contiguous
            )
            shim.get_mk_alignment_for_contiguous_layout = (
                _ct.get_mk_alignment_for_contiguous_layout
            )
            shim.set_mk_alignment_for_contiguous_layout = (
                _ct.set_mk_alignment_for_contiguous_layout
            )
    except Exception:
        import logging

        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: triton contiguous grouped GEMM not installed",
            exc_info=True,
        )

    # w8a8 block-scaled fp8 grouped GEMMs (DeepSeek 128x128 recipe): the
    # fork's fp8 arms are per-channel-only with broken signed numerics
    # (R-04), which had forced DSV4F DeepEP onto bf16 experts (2x weight
    # bytes). Triton reimplementations on native [E, N, K] fp8 weights
    # (masked + contiguous) halve the expert traffic. The wrapper's
    # TMA-aligned scale transform is bypassed with an IDENTITY on the
    # layout module -- entrypoint binds the function object at import
    # time, so the identity must be in place before sglang imports, and
    # these kernels index the logical [.., K//128] layout regardless.
    try:
        from . import hcu_env as _env

        if _env.enabled("HCU_TRITON_MASKED_FP8"):
            from . import deep_gemm_fp8_triton as _f8t

            shim.fp8_m_grouped_gemm_nt_masked = (
                _f8t.fp8_m_grouped_gemm_nt_masked
            )
            shim.m_grouped_fp8_gemm_nt_contiguous = (
                _f8t.m_grouped_fp8_gemm_nt_contiguous
            )
            layout.get_mn_major_tma_aligned_tensor = lambda t, *a, **k: t
    except Exception:
        import logging

        logging.getLogger("hcu.adapter").warning(
            "HCU adapter: triton fp8 grouped GEMMs not installed",
            exc_info=True,
        )
    return shim
