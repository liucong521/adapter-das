# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""JIT sgl_kernel/warp.cuh portability patch — HIP arm reduce<T> + shfl *_sync.

Compiler-input transformation: _apply operates only on private adapter staging
assets via jit_sources. install() delegates to that complete bundle pipeline;
no community source or module origin is rewritten.


Two independent text patches on the same header:

1. ``warp::reduce`` declares ``__hip_bfloat162 _v;`` unconditionally in both
   shuffle loops of its HIP branch, then feeds ``_v`` through a
   ``std::is_same_v<T, __hip_bfloat162>`` constexpr split. The bf16x2 packed-bit
   path is fine, but the generic ``else`` assigns the shuffle result of a *non*
   -bf16 ``T`` (e.g. float — every fp32 warp max/sum/min reduce) into a
   ``__hip_bfloat162`` and converts back: DTK hipcc rejects both the assignment
   ("no viable overloaded =") and the final ``Trait::reduce(value, _v)``
   ("no viable conversion from '__hip_bfloat162' to 'const float'"). On HIP
   this made reduce<T> un-compilable for every T except __hip_bfloat162.
   Fix: type the shuffle receiver as ``T``.

2. DTK's device headers provide the non-sync ``__shfl_xor`` family but NOT the
   ``*_sync`` variants warp.cuh's inclusive scan / broadcast helpers use (first
   hit in the wild: deepseek_v4/topk_impl.cuh radix tie-select ->
   warp::inclusive_sum -> ``__shfl_up_sync`` "use of undeclared identifier",
   which broke every ``dsa_mha_one_shot_*`` case in upstream
   attention/unittests/dsa/test_dsa.py). HIP shuffles are wave-collective and
   ignore the mask, so the non-sync form is the drop-in equivalent.

Same text-patch mechanics as jit_rope/jit_store_cache; every needle applies
idempotently (already-present replacement -> skip that needle, continue).
"""

import logging
import pathlib

LOG = logging.getLogger("hcu.adapter")

_RELPATH = "include/sgl_kernel/warp.cuh"

_NEEDLE = "__hip_bfloat162 _v;"
_LEGACY_REPL = "T _v;  // HCU adapter: receiver type follows T (bf16x2 keeps its packed-bit branch)"
_REPL = "T _v;"

# (needle, replacement): *_sync -> plain, mask dropped. Needles include the
# trailing ';' so the idempotency tag lands AFTER the statement — v1 of this
# patch omitted it and produced "...kWidth)  // tag;" where the comment ate
# the statement terminator ("expected ';' at end of declaration").
_SHFL_MAP = [
    ("__shfl_up_sync(active_mask, val, offset, kWidth);",
     "__shfl_up(val, offset, kWidth);"),
    ("__shfl_down_sync(active_mask, val, offset, kWidth);",
     "__shfl_down(val, offset, kWidth);"),
    ("__shfl_sync(active_mask, value, src_lane, kWidth);",
     "__shfl(value, src_lane, kWidth);"),
]
_SHFL_TAG = "  // HCU adapter: DTK lacks the *_sync variant"


def _apply(root, register):
    """Transform adapter-owned staging files; never pass a community root."""

    import hcu_adapter.registry as _reg

    p = root / _RELPATH
    text = p.read_text(encoding="utf-8")
    changed = False

    # -- patch 1: reduce<T> receiver type --------------------------------
    from hcu_adapter.jit_quant_fp8 import _PATCHES_WARP

    if _LEGACY_REPL in text:
        text = text.replace(_LEGACY_REPL, _REPL)
        changed = True
    known = all(
        needle in text or replacement in text
        or replacement.replace(_REPL, _NEEDLE) in text
        for _path, needle, replacement in _PATCHES_WARP
    )
    if not known:
        raise RuntimeError(
            f"jit_warp: unsupported HIP reduction layout in {_RELPATH} -- "
            "upstream changed; re-derive (governance §3)"
        )
    if all(replacement in text for _path, _needle, replacement in _PATCHES_WARP):
        LOG.warning(
            "HCU adapter: JIT warp reduce patch already installed (%s)",
            _RELPATH,
        )
    else:
        n = text.count(_NEEDLE)
        if n:
            text = text.replace(_NEEDLE, _REPL)
            changed = True
            # One allowlist record per needle (fingerprint is (op, detail); both
            # loop occurrences share the needle, so a single record covers them).
            register(
                "per_token_quant",  # the failing instantiation: fp32 warp max in
                # the indexer k-store; catalog has no warp-primitive entry
                "gfx93x-hipcc-portability",
                target=f"text:sglang/kernels/jit/{_RELPATH}",
                mechanism=_reg.MECH_TEXT,
                detail=_NEEDLE,
            )
            LOG.warning(
                "HCU adapter: JIT warp reduce portability patch installed (%d edits)",
                n,
            )

    # -- patch 2: shfl *_sync -> plain ------------------------------------
    for needle, repl in _SHFL_MAP:
        tagged = repl + _SHFL_TAG
        # v1 tag placement: before the ';' the needle had left behind
        broken = repl[:-1] + _SHFL_TAG + ";"
        if tagged in text:
            continue
        if broken in text:
            text = text.replace(broken, tagged)
            changed = True
            LOG.warning(
                "HCU adapter: jit warp shfl tag placement repaired: %s", repl
            )
            continue
        if needle not in text:
            LOG.info("jit_warp: shfl needle absent (upstream drifted?): %r", needle)
            continue
        text = text.replace(needle, tagged)
        changed = True
        register(
            "per_token_quant",
            "gfx93x-hipcc-portability",
            target=f"text:sglang/kernels/jit/{_RELPATH}",
            mechanism=_reg.MECH_TEXT,
            detail=needle,
        )
        LOG.warning(
            "HCU adapter: JIT warp shfl *_sync->plain mapping installed: %s",
            repl,
        )

    if changed:
        from hcu_adapter.atomic_io import atomic_write

        atomic_write(p, text)


def install():
    """Compatibility entry: prepare the complete immutable JIT input bundle."""
    from .jit_sources import install as install_inputs

    return install_inputs()
