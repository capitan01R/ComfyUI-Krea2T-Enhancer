import math
from typing import Any

import torch

import comfy.patcher_extension


WRAPPER_KEY = "krea2t_prompt_adherence_enhancer"

KREA2_TAP_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
KREA2_TAP_DIM = 2560
KREA2_CHUNK_COUNT = 24
KREA2_CHUNK_DIM = 1280

ENHANCER_PROFILE_12 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)
ENHANCER_CHUNK_PROFILE = ENHANCER_PROFILE_12 + ENHANCER_PROFILE_12
ENHANCER_GLOBAL_MULTIPLIER = 15.0
TXTFUSION_TOKEN_REL_CAP = 0.75


def _is_krea2_dm(dm: Any) -> bool:
    return (
        hasattr(dm, "txtfusion")
        and hasattr(dm, "txtmlp")
        and hasattr(dm, "blocks")
        and hasattr(dm, "_unpack_context")
        and int(getattr(dm, "txtlayers", 0)) == len(KREA2_TAP_LAYERS)
        and int(getattr(dm, "txtdim", 0)) == KREA2_TAP_DIM
    )


def _bounded_float(value, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _step_progress(transformer_options: dict[str, Any]) -> tuple[float, float]:
    sigma = transformer_options.get("sigmas")
    sigma_value = 0.0
    if torch.is_tensor(sigma) and sigma.numel() > 0:
        sigma_value = float(sigma.detach().flatten()[0].float().item())
    elif isinstance(sigma, (int, float)):
        sigma_value = float(sigma)

    sample_sigmas = transformer_options.get("sample_sigmas")
    if torch.is_tensor(sample_sigmas) and sample_sigmas.numel() > 1:
        sig = sample_sigmas.detach().float().flatten()
        idx = int(torch.argmin((sig - sigma_value).abs()).item())
        progress = idx / max(1, int(sig.numel()) - 1)
        return float(progress), sigma_value
    return 0.0, sigma_value


def _rms_scalar(x: torch.Tensor) -> float:
    xf = x.detach().float()
    return float(torch.sqrt(torch.mean(xf * xf)).item())


def _cosine_scalar(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    denom = torch.linalg.vector_norm(af).clamp_min(1e-8) * torch.linalg.vector_norm(bf).clamp_min(1e-8)
    return float(torch.dot(af, bf).div(denom).item())


def _chunk_gains(device: torch.device, dtype: torch.dtype, strength: float) -> torch.Tensor:
    base = torch.tensor(ENHANCER_CHUNK_PROFILE, device=device, dtype=torch.float32)
    gains = 1.0 + float(strength) * (base - 1.0)
    return gains.to(dtype=dtype)


def _run_refiners(txtfusion, y_text, mask=None, transformer_options=None):
    out = y_text
    for block in txtfusion.refiner_blocks:
        out = block(out, mask=mask, transformer_options=transformer_options or {})
    return out


def _run_txtfusion_parts(txtfusion, x, mask=None, transformer_options=None):
    transformer_options = transformer_options or {}
    b, seq, taps, dim = x.shape
    y = x.reshape(b * seq, taps, dim)
    for block in txtfusion.layerwise_blocks:
        y = block(y.contiguous(), mask=None, transformer_options=transformer_options)
    tap_mix = y.reshape(b, seq, taps, dim).permute(0, 1, 3, 2).contiguous()
    projected = txtfusion.projector(tap_mix).squeeze(-1)
    out = _run_refiners(txtfusion, projected, mask=mask, transformer_options=transformer_options)
    return out, projected


def _enhanced_txtfusion_forward(
    txtfusion,
    x,
    mask=None,
    transformer_options=None,
    strength=1.0,
    collect_debug=False,
):
    transformer_options = transformer_options or {}
    b, seq, taps, dim = x.shape
    if taps != len(KREA2_TAP_LAYERS) or dim != KREA2_TAP_DIM:
        out = txtfusion._krea2t_enhancer_original_forward(
            x,
            mask=mask,
            transformer_options=transformer_options,
        )
        return out, None

    reference_out, reference_projected = _run_txtfusion_parts(
        txtfusion,
        x,
        mask=mask,
        transformer_options=transformer_options,
    )

    if strength != 0.0:
        gains = _chunk_gains(x.device, x.dtype, strength)
        global_multiplier = 1.0 + float(strength) * (ENHANCER_GLOBAL_MULTIPLIER - 1.0)
        scaled_x = (
            x.reshape(b, seq, KREA2_CHUNK_COUNT, KREA2_CHUNK_DIM)
            * gains.view(1, 1, KREA2_CHUNK_COUNT, 1)
            * global_multiplier
        ).reshape_as(x)
        candidate_out, candidate_projected = _run_txtfusion_parts(
            txtfusion,
            scaled_x,
            mask=mask,
            transformer_options=transformer_options,
        )
    else:
        global_multiplier = 1.0
        scaled_x = x
        candidate_out = reference_out
        candidate_projected = reference_projected

    post_delta = candidate_out.detach().float() - reference_out.detach().float()
    token_base_rms = torch.sqrt(torch.mean(reference_out.detach().float() ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    token_delta_rms = torch.sqrt(torch.mean(post_delta ** 2, dim=-1, keepdim=True)).clamp_min(1e-8)
    token_rel = token_delta_rms / token_base_rms
    token_scale = (TXTFUSION_TOKEN_REL_CAP / token_rel).clamp(max=1.0)
    out = (reference_out.detach().float() + post_delta * token_scale).to(candidate_out.dtype)

    debug = None
    if collect_debug:
        ref_rms = _rms_scalar(reference_projected)
        raw_rms = _rms_scalar(candidate_projected)
        out_delta = out.detach().float() - reference_out.detach().float()
        post_base_rms = _rms_scalar(reference_out)
        debug = {
            "shape": "x".join(str(int(v)) for v in out.shape),
            "global_multiplier": float(global_multiplier),
            "projector_rms_ratio": float(raw_rms / max(ref_rms, 1e-8)),
            "output_rel_delta": float(_rms_scalar(out_delta) / max(post_base_rms, 1e-8)),
            "output_cosine": _cosine_scalar(reference_out, out),
            "clamp_mean": float(token_scale.mean().item()),
            "token_raw_rel_mean": float(token_rel.mean().item()),
            "input_tap_rms": [
                float(v) for v in torch.sqrt(torch.mean(x.detach().float() ** 2, dim=(0, 1, 3))).detach().cpu().tolist()
            ],
            "scaled_tap_rms": [
                float(v) for v in torch.sqrt(torch.mean(scaled_x.detach().float() ** 2, dim=(0, 1, 3))).detach().cpu().tolist()
            ],
        }

    return out, debug


def krea2t_enhancer_wrapper(executor, x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
    transformer_options = transformer_options or {}
    cfg = transformer_options.get("krea2t_prompt_adherence_enhancer", {})
    if not cfg or not cfg.get("enabled", True):
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    if cfg.get("_active", False):
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    dm = executor.class_obj
    if not _is_krea2_dm(dm):
        if cfg.get("debug", False):
            print("[Krea2TEnhancer] skipped: diffusion model does not match Krea2 text-fusion layout")
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    strength = _bounded_float(cfg.get("strength", 1.0), 1.0, 0.0, 2.0)
    if strength == 0.0:
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    txtfusion = dm.txtfusion
    if hasattr(txtfusion, "_krea2t_enhancer_original_forward"):
        txtfusion.forward = txtfusion._krea2t_enhancer_original_forward
        delattr(txtfusion, "_krea2t_enhancer_original_forward")
    original_forward = txtfusion.forward
    progress, sigma = _step_progress(transformer_options)
    debug_enabled = bool(cfg.get("debug", False))

    def enhanced_forward(x_in, mask=None, transformer_options=None):
        txtfusion._krea2t_enhancer_original_forward = original_forward
        try:
            after, debug = _enhanced_txtfusion_forward(
                txtfusion,
                x_in,
                mask=mask,
                transformer_options=transformer_options or {},
                strength=strength,
                collect_debug=debug_enabled,
            )
        finally:
            if hasattr(txtfusion, "_krea2t_enhancer_original_forward"):
                delattr(txtfusion, "_krea2t_enhancer_original_forward")

        if debug is not None and int(cfg.setdefault("_debug_prints", 0)) < int(cfg.get("max_debug_prints", 8)):
            cfg["_debug_prints"] = int(cfg.get("_debug_prints", 0)) + 1
            print(
                "[Krea2TEnhancer] "
                f"strength={strength:.3f} progress={progress:.3f} sigma={sigma:.6g} "
                f"global={debug['global_multiplier']:.6g} proj_ratio={debug['projector_rms_ratio']:.6g} "
                f"out_rel={debug['output_rel_delta']:.6g} out_cos={debug['output_cosine']:.6g} "
                f"clamp={debug['clamp_mean']:.6g}"
            )
        return after

    try:
        cfg["_active"] = True
        txtfusion.forward = enhanced_forward
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)
    finally:
        cfg["_active"] = False
        txtfusion.forward = original_forward


class ComfyUIKrea2TEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/krea2"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def apply(self, model, enabled=True, strength=1.0, debug=False):
        patched = model.clone()
        strength = _bounded_float(strength, 1.0, 0.0, 2.0)
        to = patched.model_options.setdefault("transformer_options", {})
        to["krea2t_prompt_adherence_enhancer"] = {
            "enabled": bool(enabled),
            "strength": strength,
            "debug": bool(debug),
            "max_debug_prints": 8,
        }
        if hasattr(patched, "remove_wrappers_with_key"):
            patched.remove_wrappers_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                WRAPPER_KEY,
            )
        wrappers = to.get("wrappers", {})
        diffusion_wrappers = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
        diffusion_wrappers.pop(WRAPPER_KEY, None)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            krea2t_enhancer_wrapper,
        )
        if debug:
            print(f"[Krea2TEnhancer] attached enabled={bool(enabled)} strength={strength:.3f}")
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "ComfyUI-Krea2T-Enhancer": ComfyUIKrea2TEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyUI-Krea2T-Enhancer": "ComfyUI-Krea2T-Enhancer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
