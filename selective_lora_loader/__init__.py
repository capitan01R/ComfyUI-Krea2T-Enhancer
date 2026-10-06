import os

import comfy.sd
import comfy.utils
import folder_paths

from .isolation import Krea2TextPathDeltaIsolation


BRANCH_PROJECTION = "projection"
BRANCH_TEXTFUSION = "textfusion"
BRANCH_TXTMLP = "txtmlp"
BRANCH_OTHER = "other"
BRANCH_ORDER = (
    BRANCH_PROJECTION,
    BRANCH_TEXTFUSION,
    BRANCH_TXTMLP,
    BRANCH_OTHER,
)


def _normalized_key(key):
    return "." + key.lower().replace("_", ".").strip(".") + "."


def _contains_any(key, module_names):
    normalized = _normalized_key(key)
    return any(f".{module_name}." in normalized for module_name in module_names)


def _classify_key(key):
    if _contains_any(
        key,
        (
            "txtfusion.projector",
            "textfusion.projector",
            "text.fusion.projector",
        ),
    ):
        return BRANCH_PROJECTION

    if _contains_any(key, ("txtfusion", "textfusion", "text.fusion")):
        return BRANCH_TEXTFUSION

    if _contains_any(
        key,
        (
            "txtmlp",
            "txt.mlp",
            "txt.in.linear.1",
            "txt.in.linear.2",
        ),
    ):
        return BRANCH_TXTMLP

    return BRANCH_OTHER


def _branch_counts(lora):
    counts = {branch: 0 for branch in BRANCH_ORDER}
    for key in lora:
        counts[_classify_key(key)] += 1
    return counts


def _filter_lora(lora, load_projection, load_textfusion, load_txtmlp):
    enabled = {
        BRANCH_PROJECTION: bool(load_projection),
        BRANCH_TEXTFUSION: bool(load_textfusion),
        BRANCH_TXTMLP: bool(load_txtmlp),
        BRANCH_OTHER: True,
    }
    kept = {
        key: value
        for key, value in lora.items()
        if enabled[_classify_key(key)]
    }
    return kept, _branch_counts(lora), _branch_counts(kept)


def _file_signature(lora_name):
    path = folder_paths.get_full_path_or_raise("loras", lora_name)
    stat = os.stat(path)
    return (
        path,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _format_counts(counts):
    return ",".join(f"{branch}:{counts[branch]}" for branch in BRANCH_ORDER)


class Krea2SelectiveLoraLoaderModelOnlyProjection:
    def __init__(self):
        self._cache_signature = None
        self._cache_lora = None
        self._cache_metadata = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "LoRA applied only to the diffusion model."},
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "load_projection": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Load only txtfusion.projector LoRA tensors.",
                    },
                ),
                "load_textfusion": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Load txtfusion layerwise/refiner tensors, excluding the projector.",
                    },
                ),
                "load_txtmlp": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Load txtmlp linear-layer LoRA tensors.",
                    },
                ),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_lora_model_only"
    CATEGORY = "Krea2/loaders"
    DESCRIPTION = (
        "Loads a model-only LoRA with independent switches for the TextFusion "
        "projector, the remaining TextFusion blocks, and txtmlp."
    )

    @classmethod
    def IS_CHANGED(
        cls,
        model,
        lora_name,
        strength_model,
        load_projection,
        load_textfusion,
        load_txtmlp,
        debug,
    ):
        return _file_signature(lora_name)

    def _load_raw_lora(self, lora_name):
        signature = _file_signature(lora_name)
        if self._cache_signature != signature:
            lora, metadata = comfy.utils.load_torch_file(
                signature[0],
                safe_load=True,
                return_metadata=True,
            )
            self._cache_signature = signature
            self._cache_lora = lora
            self._cache_metadata = metadata
        return signature[0], self._cache_lora, self._cache_metadata

    def load_lora_model_only(
        self,
        model,
        lora_name,
        strength_model,
        load_projection,
        load_textfusion,
        load_txtmlp,
        debug,
    ):
        if strength_model == 0:
            report = "Krea2 Projection-Split LoRA: bypassed because strength_model=0"
            if debug:
                print(report)
            return model, report

        lora_path, raw_lora, metadata = self._load_raw_lora(lora_name)
        filtered_lora, source_counts, kept_counts = _filter_lora(
            raw_lora,
            load_projection=load_projection,
            load_textfusion=load_textfusion,
            load_txtmlp=load_txtmlp,
        )

        if filtered_lora:
            model_lora, _ = comfy.sd.load_lora_for_models(
                model,
                None,
                filtered_lora,
                strength_model,
                0.0,
                lora_metadata=metadata,
            )
        else:
            model_lora = model

        report = (
            "Krea2 Projection-Split LoRA: "
            f"lora={lora_name}; "
            f"kept={len(filtered_lora)}/{len(raw_lora)}; "
            f"source=[{_format_counts(source_counts)}]; "
            f"loaded=[{_format_counts(kept_counts)}]; "
            f"load_projection={bool(load_projection)}; "
            f"load_textfusion={bool(load_textfusion)}; "
            f"load_txtmlp={bool(load_txtmlp)}; "
            f"strength_model={strength_model:.4g}; "
            f"path={lora_path}"
        )
        if debug:
            print(report)

        return model_lora, report


NODE_CLASS_MAPPINGS = {
    "Krea2SelectiveLoraLoaderModelOnlyProjection": Krea2SelectiveLoraLoaderModelOnlyProjection,
    "Krea2TextPathDeltaIsolation": Krea2TextPathDeltaIsolation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2TextPathDeltaIsolation": "Krea2 · Projector + External MLP · LoRA Isolation",
    "Krea2SelectiveLoraLoaderModelOnlyProjection": (
        "Krea2 Selective LoRA Loader Model Only (Projection Split)"
    ),
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
