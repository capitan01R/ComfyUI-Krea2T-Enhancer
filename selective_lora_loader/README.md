# Krea2 Selective LoRA Loader - Projection Split

Separate model-only LoRA controls for the three Krea2 text-conditioning branches:

- `load_projection`: `txtfusion.projector` only.
- `load_textfusion`: all other `txtfusion` tensors, including layerwise and refiner blocks.
- `load_txtmlp`: the linear layers under `txtmlp`.

All non-text-path tensors, including the 28 shared DiT blocks, remain loaded.

The `report` output lists source and loaded tensor counts for every branch. The LoRA file cache is invalidated when the file identity, size, or timestamps change; unchanged node inputs remain eligible for normal ComfyUI execution caching.
