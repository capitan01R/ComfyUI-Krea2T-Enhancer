# ComfyUI-Krea2T-Enhancer

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)

Prompt-adherence enhancement for Krea2 diffusion models in ComfyUI.

The enhancer nodes patch Krea2's text-fusion path during sampling. The package also includes phrase weighting, a Turbo sigma scheduler, and an image-only character LoRA loader.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer.git
```

Restart ComfyUI after installing or updating.

No extra Python packages are required beyond a working ComfyUI Krea2 setup.

## Included Nodes

| Node | Output | Purpose |
|---|---|---|
| **ComfyUI-Krea2T-Enhancer** | `MODEL` | Patches the Krea2 model path to improve prompt adherence during sampling. |
| **Krea2T Enhancer Advanced** | `MODEL` | Same enhancer path, plus a direct post-`txtmlp` `text_scale` control for fused text-token strength. |
| **Krea2 Turbo Reference Sigmas (From Latent)** | `SIGMAS`, `LATENT` | Builds a Turbo sigma schedule based on the official Krea 2 Turbo scheduler settings and validates the connected latent dimensions. |
| **Krea2 Text Encode — Attention-Weighted Phrases** | `MODEL`, `CONDITIONING`, `STRING` | Encodes weighted phrases and changes only the image-query-to-selected-text-key attention odds in Krea2's shared DiT blocks. |
| **Krea2T Character LoRA — Image Only** | `MODEL`, `STRING` | Applies a character LoRA directly to image tokens while skipping unsupported weights. |
| **Krea2 Selective LoRA Loader Model Only (Projection Split)** | `MODEL`, `STRING` | Alternative model-only loader with switches for the Krea2 projector, remaining TextFusion path, and external text MLP. |
| **Krea2 · Projector + External MLP · LoRA Isolation** | `MODEL`, `STRING` | Applies only selected saved LoRA updates for the projector and the two external text-MLP linear layers. |

## Character LoRA — Image Only

### Why use this loader?

A character LoRA can change more than likeness. Adding one to an otherwise
working setup can make requested details less reliable or change how other
LoRAs behave. This loader is designed to reduce that interference while
keeping the character's influence on the image.

Krea2 processes the prompt and image together through shared layers. With a
regular LoRA loader, character updates to those layers apply to both the image
and the prompt's internal representation. Excluding weights for dedicated
text-processing layers still leaves this shared route active.

This loader filters the supported weights and applies the character updates
only to image tokens. That removes their direct contribution to text tokens
while preserving other connected LoRAs. It targets one source of overlap;
normal interaction between text and image remains, so results still depend
on the character LoRA and the rest of the workflow.

### Setup

Use this node in place of the regular LoRA loader for your character. Select
the character LoRA or LoKr in `lora_name` and adjust `strength_model` to control its
influence.

```text
Krea2 model -> other LoRA loaders (optional) -> Krea2T Character LoRA — Image Only -> sampler
```

Other LoRAs can stay connected as usual. Load each character LoRA only once.
If using the phrase encoder, place it after this loader and connect its
MODEL and CONDITIONING outputs to the sampler.

| Input/output | Meaning |
|---|---|
| `model` | Krea2 model from your model loader or another LoRA loader. |
| `lora_name` | Your character LoRA file. |
| `strength_model` | Character LoRA strength. Default `1.0`; `0` turns its effect off. Negative values are supported. |
| `enabled` | Bypasses this loader when off. |
| `report` | Optional text output showing loaded/skipped weights, adapter types, and saved scaling. |

### How it works

Image tokens represent the image being generated; text tokens represent the
prompt. This loader applies the character LoRA's updates directly to image
tokens, including reference-image tokens, while leaving text tokens without
that direct update. Text and image still interact through the model's normal
attention, so this does not completely isolate their influence.

Unsupported weights are skipped automatically. If the LoRA has no compatible
weights, the model passes through unchanged and the report shows zero loaded
weights. The loader preserves other LoRAs already applied to the model and
adds no extra attention or sampling pass.

Requires an up-to-date ComfyUI with native Krea2 support. No core-file edits or
extra Python packages are needed. Supports Nodes 2.0.

<details>
<summary>Supported adapter formats</summary>

### LoRA and LoKr

The loader supports standard LoRA and LoKr exports using ComfyUI's Krea2 key
mapping. Compatibility depends on the saved format, not the trainer name.
LoRA pairs can use A/B, up/down, Diffusers, or PEFT naming. LoKr supports
direct and decomposed factors.

Saved alpha/rank scaling is applied before `strength_model`. For example,
rank 16 with alpha 1 applies `1/16` of the raw LoRA update at strength `1.0`.
Files without alpha retain their unscaled update. LoKr follows ComfyUI's
scaling rules for its factor layout.

Only the attention and MLP projections in Krea2's 28 shared image/text blocks
are eligible: `attn.wq`, `attn.wk`, `attn.wv`, `attn.wo`, `attn.gate`,
`mlp.gate`, `mlp.up`, and `mlp.down`. This covers up to 224 matrix updates.
Weights targeting dedicated text processing, normalization, or other layers
are skipped, including when the file uses alternate key names.

Incomplete adapters and unsupported formats such as DoRA and LoHa are skipped.
Matched adapters must have dimensions compatible with the model. LoKr runs
from its factors without allocating a full-sized Kronecker weight matrix.

If an older version produced excessive noise with a LoRA that stores alpha,
update and restart ComfyUI: previous versions ignored that scaling and could
apply the adapter too strongly. Start again at your intended strength rather
than a value chosen to compensate for that bug.

For a manual update, replace both `character_lora_image_only/__init__.py` and
`character_lora_image_only/runtime.py`, then restart ComfyUI. Updating only one
file leaves part of the old loader installed. The report's
`adapter_scale_range` shows the saved scaling before your strength setting.

</details>


## Selective LoRA Loader — Model Only

This is an extra utility for LoRAs that do not work well with the
**Character LoRA — Image Only** loader. It is not the same token-level image
isolation method: instead, it uses ComfyUI's normal model-only LoRA loading and
lets you choose whether the Krea2 text-conditioning branches are included.

Use `load_projection`, `load_textfusion`, and `load_txtmlp` to independently
keep or remove the `txtfusion.projector`, the remaining `txtfusion` layers, and
the external `txtmlp` updates. All other LoRA tensors remain eligible to load.
The `report` output shows how many tensors were found and kept in each branch.

The paired **Projector + External MLP · LoRA Isolation** node is a small testing
utility for applying only the saved projector and external text-MLP deltas from
a reference adapter.

## Usage

Place **ComfyUI-Krea2T-Enhancer** between your Krea2 diffusion model loader and sampler:

```text
Load Diffusion Model -> ComfyUI-Krea2T-Enhancer -> KSampler
```

Or use **Krea2T Enhancer Advanced** when you want the additional text-scale control:

```text
Load Diffusion Model -> Krea2T Enhancer Advanced -> KSampler
```

Use your normal Krea2 text encoder, VAE, latent, and sampler setup.

For the sigma scheduler, connect both the loaded Krea2 Turbo model and the same
Empty Latent Image that will be sent to the sampler:

```text
Load Diffusion Model --\
                        > Krea2 Turbo Reference Sigmas (From Latent) -> SIGMAS to sampler
Empty Latent Image ----/                                             -> LATENT to sampler
```

For attention-weighted phrases, connect the final model after all LoRA loaders
and the Krea2 CLIP to the node. Both primary outputs must be used:

```text
Load Diffusion Model -> LoRA loader(s) -> Krea2 Text Encode — Attention-Weighted Phrases -> MODEL to sampler
Krea2 CLIP ---------------------------> Krea2 Text Encode — Attention-Weighted Phrases -> CONDITIONING to positive
```

Write a weighted section as `(phrase:weight)`. The annotation is removed before
tokenization, while the phrase and its original Qwen token positions remain.
`1.0` is an exact no-op, values above `1.0` increase the phrase's attention odds,
values between `0.0` and `1.0` reduce them, and `0.0` suppresses them.

## Controls

### ComfyUI-Krea2T-Enhancer

| Parameter | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Turns the patch on or off. |
| `strength` | `1.0` | Blends the enhancement from neutral `0.0` to full `2.0`. |
| `debug` | `false` | Prints concise runtime diagnostics to the ComfyUI console. |

### Krea2T Enhancer Advanced

| Parameter | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Turns the patch on or off. |
| `strength` | `1.0` | Same enhancer strength as the original node, from neutral `0.0` to full `2.0`. |
| `text_scale` | `1.0` | Multiplies fused text tokens immediately after `txtmlp`, before they enter the shared Krea2 stream. |
| `debug` | `false` | Prints concise runtime diagnostics to the ComfyUI console. |

Suggested starting range for `text_scale` is `1.50` to `2.00`. The neutral value is `1.0`.

### Krea2 Turbo Reference Sigmas (From Latent)

| Parameter | Default | Meaning |
|---|---:|---|
| `model` | — | The loaded Krea2 Turbo diffusion model. |
| `latent` | — | The same latent used for sampling; it is validated and passed through unchanged. |
| `steps` | `8` | Number of Euler denoising steps. The reference Turbo setup uses eight. |
| `denoise` | `1.0` | Uses the complete schedule at `1.0`; lower values retain the final requested steps from a longer schedule. |

### Krea2 Text Encode — Attention-Weighted Phrases

| Parameter | Meaning |
|---|---|
| `model` | The final Krea2 model chain that will be sent to the sampler, including any LoRAs. |
| `clip` | A text encoder loaded with the Krea2 CLIP type. |
| `text` | Literal prompt text with optional `(phrase:weight)` sections. |

#### Why this node exists

Krea2 does not consume a conventional single-layer CLIP embedding. Its text
encoder supplies twelve selected Qwen hidden-state taps, producing a
`12 x 2560` representation for every text-token position. Krea2 then processes
that stack through its internal text-fusion path before the text and image
tokens enter the shared DiT blocks.

Conventional prompt-emphasis methods usually multiply a completed conditioning
row or repeat a token. Those operations do not map cleanly onto this pipeline:
uniform row scaling can be reduced by later normalization, while repetition
changes sequence length and can make one term overwhelm the relationships in a
long prompt.

This node keeps the original prompt sequence intact. It encodes the clean text
normally, locates every Qwen token row belonging to each weighted phrase, and
changes how strongly image queries attend to those selected text keys inside
Krea2's shared DiT attention. It does not copy, delete, average, or rescale the
conditioning rows.

For a phrase weight `w`, the node adds `log(w)` to the selected image-to-text
attention logits. After softmax, this multiplies the selected phrase's attention
odds by `w` relative to their original values. A weight is therefore an
attention-priority control, not a promise that an object will become a literal
multiple larger, more frequent, or more visible in the final image.

#### Why it has both MODEL and CLIP inputs

The CLIP input is used to tokenize and encode the annotation-free prompt into
the normal Krea2 twelve-tap conditioning tensor. The MODEL input is used to
apply the matching attention-odds operation to the exact text-row positions
identified during that encoding. This is why the node produces a paired MODEL
and CONDITIONING result rather than acting as only a text encoder or only a
model patch.

Connect the completed model chain after all desired LoRA loaders to `model`.
Connect the Krea2 text encoder to `clip`. Send the node's MODEL output to the
sampler and its CONDITIONING output to the sampler's positive-conditioning
path. The same prompt supplies both outputs, keeping the phrase-to-row mapping
aligned with the model-side attention operation.

#### Phrase syntax

Use parentheses around any complete word or multi-word phrase followed by a
colon and a non-negative numeric weight:

```text
A scene containing a (primary subject:2.0) beside a (secondary object:0.6)
```

The node removes only the surrounding weight annotation before tokenization.
The words, spaces, tokenizer pieces, token order, and token count remain those
of the clean prompt. If a phrase becomes several Qwen tokenizer pieces, the
same weight is assigned to every piece belonging to that phrase.

Weighted sections must not overlap or contain another weighted section. More
than one separate phrase can be weighted in the same prompt.

#### Weight behavior

| Weight | Effect |
|---:|---|
| `1.0` | Exact neutral value. The phrase receives its original attention odds. |
| Above `1.0` | Gives the phrase more attention priority. |
| Between `0.0` and `1.0` | Reduces the phrase's attention priority. |
| `0.0` | Applies the node's strongest suppression to the selected phrase keys. |

Weights are relative odds multipliers. For example, `2.0` gives the selected
keys twice their original odds before softmax renormalizes all available keys;
it does not guarantee twice the visible effect. Very large weights can cause a
phrase to compete too strongly with composition, spatial relationships, or
other requested details.

#### Practical usage guide

1. Build and test the complete Krea2 workflow first, including the LoRAs and
   sampler settings you intend to use.
2. Keep the seed, resolution, sigmas, sampler, prompt, and LoRA strengths fixed
   while evaluating a phrase weight.
3. Start with the ordinary unannotated prompt or annotate the phrase with
   `1.0` to establish the neutral result.
4. Add weight only to the exact phrase that needs more or less priority. Include
   the complete relationship when the relationship matters instead of weighting
   only one isolated noun.
5. Begin with a moderate increase such as `1.5` or `2.0`. Raise it in deliberate
   increments if the phrase still receives insufficient attention. Use values
   below `1.0` when a phrase is dominating the result.
6. If several prompt sections need adjustment, tune one phrase at a time before
   combining the weights. This makes composition changes attributable to a
   specific phrase instead of several simultaneous changes.
7. Recheck the result across additional seeds only after finding a useful range
   on the fixed comparison seed. Phrase weighting changes attention allocation,
   so its visible strength can vary with the generated composition.

The `STRING` output is an inspection report. It records the clean text, every
weighted phrase, its numeric weight, and the exact Qwen token rows, token IDs,
and decoded pieces selected for that phrase. It can be connected to a text
preview node when the precise tokenizer mapping needs to be verified; it is not
required by the sampler.

This node is specifically validated for text-only Krea2 conditioning with the
`12 x 2560` layout. It rejects a mismatched text encoder, visual or custom
embedding tokens, changed token counts, and a MODEL that does not expose the
expected Krea2 text-fusion and shared-block architecture instead of silently
applying an uncertain mapping.

## Notes

- Designed for Krea2 models using the `12 x 2560` Krea2 text-conditioning layout.
- The original and advanced enhancer nodes return only a patched `MODEL`; they do not modify prompt text or require extra conditioning nodes.
- The attention-weighted phrase node must supply both the sampler's `MODEL` and positive `CONDITIONING` paths. It never copies, deletes, averages, or scales conditioning rows.
- If the loaded model does not match the expected Krea2 text-fusion layout, the patch is skipped.
- The advanced node restores every temporary runtime patch after each model call and does not store debug counters or step-local state in the model config. With the same seed and the same node parameters, ComfyUI can reuse cached graph results normally.
- The reference sigma node uses the Turbo fixed timestep shift `mu=1.15`, based on the official Krea 2 Turbo scheduler settings. It validates that the connected image dimensions are divisible by 16. Turbo does not use the RAW checkpoint's resolution-dependent shift rule.
