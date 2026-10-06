import json
import logging
import math
import os
from sys import float_info

from safetensors import safe_open

import comfy.lora
import folder_paths


TARGETS = {
    'projector_mix': 'diffusion_model.txtfusion.projector',
    'before_gelu': 'diffusion_model.txtmlp.1',
    'after_gelu': 'diffusion_model.txtmlp.3',
}
FACTOR_PAIRS = (
    ('lora_up.weight', 'lora_down.weight'),
    ('lora_B.weight', 'lora_A.weight'),
)


def load_selected_updates(path, selected, model_weights):
    state = {}
    key_map = {}
    with safe_open(path, framework='pt', device='cpu') as source:
        keys = set(source.keys())
        for control in selected:
            prefix = TARGETS[control]
            weight_key = prefix + '.weight'
            if weight_key not in model_weights:
                raise ValueError(f'Krea2 path isolation requires {weight_key} on the input model.')
            pairs = [(prefix + '.' + up, prefix + '.' + down) for up, down in FACTOR_PAIRS
                     if prefix + '.' + up in keys and prefix + '.' + down in keys]
            if len(pairs) != 1:
                raise ValueError(f'{control}: the source needs exactly one LoRA factor pair for {prefix}.')
            up_key, down_key = pairs[0]
            alpha_key = prefix + '.alpha'
            target_keys = {key for key in keys if key.startswith(prefix + '.')}
            unsupported = target_keys - {up_key, down_key, alpha_key}
            if unsupported:
                raise ValueError(f'{control}: unsupported extra tensors at this target: {sorted(unsupported)}')
            up_shape = source.get_slice(up_key).get_shape()
            down_shape = source.get_slice(down_key).get_shape()
            if (len(up_shape) != 2 or len(down_shape) != 2 or up_shape[1] != down_shape[0]
                    or (up_shape[0], down_shape[1]) != tuple(model_weights[weight_key].shape)):
                raise ValueError(f'{control}: the source factors do not match the input model weight shape.')
            state[up_key] = source.get_tensor(up_key)
            state[down_key] = source.get_tensor(down_key)
            if alpha_key in keys:
                alpha = source.get_tensor(alpha_key)
                if alpha.numel() != 1 or not math.isfinite(alpha.item()):
                    raise ValueError(f'{control}: alpha must be one finite number.')
                state[alpha_key] = alpha
            key_map[prefix] = weight_key
    patches = comfy.lora.load_lora(state, key_map)
    if set(patches) != set(key_map.values()):
        raise ValueError('The source did not produce exactly the requested Krea2 weight updates.')
    return patches


def update_input(default, meaning):
    return ('FLOAT', {
        'default': default, 'min': -float_info.max, 'max': float_info.max, 'step': 0.05,
        'tooltip': meaning + ' 0 = no added update; 1 = the saved adapter update; '
                   '0.5 = half that update; -1 = subtract it. This does not scale the whole layer output.',
    })


class Krea2TextPathDeltaIsolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL', {'tooltip': 'Use the base Krea2 MODEL output for an isolated test. Existing input patches are retained.'}),
            'source_lora': ([name for name in folder_paths.get_filename_list('loras')
                             if name.lower().endswith(('.safetensors', '.sft'))], {
                'tooltip': 'Reference adapter. Only the three named weight updates are read. Its attention, internal MLP, norm, and other updates are excluded.',
            }),
            'projector_mix': update_input(1.0, 'Changes how the 12 text taps are combined between LW1 and R0.'),
            'before_gelu': update_input(1.0, 'Changes external MLP layer 1: the linear transformation before GELU, after R1.'),
            'after_gelu': update_input(1.0, 'Changes external MLP layer 3: the linear transformation after GELU, before the DiT.'),
            'enabled': ('BOOLEAN', {'default': True, 'tooltip': 'Off returns the input model unchanged. Useful for a baseline with the same seed.'}),
        }}

    RETURN_TYPES = ('MODEL', 'STRING')
    RETURN_NAMES = ('model', 'configuration')
    FUNCTION = 'apply'
    CATEGORY = 'Krea2/loaders'
    DESCRIPTION = (
        'Isolates saved LoRA weight updates for the 12-tap projector and the two external '
        'text MLP linear layers. One model input and output; each control is independent. '
        'Uses native weight patching. No activation sign selection or extra model passes.'
    )

    @classmethod
    def IS_CHANGED(cls, model, source_lora, projector_mix, before_gelu, after_gelu, enabled):
        if not enabled or not any((projector_mix, before_gelu, after_gelu)):
            return 'bypassed'
        path = folder_paths.get_full_path_or_raise('loras', source_lora)
        stat = os.stat(path)
        return path, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def apply(self, model, source_lora, projector_mix, before_gelu, after_gelu, enabled):
        strengths = dict(zip(TARGETS, (projector_mix, before_gelu, after_gelu)))
        if not all(math.isfinite(value) for value in strengths.values()):
            raise ValueError('All three update amounts must be finite numbers.')
        selected = {name: strength for name, strength in strengths.items() if enabled and strength != 0}
        report = {
            'source_lora': source_lora,
            'status': 'active' if selected else 'bypassed',
            'meaning': 'input weight + amount * saved adapter update',
            'applied': {TARGETS[name] + '.weight': strength for name, strength in selected.items()},
            'existing_input_patches': 'retained',
        }
        if not selected:
            return model, json.dumps(report, indent=2)
        path = folder_paths.get_full_path_or_raise('loras', source_lora)
        if not path.lower().endswith(('.safetensors', '.sft')):
            raise ValueError('Choose a safetensors reference adapter.')
        model_weights = model.model_state_dict(filter_prefix='diffusion_model.txt')
        patches = load_selected_updates(path, selected, model_weights)
        patched = model.clone()
        for name, strength in selected.items():
            key = TARGETS[name] + '.weight'
            if set(patched.add_patches({key: patches[key]}, strength_patch=strength)) != {key}:
                raise ValueError(f'The input model did not accept the {name} update.')
        logging.info('Krea2 path isolation | %s | %d weight updates from %s',
                     ', '.join(f'{name}={value:g}' for name, value in selected.items()),
                     len(selected), source_lora)
        return patched, json.dumps(report, indent=2)
