from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from ..core.utils import get_device


def get_decoder_layers(model):
    m = model
    for attr in ("base_model", "model"):
        if hasattr(m, attr):
            candidate = getattr(m, attr)
            if not hasattr(candidate, "layers"):
                m = candidate
            else:
                m = candidate
                break
    if not hasattr(m, "layers"):
        for name, module in model.named_modules():
            if hasattr(module, "layers") and len(list(module.layers)) > 0:
                m = module
                break
    return m.layers


class SteeringVectorExtractor:
    def __init__(self, model, tokenizer, layers: List[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self._hooks = []
        self._cache: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
        self._decoder_layers = get_decoder_layers(model)

    def _register_hooks(self):
        for layer_idx in self.layers:
            layer = self._decoder_layers[layer_idx]

            def make_capture_hook(li):
                def hook(module, inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._cache[li].append(hidden[0, -1, :].detach().cpu().float())
                    return output

                return hook

            self._hooks.append(
                layer.register_forward_hook(make_capture_hook(layer_idx))
            )

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def collect_activations(self, texts: List[str]) -> Dict[int, torch.Tensor]:
        self._cache = {l: [] for l in self.layers}
        self._register_hooks()
        for text in tqdm(texts, desc="  Extracting activations", leave=False):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to(self.model.device)
            self.model(**inputs)
        self._remove_hooks()
        return {l: torch.stack(self._cache[l]).mean(dim=0) for l in self.layers}


class SteeringHook:
    def __init__(
        self,
        model,
        harm_vectors,
        reasoning_vectors,
        harm_alpha=12.0,
        reasoning_alpha=-10.0,
    ):
        self.model = model
        self.harm_vectors = harm_vectors
        self.reasoning_vectors = reasoning_vectors
        self.harm_alpha = harm_alpha
        self.reasoning_alpha = reasoning_alpha
        self._hooks = []
        self._decoder_layers = get_decoder_layers(model)

    def __enter__(self):
        for layer_idx, layer in enumerate(self._decoder_layers):
            if layer_idx not in self.harm_vectors:
                continue
            hv = self.harm_vectors[layer_idx].to(self.model.device)
            rv = self.reasoning_vectors[layer_idx].to(self.model.device)
            ha, ra = self.harm_alpha, self.reasoning_alpha

            def make_hook(h_vec, r_vec, h_alpha, r_alpha):
                def hook(module, inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    delta = h_alpha * h_vec + r_alpha * r_vec
                    hidden = hidden + delta.unsqueeze(0).unsqueeze(0)
                    return (
                        (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                    )

                return hook

            self._hooks.append(layer.register_forward_hook(make_hook(hv, rv, ha, ra)))
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
