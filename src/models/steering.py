import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from unsloth import FastLanguageModel

from ..utils.constants import DEVICE, MAX_NEW_TOKENS, MAX_SEQ_LEN, STEERING_LAYERS
from ..utils.helpers import set_seed

set_seed()


class SteeringVectorExtractor:
    def __init__(self, model, tokenizer, layers):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self._hooks = []
        self._cache = {l: [] for l in layers}
        self._decoder_layers = self._get_decoder_layers()

    def _get_decoder_layers(self):
        m = self.model
        for attr in ["base_model", "model"]:
            if hasattr(m, attr):
                m = getattr(m, attr)
        if hasattr(m, "layers"):
            return m.layers
        raise AttributeError("Could not find .layers in model")

    def _register_hooks(self):
        for layer_idx in self.layers:
            layer = self._decoder_layers[layer_idx]

            def hook_fn(li):
                def hook(module, inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self._cache[li].append(hidden[0, -1, :].detach().cpu().float())
                    return output

                return hook

            h = layer.register_forward_hook(hook_fn(layer_idx))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def _collect(self, texts):
        self._cache = {l: [] for l in self.layers}
        self._register_hooks()
        for text in tqdm(texts, desc="Extracting activations", leave=False):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
            ).to(DEVICE)
            self.model(**inputs)
        self._remove_hooks()
        result = {}
        for l in self.layers:
            if len(self._cache[l]) == 0:
                raise RuntimeError(f"No activations for layer {l}")
            result[l] = torch.stack(self._cache[l]).mean(dim=0)
        return result

    def compute_contrastive_vector(self, positive_texts, negative_texts):
        pos = self._collect(positive_texts)
        neg = self._collect(negative_texts)
        vectors = {}
        for l in self.layers:
            diff = pos[l] - neg[l]
            vectors[l] = diff / (diff.norm() + 1e-8)
        return vectors


class SteeringHook:
    def __init__(
        self, model, harm_vectors, reasoning_vectors, harm_alpha, reasoning_alpha
    ):
        self.model = model
        self.harm_vectors = harm_vectors
        self.reasoning_vectors = reasoning_vectors
        self.harm_alpha = harm_alpha
        self.reasoning_alpha = reasoning_alpha
        self._hooks = []
        self._decoder_layers = self._get_decoder_layers()

    def _get_decoder_layers(self):
        m = self.model
        for attr in ["base_model", "model"]:
            if hasattr(m, attr):
                m = getattr(m, attr)
        if hasattr(m, "layers"):
            return m.layers
        raise AttributeError("Could not find .layers in model")

    def __enter__(self):
        for layer_idx, layer in enumerate(self._decoder_layers):
            if layer_idx not in self.harm_vectors:
                continue
            hv = self.harm_vectors[layer_idx].to(DEVICE)
            rv = self.reasoning_vectors[layer_idx].to(DEVICE)
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

            h = layer.register_forward_hook(make_hook(hv, rv, ha, ra))
            self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def run_steering_experiment(adapter_path, harm_alpha, reasoning_alpha, dataset_df):
    # Load base + adapter
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    FastLanguageModel.for_inference(model)

    # Extract vectors (use PKU dataset)
    pku_df = pd.read_csv("data/pku/pku-beavertails.csv")
    harmful_texts = pku_df[pku_df["label"] == "UNSAFE"]["prompt"].tolist()[:150]
    safe_texts = pku_df[pku_df["label"] == "SAFE"]["prompt"].tolist()[:150]
    extractor = SteeringVectorExtractor(model, tokenizer, STEERING_LAYERS)
    harm_vectors = extractor.compute_contrastive_vector(harmful_texts, safe_texts)
    reasoning_vectors = harm_vectors  # same direction, different alpha

    # Run inference on dataset with steering
    hook = SteeringHook(
        model, harm_vectors, reasoning_vectors, harm_alpha, reasoning_alpha
    )
    predictions = []
    for _, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        prompt = build_inference_prompt(row, tokenizer)  # implement
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
        ).to(DEVICE)
        with hook, torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        label = parse_output(generated)  # implement
        predictions.append(label)
    return predictions
