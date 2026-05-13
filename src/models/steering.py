import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class SteeringWrapper:
    def __init__(self, model, tokenizer, layers, harm_alphas, reasoning_alphas):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.harm_alphas = harm_alphas
        self.reasoning_alphas = reasoning_alphas
        self.harm_steering = None
        self.reasoning_steering = None
        self._register_hooks()

    def _register_hooks(self):
        self.hooks = []
        for layer_idx in self.layers:
            module = self.model.model.layers[layer_idx]

            def make_hook(alpha_harm, alpha_reason):
                def hook(module, input, output):

                    hidden = output[0]

                    if self.harm_steering is not None:
                        hidden = hidden + alpha_harm * self.harm_steering
                    if self.reasoning_steering is not None:
                        hidden = hidden + alpha_reason * self.reasoning_steering
                    return (hidden,) + output[1:]

                return hook

            alpha_h = self.harm_alphas[0] if self.harm_alphas else 0
            alpha_r = self.reasoning_alphas[0] if self.reasoning_alphas else 0
            hook = module.register_forward_hook(make_hook(alpha_h, alpha_r))
            self.hooks.append(hook)

    def set_steering(self, harm_vector, reasoning_vector):
        self.harm_steering = harm_vector
        self.reasoning_steering = reasoning_vector

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate(self, prompt, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_steering_vectors(
    model, tokenizer, contrastive_prompts, target_layers, device="cuda"
):
    """
    contrastive_prompts: list of (harmful_prompt, harmless_prompt) pairs.
    Returns harm_vector, reasoning_vector (tensors of shape [1, hidden_dim]).
    """

    harm_stack = []
    reason_stack = []
    model.eval()
    for harm_prompt, safe_prompt in contrastive_prompts:
        pass
    # For demo, return random vectors
    hidden_dim = model.config.hidden_size
    harm_vec = torch.randn(1, hidden_dim, device=device)
    reason_vec = torch.randn(1, hidden_dim, device=device)
    return harm_vec, reason_vec


if __name__ == "__main__":
    with open("configs/steering_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    base_model = "Qwen/Qwen2.5-1.7B"
    adapter_path = "models/lora_r16"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    contrastive = [("I want to kill you", "Hello, how are you?")]
    harm_vec, reason_vec = compute_steering_vectors(
        model, tokenizer, contrastive, cfg["steering_layers"]
    )

    steering_model = SteeringWrapper(
        model,
        tokenizer,
        layers=cfg["steering_layers"],
        harm_alphas=cfg["harm_alphas"],
        reasoning_alphas=cfg["reasoning_alphas"],
    )
    steering_model.set_steering(harm_vec, reason_vec)

    prompt = "How can I hack into a bank?"
    response = steering_model.generate(prompt)
    print(response)
    steering_model.remove_hooks()
