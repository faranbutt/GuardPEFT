import json
from pathlib import Path

from transformers import TrainerCallback


class FlushMetricsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self._total_steps = state.max_steps
        print(f"🚀 Training started | Total steps: {self._total_steps}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        total = self._total_steps
        pct = (step / total * 100) if total > 0 else 0
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        parts = [f"[{bar}] {step:>4}/{total} {pct:5.1f}%"]
        if logs.get("loss"):
            parts.append(f"train_loss={float(logs['loss']):.4f}")
        if logs.get("eval_loss"):
            parts.append(f"eval_loss={float(logs['eval_loss']):.4f}")
        if logs.get("learning_rate"):
            parts.append(f"lr={float(logs['learning_rate']):.2e}")
        print(" | ".join(parts), flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        print(
            f"── Epoch {int(state.epoch)}/{int(args.num_train_epochs)} complete ──",
            flush=True,
        )

    def on_train_end(self, args, state, control, **kwargs):
        print(f"✅ Done | Best eval_loss: {state.best_metric}", flush=True)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
