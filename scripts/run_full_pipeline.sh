#!/bin/bash

# 1. Prepare combined dataset
python src/data/prepare_combined.py

# 2. Run baselines on combined test & out-of-domain
python src/evaluation/baseline.py

# 3. Fine-tune LoRA adapters
python src/models/finetune.py

# 4. Evaluate adapters
python src/evaluation/eval_adapter.py

# 5. Adaptive steering (build vectors, evaluate)
python src/models/steering.py
python src/evaluation/eval_steering.py
