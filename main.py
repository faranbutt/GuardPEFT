import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="GuardPEFT CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    prep_parser = subparsers.add_parser("prep", help="Run data preparation scripts")
    prep_parser.add_argument("--dataset", required=True, choices=["aegis", "biasmd", "ethics", "gretel", "pku", "toxigen", "merge"], help="Dataset to prepare")


    train_parser = subparsers.add_parser("train", help="Run training scripts")
    train_parser.add_argument("--method", required=True, choices=["dora", "qlora", "dora_steering", "qlora_steering"], help="Training method")


    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation scripts")
    eval_parser.add_argument("--target", required=True, choices=["ethics", "pku", "dora_ethics", "dora_pku", "test_set"], help="Evaluation target")

    args = parser.parse_args()

    if args.command == "prep":
        if args.dataset == "merge":
            script = "src/data/merge_and_split.py"
        else:
            script = f"src/data/prepare_{args.dataset}.py"
        subprocess.run([sys.executable, script], check=True)
        
    elif args.command == "train":
        script_map = {
            "dora": "src/training/dora_finetune.py",
            "qlora": "src/training/qlora_finetune.py",
            "dora_steering": "src/training/dora_adaptive_steering.py",
            "qlora_steering": "src/training/qlora_adaptive_steering.py",
        }
        subprocess.run([sys.executable, script_map[args.method]], check=True)
        
    elif args.command == "evaluate":
        script_map = {
            "ethics": "src/evaluation/evaluate_adapters+baselines_ethics.py",
            "pku": "src/evaluation/evaluate_adapters+baselines_pku.py",
            "dora_ethics": "src/evaluation/evaluate_dora_adapters_without_steering_ethics.py",
            "dora_pku": "src/evaluation/evaluate_dora_adapters_without_steering_on_pku.py",
            "test_set": "src/evaluation/evaluate_test_set.py",
        }
        subprocess.run([sys.executable, script_map[args.target]], check=True)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
