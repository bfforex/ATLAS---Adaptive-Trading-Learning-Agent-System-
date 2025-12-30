import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import os

from atlas_v2_config import ATLASConfig  # Must customize if not matching prototype
from atlas_v2_data import prepare_dataloaders, ATLASDataLoader
from atlas_v2_training import train_model, ATLASLightningModule
from atlas_v2_evaluation import ComprehensiveEvaluator
from atlas_v2_memory import ConfigSnapshotManager
from atlas_v2_simulation import MarketDataProvider, LiveTradingSimulator

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Safe fallback

def get_git_commit_info():
    # Print Git commit info for full reproducibility, if in a repo.
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        dirty = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD']) != 0
        return commit + (" (uncommitted changes)" if dirty else "")
    except Exception:
        return "unknown"

def print_banner(run_name):
    print("=" * 70)
    print("ATLAS V2 Modular ML Trading System")
    print(f"Run: {run_name}")
    print(f"Git commit: {get_git_commit_info()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 70)

def friendly_error(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)

def parse_dynamic_overrides(override_list):
    # --set key1=val1 --set key2=val2 ... turns into {key1: val1,...}
    updates = {}
    for kv in override_list:
        if '=' not in kv:
            print(f"[WARN] Invalid override: {kv}")
            continue
        key, value = kv.split('=', 1)
        # Try to interpret as float/int/bool if possible; else str
        try:
            if value.lower() in ('true', 'false'):
                v = value.lower() == 'true'
            else:
                v = int(value)
        except Exception:
            try:
                v = float(value)
            except Exception:
                v = value
        updates[key] = v
    return updates

def load_config(config_path=None, overrides=None):
    # Load and apply in-order: config file, then --set key=val
    if config_path and Path(config_path).exists():
        config = ATLASConfig.load(config_path)
        print(f"[INFO] Loaded config: {config_path}")
    else:
        config = ATLASConfig()
        print("[WARN] No config file specified/found, using default config.")

    if overrides:
        for k, v in overrides.items():
            if v is not None and hasattr(config, k):
                print(f"[INFO] Override config: {k} = {v}")
                setattr(config, k, v)
    return config

def validate_config_and_data(config):
    # Minimal check for file/dataset availability
    missing = []
    for s in getattr(config, 'symbols', []):
        data_path = Path('data/raw') / f"{s}.csv"
        if not data_path.exists():
            print(f"[WARN] No CSV data for {s} at {data_path}, will use synthetic data.")
    if getattr(config, 'device', 'cpu') == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                friendly_error("Configured device=cuda but no GPU detected!")
        except ImportError:
            friendly_error("torch is required for CUDA support")
    # Add more as needed (feature_size, batch_size, limits...)

def ensure_run_dir(name):
    name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
    d = Path("runs") / name
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_full_config(run_dir, config):
    config_path = Path(run_dir) / "active_config.json"
    as_dict = config.asdict() if hasattr(config, "asdict") else dict(config.__dict__)
    with open(config_path, "w") as f:
        json.dump(as_dict, f, indent=2)
    print(f"[INFO] Config snapshot written to {config_path}")

def main():
    parser = argparse.ArgumentParser(description="ATLAS V2 Modular Trading System (Best UX Features)")
    parser.add_argument("command", choices=["train", "simulate", "evaluate"], help="Run mode")
    parser.add_argument("--name", type=str, help="Run name/label for outputs")
    parser.add_argument("--config", type=str, help="Config JSON to load")
    parser.add_argument("--set", nargs="*", default=[], help="Override config values, e.g. --set learning_rate=0.0002 batch_size=256")
    parser.add_argument("--steps", type=int, help="Simulation steps (simulate)")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path (simulate/evaluate)")
    parser.add_argument("--output", type=str, help="Custom output directory (overrides --name)")
    parser.add_argument("--epochs", type=int, help="Epoch override (train)")
    parser.add_argument("--device", type=str, help="device override (train/sim/eval)")

    args = parser.parse_args()
    run_name = args.name or (args.output if args.output else None)
    run_dir = ensure_run_dir(run_name)

    print_banner(run_name or str(run_dir.name))

    # Step 1: Load config, apply CLI --set overrides
    cli_overrides = parse_dynamic_overrides(args.set)
    config = load_config(args.config, cli_overrides)
    if args.epochs: setattr(config, "num_epochs", args.epochs)
    if args.device: setattr(config, "device", args.device)
    save_full_config(run_dir, config)

    # Step 2: Preflight validation and warnings
    validate_config_and_data(config)

    if args.command == "train":
        print("[INFO] Preparing data ...")
        train_loader, val_loader, test_loader, _ = prepare_dataloaders(config)
        print("[INFO] Starting training ...")
        try:
            from tqdm import tqdm
            model, trainer = train_model(config, train_loader, val_loader, test_loader)
        except Exception as e:
            friendly_error(f"Training failed: {e}")
        print("[SUCCESS] Training complete.")
        # Print out checkpoint dir and best val metric
        print(f"[INFO] Checkpoints saved in: {config.memory_dir}/checkpoints/")
        print(f"[INFO] See run artifacts at {run_dir}")

    elif args.command == "evaluate":
        if not args.checkpoint:
            friendly_error("You must specify a --checkpoint for evaluation!")
        # Evaluation/prediction
        _, _, test_loader, _ = prepare_dataloaders(config)
        print(f"[INFO] Loading model from checkpoint: {args.checkpoint}")
        model = ATLASLightningModule.load_from_checkpoint(args.checkpoint, config=config)
        evaluator = ComprehensiveEvaluator(config)
        print("[INFO] Evaluating model ...")
        results = evaluator.evaluate_model(model, test_loader, device=config.device)
        report_path = run_dir / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write(evaluator.generate_report(results))
        print(f"[SUCCESS] Evaluation complete. Report written to: {report_path}")
        print("[INFO] Next step: Visualize results, compare with other runs, etc.")

    elif args.command == "simulate":
        if not args.checkpoint:
            friendly_error("You must specify a --checkpoint for simulation!")
        assets_data, _ = ATLASDataLoader.load_multi_asset(config.symbols, config.feature_size)
        provider = MarketDataProvider(config.symbols, assets_data)
        model = ATLASLightningModule.load_from_checkpoint(args.checkpoint, config=config)
        simulator = LiveTradingSimulator(model, config, market_data_provider=provider, results_dir=run_dir)
        steps = args.steps if args.steps else len(provider)
        print(f"[INFO] Running simulation for {steps} steps ...")
        simulator.run(n_steps=steps)
        print("[SUCCESS] Simulation complete.")
        print(f"[INFO] Results/logs at {simulator.results_dir}")
    else:
        friendly_error("Unknown command. Use --help for options.")

    # Print run summary and pointing to outputs  
    print("\n=== RUN COMPLETE ===")
    print(f"Config snapshot: {run_dir / 'active_config.json'}")
    print(f"Outputs/logs: {run_dir}")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        sys.exit(1)
