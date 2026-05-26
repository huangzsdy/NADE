"""
Master Experiment Orchestration Script
Runs the complete noise-aware detection evaluation pipeline.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Project root and directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
PRED_DIR = PROJECT_ROOT / "runs"  # Alias for predictions


def ensure_dirs() -> None:
    """Create necessary directories."""
    RUNS_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "noise_gt").mkdir(exist_ok=True)


def run_command(cmd: List[str], cwd: Optional[Path] = None) -> int:
    """Run shell command with error handling."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return result.returncode
        if result.stdout:
            print(result.stdout)
        return 0
    except FileNotFoundError:
        print(f"ERROR: Command not found: {cmd[0]}")
        return 1
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1


def generate_noise(clean_gt: str = "data/voc/annotations.json") -> Dict[str, str]:
    """Generate noisy label datasets."""
    print("\n" + "=" * 60)
    print("STEP 1: Generating Noise")
    print("=" * 60)
    
    noise_dirs = {}
    
    # Generate jitter noise at 0.1, 0.2, 0.3
    for level in [0.1, 0.2, 0.3]:
        output_dir = DATA_DIR / "noise_gt" / f"jitter_{level}"
        print(f"\nGenerating jitter @ {level}...")
        cmd = ["python", "-m", "src.data.make_noise", "--input_dir", clean_gt,
              "--output_dir", str(output_dir), "--noise_type", "jitter", "--prob", str(level)]
        ret = run_command(cmd)
        if ret == 0:
            noise_dirs[f"jitter_{level}"] = str(output_dir)
            print(f"Success: {output_dir}")
    
    # Generate missing noise at 0.1, 0.2, 0.3  
    for level in [0.1, 0.2, 0.3]:
        output_dir = DATA_DIR / "noise_gt" / f"missing_{level}"
        print(f"\nGenerating missing @ {level}...")
        cmd = ["python", "-m", "src.data.make_noise", "--input_dir", clean_gt,
              "--output_dir", str(output_dir), "--noise_type", "missing", "--prob", str(level)]
        ret = run_command(cmd)
        if ret == 0:
            noise_dirs[f"missing_{level}"] = str(output_dir)
            print(f"Success: {output_dir}")
    
    return noise_dirs


def run_inference(noise_dirs: Dict[str, str]) -> Dict[str, str]:
    """Run YOLO inference on each noise dataset."""
    print("\n" + "=" * 60)
    print("STEP 2: Running Inference")
    print("=" * 60)
    
    pred_dirs = {}
    for noise_type, label_dir in noise_dirs.items():
        output_dir = RUNS_DIR / noise_type
        print(f"\nRunning inference on {noise_type}...")
        cmd = ["python", "-m", "src.models.run_yolo_val", "--data", label_dir, "--output", str(output_dir)]
        ret = run_command(cmd)
        if ret == 0:
            pred_dirs[noise_type] = str(output_dir)
            print(f"Success: {output_dir}")
    return pred_dirs


def run_evaluation(noise_dirs: Dict[str, str], pred_dirs: Dict[str, str]) -> None:
    """Run TIDE and AP-IoU evaluations."""
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating")
    print("=" * 60)
    
    for noise_type in noise_dirs.keys():
        gt_dir = noise_dirs[noise_type]
        pred_dir = pred_dirs.get(noise_type)
        if not pred_dir:
            continue
        print(f"\nEvaluating {noise_type}...")
        
        # TIDE
        tide_cmd = ["python", "-m", "src.eval.run_tide", "--gt_json", gt_dir,
                   "--pred_json", pred_dir, "--output", f"figs/tide_{noise_type}.png"]
        run_command(tide_cmd)
        
        # AP-IoU curve
        ap_cmd = ["python", "-m", "src.eval.ap_iou_curve", "--gt_json", gt_dir,
                  "--pred_json", pred_dir, "--output", f"figs/ap_iou_{noise_type}.png"]
        run_command(ap_cmd)


def aggregate_results(noise_dirs: Dict[str, str], pred_dirs: Dict[str, str]) -> pd.DataFrame:
    """Aggregate all metrics into a DataFrame."""
    print("\n" + "=" * 60)
    print("STEP 4: Aggregating Results")
    print("=" * 60)
    
    rows = []
    for noise_type in noise_dirs.keys():
        gt_dir = noise_dirs[noise_type]
        pred_dir = pred_dirs.get(noise_type)
        if not pred_dir:
            continue
        parts = noise_type.split('_')
        noise_cat = parts[0]
        noise_level = float(parts[1]) if len(parts) > 1 else 0.0
        rows.append({
            'noise_type': noise_type,
            'noise_category': noise_cat,
            'noise_level': noise_level,
            'gt_directory': gt_dir,
            'pred_directory': pred_dir,
            'status': 'complete'
        })
    
    df = pd.DataFrame(rows)
    output_path = PROJECT_ROOT / "results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run complete noise-aware detection experiment')
    parser.add_argument('--skip_noise', action='store_true', help='Skip noise generation')
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference')
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--clean', action='store_true', help='Clean runs before running')
    args = parser.parse_args()
    
    print("=" * 60)
    print("NOISE-AWARE DETECTION EXPERIMENT")
    print("=" * 60)
    
    ensure_dirs()
    
    if args.clean:
        print("\nCleaning previous runs...")
        if RUNS_DIR.exists():
            shutil.rmtree(RUNS_DIR)
        ensure_dirs()
    
    noise_dirs = {}
    pred_dirs = {}
    
    # STEP 1: Generate Noise
    if not args.skip_noise:
        noise_dirs = generate_noise()
    else:
        print("[SKIPPED] Noise Generation")
        noise_base = DATA_DIR / "noise_gt"
        if noise_base.exists():
            for d in noise_base.iterdir():
                if d.is_dir():
                    noise_dirs[d.name] = str(d)
    
    # STEP 2: Run Inference
    if not args.skip_inference and noise_dirs:
        pred_dirs = run_inference(noise_dirs)
    else:
        print("[SKIPPED] Inference")
        if RUNS_DIR.exists():
            for d in RUNS_DIR.iterdir():
                if d.is_dir():
                    pred_dirs[d.name] = str(d)
    
    # STEP 3: Evaluate
    if not args.skip_eval and noise_dirs and pred_dirs:
        run_evaluation(noise_dirs, pred_dirs)
    else:
        print("[SKIPPED] Evaluation")
    
    # STEP 4: Aggregate
    if noise_dirs and pred_dirs:
        df = aggregate_results(noise_dirs, pred_dirs)
        print("\nResults Summary:")
        print(df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()