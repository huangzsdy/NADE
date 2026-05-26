"""
TIDE Evaluation Script
Interfaces with tidecv library for object detection error analysis.
Generates publication-quality visualizations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_coco_json(file_path: str) -> dict:
    """Load COCO-format JSON file.
    
    Args:
        file_path: Path to COCO JSON file.
        
    Returns:
        Dictionary containing COCO annotations.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def run_tide_evaluation(
    gt_json: str,
    pred_json: str,
    iou_threshold: float = 0.5
) -> 'TIDE':
    """Run TIDE evaluation on predictions.
    
    Args:
        gt_json: Path to ground truth COCO JSON.
        pred_json: Path to predictions COCO JSON.
        iou_threshold: IOU threshold for matching.
        
    Returns:
        TIDE object with results.
    """
    try:
        from tidecv import TIDE
        from tidecv.data import COCO
    except ImportError:
        print("Error: tidecv not installed. Install with: pip install tidecv")
        sys.exit(1)
    
    # Load data
    gt = COCO(gt_json)
    preds = COCO(pred_json)
    
    # Run TIDE evaluation
    tide = TIDE(iou_thresholds=[iou_threshold])
    tide.evaluate(gt, preds)
    
    return tide


def print_tide_summary(tide: 'TIDE') -> None:
    """Print TIDE summary table to console.
    
    Args:
        tide: TIDE object with evaluation results.
    """
    print("\n" + "=" * 60)
    print("TIDE Error Breakdown Summary")
    print("=" * 60)
    
    # Get main results
    main = tide.results['main']
    
    # Print error type table
    print(f"\n{'Error Type':<15} {'AP Loss':<12} {'% of Total':<12}")
    print("-" * 39)
    
    error_types = {
        'Cls': 'Classification',
        'Loc': 'Localization', 
        'Dupe': 'Duplicate',
        'Bkg': 'Background',
        'Miss': 'Missing'
    }
    
    total_ap_loss = main.apLoss
    for error_code, error_name in error_types.items():
        if error_code in main.error_breakdown:
            ap_loss = main.error_breakdown[error_code]
            pct = (ap_loss / total_ap_loss * 100) if total_ap_loss > 0 else 0
            print(f"{error_name:<15} {ap_loss:.4f}       {pct:>6.1f}%")
    
    print("-" * 39)
    print(f"{'Total AP Loss':<15} {main.apLoss:.4f}")
    print(f"{'True Positive':<15} {main.tp}")
    print(f"{'False Positive':<15} {main.fp}")
    print(f"{'False Negative':<15} {main.fn}")
    print("=" * 60 + "\n")


def generate_error_bar_chart(
    tide: 'TIDE',
    output_path: str = 'figs/tide_decomposition.png',
    dpi: int = 300
) -> None:
    """Generate publication-quality bar chart of AP loss by error type.
    
    Args:
        tide: TIDE object with evaluation results.
        output_path: Path to save figure.
        dpi: DPI for figure resolution.
    """
    # Get error breakdown
    main = tide.results['main']
    error_breakdown = main.error_breakdown
    
    # Define error types and colors
    error_types = ['Cls', 'Loc', 'Dupe', 'Bkg', 'Miss']
    error_names = ['Classification', 'Localization', 'Duplicate', 'Background', 'Missing']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    # Extract values
    values = [error_breakdown.get(et, 0) for et in error_types]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    
    # Create bar positions
    x_pos = np.arange(len(error_types))
    
    # Create bars
    bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Customize plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(error_names, fontsize=12, fontweight='bold')
    ax.set_ylabel('AP Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Error Type', fontsize=14, fontweight='bold')
    ax.set_title('TIDE Error Decomposition', fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )
    
    # Style improvements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to: {output_path}")


def compare_tide(
    gt_clean: str,
    gt_noisy: str,
    preds: str,
    output_path: str = 'figs/tide_comparison.png',
    dpi: int = 300
) -> Dict[str, Dict[str, float]]:
    """Compare TIDE error breakdown between clean and noisy ground truth.
    
    Args:
        gt_clean: Path to clean ground truth COCO JSON.
        gt_noisy: Path to noisy ground truth COCO JSON.
        preds: Path to predictions COCO JSON.
        output_path: Path to save comparison figure.
        dpi: DPI for figure resolution.
        
    Returns:
        Dictionary with 'clean' and 'noisy' results.
    """
    # Run evaluation on clean GT
    print("Evaluating on clean ground truth...")
    tide_clean = run_tide_evaluation(gt_clean, preds)
    clean_results = tide_clean.results['main']
    
    # Run evaluation on noisy GT
    print("Evaluating on noisy ground truth...")
    tide_noisy = run_tide_evaluation(gt_noisy, preds)
    noisy_results = tide_noisy.results['main']
    
    # Print comparison
    print("\n" + "=" * 60)
    print("Clean vs Noisy GT Comparison")
    print("=" * 60)
    
    error_types = ['Cls', 'Loc', 'Dupe', 'Bkg', 'Miss']
    error_names = ['Classification', 'Localization', 'Duplicate', 'Background', 'Missing']
    
    print(f"\n{'Error Type':<15} {'Clean':<12} {'Noisy':<12} {'Delta':<12}")
    print("-" * 51)
    
    comparison = {'clean': {}, 'noisy': {}, 'delta': {}}
    
    for et, en in zip(error_types, error_names):
        clean_val = clean_results.error_breakdown.get(et, 0)
        noisy_val = noisy_results.error_breakdown.get(et, 0)
        delta = noisy_val - clean_val
        
        comparison['clean'][et] = clean_val
        comparison['noisy'][et] = noisy_val
        comparison['delta'][et] = delta
        
        print(f"{en:<15} {clean_val:.4f}     {noisy_val:.4f}     {delta:+.4f}")
    
    print("-" * 51)
    print(f"{'Total AP Loss':<15} {clean_results.apLoss:.4f}     {noisy_results.apLoss:.4f}     {noisy_results.apLoss - clean_results.apLoss:+.4f}")
    print("=" * 60 + "\n")
    
    # Generate comparison figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    
    x = np.arange(len(error_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [comparison['clean'][et] for et in error_types], 
                  width, label='Clean GT', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, [comparison['noisy'][et] for et in error_types],
                  width, label='Noisy GT', color='#e74c3c', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(error_names, fontsize=12, fontweight='bold')
    ax.set_ylabel('AP Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Error Type', fontsize=14, fontweight='bold')
    ax.set_title('TIDE Error Comparison: Clean vs Noisy GT', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    
    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison figure to: {output_path}")
    
    return comparison


def main():
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description='Run TIDE evaluation on object detection predictions'
    )
    parser.add_argument(
        '--gt_json',
        type=str,
        required=True,
        help='Path to ground truth COCO JSON'
    )
    parser.add_argument(
        '--pred_json',
        type=str,
        required=True,
        help='Path to predictions COCO JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figs/tide_decomposition.png',
        help='Output path for visualization (default: figs/tide_decomposition.png)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output figure (default: 300)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IOU threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    print(f"Running TIDE evaluation...")
    print(f"Ground Truth: {args.gt_json}")
    print(f"Predictions: {args.pred_json}")
    print(f"IOU Threshold: {args.iou}")
    
    # Run evaluation
    tide = run_tide_evaluation(args.gt_json, args.pred_json, args.iou)
    
    # Print summary
    print_tide_summary(tide)
    
    # Generate visualization
    generate_error_bar_chart(tide, args.output, args.dpi)


if __name__ == "__main__":
    main()