"""
AP vs IoU Sensitivity Analysis
Analyzes how Average Precision varies across different IoU thresholds.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
        
    Returns:
        IoU value.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap_at_iou(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_threshold: float,
    num_classes: int = 80
) -> float:
    """Compute AP at a specific IoU threshold.
    
    Efficient implementation avoiding redundant computations.
    
    Args:
        predictions: List of prediction dicts with keys: image_id, category_id, bbox, score
        ground_truth: List of GT dicts with keys: image_id, category_id, bbox
        iou_threshold: IoU threshold for detection matching
        num_classes: Number of object classes
        
    Returns:
        Average Precision value.
    """
    # Group by image and class for faster matching
    gt_by_image_class: Dict[Tuple[int, int], List[Dict]] = {}
    for gt in ground_truth:
        key = (gt['image_id'], gt['category_id'])
        if key not in gt_by_image_class:
            gt_by_image_class[key] = []
        gt_by_image_class[key].append({
            'bbox': gt['bbox'],
            'matched': False
        })
    
    # Sort predictions by score (descending)
    sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Track TP/FP for each class
    tp_by_class: Dict[int, List[bool]] = {i: [] for i in range(num_classes)}
    fp_by_class: Dict[int, List[bool]] = {i: [] for i in range(num_classes)}
    
    # Match predictions
    for pred in sorted_preds:
        img_id = pred['image_id']
        class_id = pred['category_id']
        pred_bbox = pred['bbox']
        
        key = (img_id, class_id)
        if key not in gt_by_image_class:
            # No GT for this class in this image - false positive
            fp_by_class[class_id].append(True)
            continue
        
        gts = gt_by_image_class[key]
        
        # Find best matching GT
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if gt['matched']:
                continue
            iou = box_iou(pred_bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        
        if best_iou >= iou_threshold and best_idx >= 0:
            gts[best_idx]['matched'] = True
            tp_by_class[class_id].append(True)
            fp_by_class[class_id].append(False)
        else:
            tp_by_class[class_id].append(False)
            fp_by_class[class_id].append(True)
    
    # Calculate AP for each class using 11-point interpolation
    ap_sum = 0.0
    num_classes_with_gt = 0
    
    for class_id in range(num_classes):
        tps = tp_by_class[class_id]
        fps = fp_by_class[class_id]
        
        if not tps:
            continue
        
        num_classes_with_gt += 1
        
        # Cumulative sums
        tps_cumsum = np.cumsum(tps)
        fps_cumsum = np.cumsum(fps)
        
        # Recall and precision
        recalls = tps_cumsum / len(tps)
        precisions = tps_cumsum / (tps_cumsum + fps_cumsum)
        
        # 11-point interpolation
        ap = 0.0
        for recall_level in np.arange(0, 1.1, 0.1):
            prec_at_recall = precisions[recalls >= recall_level]
            ap += prec_at_recall.max() if len(prec_at_recall) > 0 else 0
        
        ap /= 11
        ap_sum += ap
    
    return ap_sum / num_classes_with_gt if num_classes_with_gt > 0 else 0.0


def load_coco_annotations(json_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load COCO format annotations.
    
    Args:
        json_path: Path to COCO JSON file.
        
    Returns:
        Tuple of (predictions, ground_truth) lists.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    predictions = []
    ground_truth = []
    
    # Handle both annotation and prediction formats
    if 'annotations' in data:
        for ann in data['annotations']:
            ground_truth.append({
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox']  # [x, y, w, h] format
            })
    
    if 'predictions' in data:
        for pred in data['predictions']:
            predictions.append({
                'image_id': pred['image_id'],
                'category_id': pred['category_id'],
                'bbox': pred['bbox'],
                'score': pred.get('score', 1.0)
            })
    
    # Also check for results (common in detection outputs)
    if not predictions and 'results' in data:
        for res in data['results']:
            predictions.append({
                'image_id': res['image_id'],
                'category_id': res['category_id'],
                'bbox': res['bbox'],
                'score': res.get('score', 1.0)
            })
    
    return predictions, ground_truth


def compute_ap_curve(
    predictions: List[Dict],
    ground_truth: List[Dict],
    iou_thresholds: List[float],
    num_classes: int = 80
) -> Dict[float, float]:
    """Compute AP values across multiple IoU thresholds.
    
    Args:
        predictions: Predictions list
        ground_truth: Ground truth list
        iou_thresholds: List of IoU thresholds
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping IoU threshold to AP value.
    """
    ap_values = {}
    
    for iou in iou_thresholds:
        ap = compute_ap_at_iou(predictions, ground_truth, iou, num_classes)
        ap_values[iou] = ap
    
    return ap_values


def plot_ap_iou_sensitivity(
    data_dict: Dict[str, Dict[float, float]],
    output_path: str = 'figs/ap_iou_sensitivity.png',
    dpi: int = 300
) -> None:
    """Plot AP vs IoU threshold sensitivity curve.
    
    Args:
        data_dict: Dictionary mapping label to {iou: ap} values
        output_path: Path to save figure
        dpi: DPI for figure resolution
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    
    # Color palette
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    markers = ['o', 's', '^', 'D']
    
    for idx, (label, ap_values) in enumerate(data_dict.items()):
        ious = sorted(ap_values.keys())
        aps = [ap_values[iou] for iou in ious]
        
        ax.plot(
            ious, aps,
            label=label,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            markersize=8,
            linewidth=2,
            markeredgecolor='black',
            markeredgewidth=1
        )
    
    ax.set_xlabel('IoU Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Precision (AP)', fontsize=14, fontweight='bold')
    ax.set_title('AP vs IoU Threshold Sensitivity', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xticks(sorted(list(data_dict.values())[0].keys()))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze AP vs IoU threshold sensitivity'
    )
    parser.add_argument(
        '--gt_json',
        type=str,
        required=True,
        help='Path to ground truth COCO JSON'
    )
    parser.add_argument(
        '--preds_json',
        type=str,
        nargs='+',
        required=True,
        help='Paths to prediction JSONs (supports multiple noise levels)'
    )
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=None,
        help='Labels for each prediction JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='figs/ap_iou_sensitivity.png',
        help='Output path for figure'
    )
    parser.add_argument(
        '--iou_range',
        type=float,
        nargs='+',
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.75],
        help='IoU thresholds to evaluate'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Figure DPI'
    )
    
    args = parser.parse_args()
    
    # Default labels
    if args.labels is None:
        args.labels = [f'Noise {i}' for i in range(len(args.preds_json))]
    
    # Ensure same length
    if len(args.preds_json) != len(args.labels):
        args.labels = args.labels[:len(args.preds_json)] if args.labels else [
            f'Noise {i}' for i in range(len(args.preds_json))
        ]
    
    # Load ground truth once
    print(f"Loading ground truth from: {args.gt_json}")
    _, gt = load_coco_annotations(args.gt_json)
    print(f"Loaded {len(gt)} ground truth annotations")
    
    # Compute AP curves for each prediction file
    data_dict = {}
    
    for pred_json, label in zip(args.preds_json, args.labels):
        print(f"Processing: {label}")
        preds, _ = load_coco_annotations(pred_json)
        print(f"Loaded {len(preds)} predictions")
        
        # Compute AP at each IoU
        ap_values = compute_ap_curve(preds, gt, args.iou_range)
        data_dict[label] = ap_values
        
        # Print results
        print(f"\n{label}:")
        for iou in sorted(ap_values.keys()):
            print(f"  IoU@{iou:.2f}: AP={ap_values[iou]:.4f}")
    
    # Plot
    plot_ap_iou_sensitivity(data_dict, args.output, args.dpi)


if __name__ == "__main__":
    main()