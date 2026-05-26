"""
Synthetic Noise Generator for YOLO Format Labels
Adds various types of noise to YOLO bounding box annotations.
"""

import argparse
import os
import random
import logging
import numpy as np
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_yolo_label(line):
    """Parse a single YOLO label line.
    
    Args:
        line: String in format "class_id x_center y_center width height"
        
    Returns:
        Tuple of (class_id, xc, yc, w, h) as floats
    """
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label format: {line}")
    class_id = int(parts[0])
    xc = float(parts[1])
    yc = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])
    return (class_id, xc, yc, w, h)


def format_yolo_label(class_id, xc, yc, w, h):
    """Format a YOLO label tuple back to string.
    
    Args:
        class_id: Integer class ID
        xc, yc, w, h: Float coordinates (normalized to [0, 1])
        
    Returns:
        Formatted string "class_id xc yc w h"
    """
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def apply_jitter(box, prob, img_width=1.0, img_height=1.0):
    """Apply jitter noise to bounding box coordinates.
    
    Args:
        box: Tuple of (class_id, xc, yc, w, h)
        prob: Noise probability/scale factor
        img_width: Image width for normalization
        img_height: Image height for normalization
        
    Returns:
        Tuple of (class_id, new_xc, new_yc, w, h) with jitter applied
    """
    class_id, xc, yc, w, h = box
    
    # Standard deviation proportional to prob and box dimensions
    std_x = prob * w
    std_y = prob * h
    
    # Add Gaussian noise
    new_xc = xc + np.random.normal(0, std_x)
    new_yc = yc + np.random.normal(0, std_y)
    
    # Clip to [0, 1]
    new_xc = np.clip(new_xc, 0, 1)
    new_yc = np.clip(new_yc, 0, 1)
    
    return (class_id, new_xc, new_yc, w, h)


def apply_missing(boxes, prob):
    """Apply missing noise by randomly removing bounding boxes.
    
    Args:
        boxes: List of (class_id, xc, yc, w, h) tuples
        prob: Probability of removing each box
        
    Returns:
        List of boxes with some removed
    """
    kept_boxes = []
    for box in boxes:
        if random.random() > prob:
            kept_boxes.append(box)
    return kept_boxes


def apply_flip(boxes, prob, valid_class_ids):
    """Apply flip noise by randomly changing class IDs.
    
    Args:
        boxes: List of (class_id, xc, yc, w, h) tuples
        prob: Probability of flipping each box's class
        valid_class_ids: List of valid class IDs in the dataset
        
    Returns:
        List of boxes with some flipped
    """
    if len(valid_class_ids) <= 1:
        # Cannot flip if only one class
        return boxes
    
    flipped_boxes = []
    for box in boxes:
        class_id, xc, yc, w, h = box
        if random.random() < prob:
            # Change to a different class ID
            other_classes = [c for c in valid_class_ids if c != class_id]
            new_class_id = random.choice(other_classes)
            flipped_boxes.append((new_class_id, xc, yc, w, h))
        else:
            flipped_boxes.append(box)
    return flipped_boxes


def apply_ghost(boxes, prob, valid_class_ids):
    """Apply ghost noise by adding fake bounding boxes.
    
    Args:
        boxes: List of existing (class_id, xc, yc, w, h) tuples
        prob: Probability of adding ghost box per original box
        valid_class_ids: List of valid class IDs in the dataset
        
    Returns:
        List of boxes with ghost boxes added
    """
    new_boxes = list(boxes)
    num_original = len(boxes)
    
    for _ in range(num_original):
        if random.random() < prob:
            # Generate random bounding box
            xc = random.uniform(0.1, 0.9)
            yc = random.uniform(0.1, 0.9)
            w = random.uniform(0.05, 0.3)
            h = random.uniform(0.05, 0.3)
            class_id = random.choice(valid_class_ids)
            new_boxes.append((class_id, xc, yc, w, h))
    
    return new_boxes


def scan_valid_classes(input_dir):
    """Scan all label files to find valid class IDs.
    
    Args:
        input_dir: Path to directory containing YOLO labels
        
    Returns:
        Sorted list of unique class IDs found
    """
    class_ids = set()
    label_files = list(Path(input_dir).glob("*.txt"))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
                    except (ValueError, IndexError):
                        continue
    
    return sorted(class_ids)


def process_labels(input_dir, output_dir, noise_type, prob):
    """Process all YOLO label files and add noise.
    
    Args:
        input_dir: Directory containing original YOLO labels
        output_dir: Directory to save noisy labels
        noise_type: Type of noise to apply
        prob: Probability of noise
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    label_files = list(input_path.glob("*.txt"))
    
    if len(label_files) == 0:
        logger.warning(f"No label files found in {input_dir}")
        return
    
    # Scan valid class IDs for flip/ghost operations
    valid_class_ids = scan_valid_classes(input_dir)
    logger.info(f"Found valid class IDs: {valid_class_ids}")
    
    # Track statistics
    total_modified = 0
    total_deleted = 0
    total_added = 0
    
    # Process each file
    for label_file in label_files:
        # Read original boxes
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        boxes = [parse_yolo_label(line) for line in lines]
        
        original_count = len(boxes)
        
        # Apply noise based on type
        if noise_type == 'jitter':
            new_boxes = []
            for box in boxes:
                new_box = apply_jitter(box, prob)
                if new_box != box:
                    total_modified += 1
                new_boxes.append(new_box)
            boxes = new_boxes
            
        elif noise_type == 'missing':
            original_count = len(boxes)
            boxes = apply_missing(boxes, prob)
            total_deleted += original_count - len(boxes)
            
        elif noise_type == 'flip':
            new_boxes = apply_flip(boxes, prob, valid_class_ids)
            total_modified += sum(1 for old, new in zip(boxes, new_boxes) if old[0] != new[0])
            boxes = new_boxes
            
        elif noise_type == 'ghost':
            original_count = len(boxes)
            boxes = apply_ghost(boxes, prob, valid_class_ids)
            total_added += len(boxes) - original_count
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Write noisy labels
        output_file = output_path / label_file.name
        with open(output_file, 'w') as f:
            for box in boxes:
                f.write(format_yolo_label(*box) + '\n')
    
    # Log statistics
    logger.info(f"Processed {len(label_files)} files")
    if noise_type == 'jitter':
        logger.info(f"Modified {total_modified} boxes")
    elif noise_type == 'missing':
        logger.info(f"Deleted {total_deleted} boxes")
    elif noise_type == 'flip':
        logger.info(f"Flipped {total_modified} class IDs")
    elif noise_type == 'ghost':
        logger.info(f"Added {total_added} ghost boxes")


def main():
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic noise for YOLO format labels'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to original YOLO labels'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to save noisy labels'
    )
    parser.add_argument(
        '--noise_type',
        type=str,
        required=True,
        choices=['jitter', 'missing', 'flip', 'ghost'],
        help='Type of noise to apply'
    )
    parser.add_argument(
        '--prob',
        type=float,
        required=True,
        help='Probability of noise (0 to 1)'
    )
    
    args = parser.parse_args()
    
    # Validate prob
    if not 0 <= args.prob <= 1:
        raise ValueError("--prob must be between 0 and 1")
    
    logger.info(f"Starting noise generation: {args.noise_type}")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Probability: {args.prob}")
    
    process_labels(
        args.input_dir,
        args.output_dir,
        args.noise_type,
        args.prob
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # No arguments - run tests using pytest-style assertions
        import sys
        np.random.seed(42)
        random.seed(42)
        
        # Test 1: JITTER
        box = (0, 0.5, 0.5, 0.2, 0.2)
        new_box = apply_jitter(box, prob=0.1)
        assert new_box[0] == 0  # Class unchanged
        assert 0 <= new_box[1] <= 1  # xc clipped
        assert 0 <= new_box[2] <= 1  # yc clipped
        print("[PASS] test_jitter")
        
        # Test 2: MISSING
        boxes = [(0, 0.5, 0.5, 0.2, 0.2)] * 10
        new_boxes = apply_missing(boxes, prob=1.0)
        assert len(new_boxes) == 0
        new_boxes = apply_missing(boxes, prob=0.0)
        assert len(new_boxes) == 10
        print("[PASS] test_missing")
        
        # Test 3: FLIP  
        valid_class_ids = [0, 1, 2]
        boxes = [(0, 0.5, 0.5, 0.2, 0.2)] * 5
        new_boxes = apply_flip(boxes, prob=1.0, valid_class_ids=valid_class_ids)
        for nb in new_boxes:
            assert nb[0] != 0  # Flipped to non-zero
        new_boxes = apply_flip(boxes, prob=0.0, valid_class_ids=valid_class_ids)
        for nb in new_boxes:
            assert nb[0] == 0  # Not flipped
        print("[PASS] test_flip")
        
        # Test 4: GHOST
        boxes = [(0, 0.5, 0.5, 0.2, 0.2)]
        new_boxes = apply_ghost(boxes, prob=0.0, valid_class_ids=valid_class_ids)
        assert len(new_boxes) == 1
        new_boxes = apply_ghost(boxes, prob=1.0, valid_class_ids=valid_class_ids)
        assert len(new_boxes) == 2
        for nb in new_boxes:
            assert nb[0] in valid_class_ids
        print("[PASS] test_ghost")
        
        # Test 5: PARSE/FORMAT
        line = "0 0.500000 0.500000 0.200000 0.200000"
        box = parse_yolo_label(line)
        assert box[0] == 0 and box[1] == 0.5
        formatted = format_yolo_label(*box)
        assert formatted == line
        print("[PASS] test_parse_format")
        
        # Test 6: GHOST COORDINATES RANGE
        random.seed(123)
        boxes = [(0, 0.5, 0.5, 0.2, 0.2)]
        for _ in range(100):
            new_boxes = apply_ghost(boxes, prob=1.0, valid_class_ids=valid_class_ids)
            for nb in new_boxes:
                assert 0 <= nb[1] <= 1 and 0 <= nb[2] <= 1
                assert nb[3] > 0 and nb[4] > 0
        print("[PASS] test_ghost_coordinates_range")
        
        print("\n*** ALL TESTS PASSED ***")
    else:
        main()


# ============================================================================
# Unit Tests
# ============================================================================

def test_jitter():
    """Test jitter noise function."""
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Test basic jitter
    box = (0, 0.5, 0.5, 0.2, 0.2)
    new_box = apply_jitter(box, prob=0.1)
    
    # Class ID should remain unchanged
    assert new_box[0] == 0
    
    # Coordinates should be clipped to [0, 1]
    assert 0 <= new_box[1] <= 1  # xc
    assert 0 <= new_box[2] <= 1  # yc
    assert 0 <= new_box[3] <= 1  # w
    assert 0 <= new_box[4] <= 1  # h
    
    print("test_jitter: PASSED")


def test_missing():
    """Test missing noise function."""
    random.seed(42)
    
    boxes = [(0, 0.5, 0.5, 0.2, 0.2)] * 10
    
    # With prob=1.0, all should be removed
    new_boxes = apply_missing(boxes, prob=1.0)
    assert len(new_boxes) == 0
    
    # With prob=0.0, none should be removed
    new_boxes = apply_missing(boxes, prob=0.0)
    assert len(new_boxes) == 10
    
    print("test_missing: PASSED")


def test_flip():
    """Test flip noise function."""
    random.seed(42)
    
    valid_class_ids = [0, 1, 2]
    boxes = [(0, 0.5, 0.5, 0.2, 0.2)] * 5
    
    # With prob=1.0, all should flip
    new_boxes = apply_flip(boxes, prob=1.0, valid_class_ids=valid_class_ids)
    
    # Check all classes changed from 0 to something else
    for box in new_boxes:
        assert box[0] != 0  # Should not be 0 anymore
    
    # With prob=0.0, none should flip
    new_boxes = apply_flip(boxes, prob=0.0, valid_class_ids=valid_class_ids)
    for box in new_boxes:
        assert box[0] == 0
    
    print("test_flip: PASSED")


def test_ghost():
    """Test ghost noise function."""
    random.seed(42)
    
    valid_class_ids = [0, 1, 2]
    boxes = [(0, 0.5, 0.5, 0.2, 0.2)]
    
    # With prob=0.0, should add 0 ghost boxes
    new_boxes = apply_ghost(boxes, prob=0.0, valid_class_ids=valid_class_ids)
    assert len(new_boxes) == 1
    
    # With prob=1.0, should add 1 ghost box
    new_boxes = apply_ghost(boxes, prob=1.0, valid_class_ids=valid_class_ids)
    assert len(new_boxes) == 2
    
    # Ghost boxes should have valid class IDs
    for box in new_boxes:
        assert box[0] in valid_class_ids
        # Should be in valid range
        assert 0 <= box[1] <= 1  # xc
        assert 0 <= box[2] <= 1  # yc
        assert 0 < box[3] <= 1   # w
        assert 0 < box[4] <= 1  # h
    
    print("test_ghost: PASSED")


def test_parse_format():
    """Test YOLO label parsing and formatting."""
    line = "0 0.500000 0.500000 0.200000 0.200000"
    box = parse_yolo_label(line)
    
    assert box[0] == 0  # class_id
    assert box[1] == 0.5  # xc
    assert box[2] == 0.5  # yc
    assert box[3] == 0.2  # w
    assert box[4] == 0.2  # h
    
    # Round-trip should produce same values
    formatted = format_yolo_label(*box)
    assert formatted == line
    
    print("test_parse_format: PASSED")


def test_ghost_coordinates_range():
    """Test that ghost box coordinates are in valid range."""
    random.seed(123)
    
    valid_class_ids = [0, 1, 2]
    boxes = [(0, 0.5, 0.5, 0.2, 0.2)]
    
    # Add many ghost boxes
    for _ in range(100):
        new_boxes = apply_ghost(boxes, prob=1.0, valid_class_ids=valid_class_ids)
    
    # All boxes should have valid coordinates
    for box in new_boxes:
        _, xc, yc, w, h = box
        # Must be positive
        assert w > 0 and h > 0
        # Must be in [0, 1]
        assert 0 <= xc <= 1
        assert 0 <= yc <= 1
        assert 0 <= w <= 1
        assert 0 <= h <= 1
    
    print("test_ghost_coordinates_range: PASSED")


if __name__ == "__main__":
    # Run all tests
    test_jitter()
    test_missing()
    test_flip()
    test_ghost()
    test_parse_format()
    test_ghost_coordinates_range()
    
    print("\n=== ALL TESTS PASSED ===")