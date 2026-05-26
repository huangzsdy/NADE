"""
YOLO Validation Script
Runs YOLOv8n validation on VOC dataset with CPU optimization.
Outputs predictions in COCO JSON format.
"""

from ultralytics import YOLO
import os


def run_yolo_validation():
    """Run YOLO validation on VOC dataset."""
    
    # Try yolo11n.pt first, fall back to yolov8n.pt
    try:
        model = YOLO('yolo11n.pt')
        model_name = 'yolo11n'
    except Exception:
        model = YOLO('yolov8n.pt')
        model_name = 'yolov8n'
    
    # Run validation on VOC dataset with CPU settings
    # device='cpu', imgsz=416, batch=1, save_json=True
    results = model.val(
        data='voc',
        device='cpu',
        imgsz=416,
        batch=1,
        save_json=True,
        verbose=True
    )
    
    print(f"{model_name.capitalize()} CPU Validation Successful")
    return results


if __name__ == "__main__":
    run_yolo_validation()