# NADE - Noise-Aware Detection Framework

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Object%20Detection-Noise%20Aware-orange.svg" alt="Project Tag">
</p>

## 📖 Introduction

NADE (Noise-Aware Detection) is an experimental framework for researching and evaluating object detection models under noisy annotation conditions. This project provides a complete pipeline supporting:

- 🏷️ **Noisy Label Generation**: Multiple noise types (Jitter, Missing, Flip, Ghost)
- 🔍 **Object Detection Inference**: YOLO model integration
- 📊 **Comprehensive Evaluation**: TIDE and AP-IoU curve analysis
- 📈 **Result Aggregation**: Automatic result summarization and visualization

## 🗂️ Project Structure

```
NADE/
├── main.py                    # BeyondMap model training entry
├── run_experiment.py          # Master experiment orchestration
├── configs/
│   └── config.yaml            # Configuration file
├── data/                      # Data directory
│   ├── datasets.py            # Dataset loaders
│   ├── voc/                   # VOC dataset
│   ├── coco_subset/           # COCO subset
│   └── noise_gt/              # Generated noisy labels
├── src/
│   ├── data/
│   │   └── make_noise.py      # Noise generation module
│   ├── models/
│   │   └── run_yolo_val.py   # YOLO inference module
│   └── eval/
│       ├── ap_iou_curve.py   # AP-IoU evaluation
│       └── run_tide.py         # TIDE evaluation
├── models/
│   └── beyondmap_model.py    # BeyondMap model implementation
├── trainers/
│   └── trainer.py            # Trainer
└── figs/                     # Output figures
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- PyTorch ≥ 1.9
- NumPy, Pandas, Matplotlib

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Complete Experiment

```bash
# Run complete noise-aware detection experiment
python run_experiment.py

# Optional flags
python run_experiment.py --skip_noise        # Skip noise generation
python run_experiment.py --skip_inference  # Skip inference
python run_experiment.py --skip_eval       # Skip evaluation
python run_experiment.py --clean          # Clean and rerun
```

### Run BeyondMap Model Training

```bash
python main.py
```

## 🎯 Supported Noise Types

| Noise Type | Description | Parameters |
|-----------|-------------|------------|
| **Jitter** | Add Gaussian noise to bounding box coordinates | `--noise_type jitter --prob 0.1` |
| **Missing** | Randomly remove annotation boxes | `--noise_type missing --prob 0.1` |
| **Flip** | Randomly flip class labels | `--noise_type flip --prob 0.1` |
| **Ghost** | Add spurious bounding boxes | `--noise_type ghost --prob 0.1` |

### Generate Noisy Labels Example

```bash
# Generate Jitter noise (levels 0.1, 0.2, 0.3)
python -m src.data.make_noise \
    --input_dir data/voc/annotations.json \
    --output_dir data/noise_gt/jitter_0.1 \
    --noise_type jitter \
    --prob 0.1
```

## 📊 Evaluation Metrics

- **TIDE** (Tilted Detection Error Analysis)
- **AP@IoU** (Average Precision at various IoU thresholds)
- **Precision-Recall Curve**

## ⚙️ Configuration

Configuration file located at [`configs/config.yaml`](configs/config.yaml):

```yaml
# Hardware Settings
device: cpu
batch_size: 1
num_workers: 0

# Training Settings
learning_rate: 0.001
num_epochs: 10

# Model Settings
use_amp: false
low_memory_mode: true
```

## 📝 Module Documentation

### Noise Generation (`src/data/make_noise.py`)

Four noise types supported:
- `apply_jitter()`: Add coordinate jitter noise
- `apply_missing()`: Remove missing annotations
- `apply_flip()`: Flip class labels
- `apply_ghost()`: Add ghost boxes

### Evaluation Modules (`src/eval/`)

- `ap_iou_curve.py`: Compute AP across different IoU thresholds
- `run_tide.py`: Run TIDE analysis

### Experiment Orchestration (`run_experiment.py`)

Master script executing in order:
1. Generate noisy labels
2. Run YOLO inference
3. Execute evaluation
4. Aggregate results

## 🤝 Contributing

Pull Requests and Issues are welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

This project references:
- [YOLO](https://github.com/ultralytics/yolov5) - Object detection model
- [TIDE](https://github.com/dblaze/tide) - Detection error analysis tool