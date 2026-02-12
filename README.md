# Robustness Analysis of Face Detection in Low-Light Conditions Using Data Augmentation and YOLOv8

This repository contains the implementation code for a research project analyzing the robustness of YOLOv8-based face detection under low-light conditions. The study compares a baseline model trained on standard data against an augmented model trained with synthetically darkened images to evaluate whether data augmentation improves detection reliability in poor lighting.

## Overview

Face detection models often degrade significantly when deployed in low-light environments. This project investigates whether training with augmented low-light data can reduce that performance drop. The approach is straightforward:

1. Train a **baseline** YOLOv8n model on normal-lighting face images.
2. Train an **augmented** model on the same dataset expanded with synthetic low-light variations (gamma correction and brightness reduction).
3. Evaluate both models on normal and low-light validation sets, then compare robustness.

### Key Findings

| Condition | Baseline mAP@50 | Augmented mAP@50 |
|---|---|---|
| Normal Lighting | 0.903 | 0.868 |
| Low-Light | 0.705 | 0.791 |
| **Performance Drop** | **-21.97%** | **-8.90%** |

The augmented model retains significantly more of its detection capability under low-light conditions, with a performance drop roughly 13 percentage points smaller than the baseline.

<details>
<summary>Full metrics table</summary>

| Scenario | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| Baseline + Normal | 0.9133 | 0.8320 | 0.9030 | 0.6129 |
| Baseline + Low-Light | 0.8908 | 0.6220 | 0.7047 | 0.4339 |
| Augmented + Normal | 0.9113 | 0.7856 | 0.8682 | 0.5726 |
| Augmented + Low-Light | 0.8918 | 0.7151 | 0.7910 | 0.5145 |

</details>

## Dataset

The project uses a subset of the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset, reorganized into YOLO format. The dataset is not included in this repository due to its size. To reproduce:

1. Download WIDER FACE and place it in `datasets/skripshit_yolo/`.
2. Run `organize_yolo.py` to structure images and labels.
3. Run `split_val.py` to create the validation split.

## Project Structure

```
.
├── augment_data.py            # Low-light augmentation functions (gamma, brightness)
├── batch_augment.py           # Batch processing for augmentation
├── camera_demo.py             # Real-time webcam demo (baseline vs augmented)
├── download_data.py           # Dataset download utility
├── evaluate_robustness.py     # 4-way evaluation: 2 models x 2 conditions
├── generate_lowlight_val.py   # Generate low-light validation set
├── generate_report.py         # Charts and summary report generation
├── organize_yolo.py           # Dataset organization into YOLO format
├── smart_sample.py            # Intelligent data sampling
├── split_val.py               # Train/val split
├── train.py                   # Generic training function
├── train_augmented.py         # Train augmented model
├── train_baseline.py          # Train baseline model
├── visual_comparison.py       # Side-by-side detection comparison images
├── visualize_eda.py           # Exploratory data analysis
├── skripshit_data.yaml        # YOLO dataset configuration
├── requirements.txt           # Python dependencies
├── implementation.txt         # Phase-by-phase implementation plan
└── evaluation_results/
    ├── robustness_results.csv       # Raw evaluation metrics
    ├── REPORT_SUMMARY.txt           # Text-based report
    ├── DASHBOARD_SUMMARY.png        # Summary dashboard chart
    ├── chart_4way_comparison.png    # 4-way bar chart
    ├── chart_robustness_drop.png    # Robustness drop visualization
    ├── chart_radar_lowlight.png     # Radar chart (low-light performance)
    └── chart_training_history.png   # Training curves overlay
```

## Augmentation Method

The low-light simulation applies two transformations sequentially:

- **Gamma correction** with γ = 0.4 (non-linear darkening)
- **Brightness reduction** of -30 (linear shift in HSV space)

These parameters were chosen to approximate realistic low-light degradation without making faces completely invisible to human observers.

## Reproducing the Experiments

### Requirements

```
pip install -r requirements.txt
```

The project requires Python 3.8+ and a CUDA-capable GPU is recommended for training.

### Training

```bash
# Train baseline model
python train_baseline.py

# Train augmented model
python train_augmented.py
```

Both scripts train YOLOv8n for 50 epochs at 640px image size with batch size 16.

### Evaluation

```bash
# Generate low-light validation set
python generate_lowlight_val.py

# Run 4-way robustness evaluation
python evaluate_robustness.py

# Generate visual comparison images
python visual_comparison.py

# Generate charts and report
python generate_report.py
```

### Live Demo

```bash
# Real-time webcam comparison
python camera_demo.py
```

Controls: `L` to toggle low-light simulation, `S` to screenshot, `Q` to quit.

## Model

Architecture: **YOLOv8n** (Nano variant, ~3.2M parameters)  
Framework: [Ultralytics](https://github.com/ultralytics/ultralytics)  
Task: Single-class face detection

Model weights are excluded from this repository. After training, they are saved under `runs/detect/`.

## License

This project was developed as part of an undergraduate thesis. The WIDER FACE dataset is subject to its own [license terms](http://shuoyang1213.me/WIDERFACE/).
