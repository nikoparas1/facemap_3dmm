# FaceMap 3DMM – TFLite Inference

Minimal setup to run landmark inference with Qualcomm’s FaceMap 3DMM (float TFLite) in a Jupyter notebook and save annotated images.

## What you get

- `inference.ipynb` that:
  - loads `facemap_3dmm-facial-landmark-detection-float.tflite`
  - runs a forward pass using `ai_edge_litert`
  - decodes the 265-D output into 68 landmarks via 3DMM assets
  - draws landmarks and overlays pitch/yaw/roll
  - writes results to `./results`

## Prereqs

- Python 3.9–3.12

## Quick start

```bash
# 1) clone repo
git clone https://github.com/nikoparas1/facemap_3dmm
cd facemap_3dmm

# 2) If not already included, get the TFLite model (float)
#   Download from: https://aihub.qualcomm.com/iot/models/facemap_3dmm
#   Place here:
#   ./facemap_3dmm-facial-landmark-detection-float.tflite

# 3) Follow instructions in the jupyter notebook
jupyter notebook inference.ipynb
```
