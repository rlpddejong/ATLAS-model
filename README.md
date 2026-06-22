# ATLAS-model

**Surgical Anatomy Recognition with Context Learning using Foundation Representations**

This repository contains the model implementation and training code for **ATLAS**, a
clip-level semantic segmentation model for surgical anatomy recognition in minimally
invasive surgery (MIS). It is trained and evaluated on the
[ATLAS-120k](https://github.com/TimJaspers0801/atlas) dataset — over 120,000 annotated
frames from 100 surgical videos spanning 14 procedures and 42 anatomical classes.

ATLAS builds on the [EoMT](https://github.com/tue-mps/eomt) (Encoder-only Mask Transformer) architecture, using
[SurgeNet](https://github.com/TimJaspers0801/SurgeNet)-pretrained **DINOv3** surgical
foundation backbones as the encoder. On top of standard mask classification it adds
surgical-context learning: per-procedure recognition, learnable **phase tokens**, and a
**phase contrastive** objective over short video clips.

> Part of the [ATLAS project ecosystem](https://github.com/TimJaspers0801/atlas).

## Features

- **Clip-level semantic segmentation** of surgical anatomy (operates on short video clips, not just single frames).
- **DINOv3 surgical backbones** (ViT-S / ViT-B / ViT-L) pretrained on SurgeNet.
- **Phase tokens & phase contrastive learning** — learnable tokens that capture surgical
  phase/context, trained with a contrastive loss across temporally neighbouring frames.
- **Procedure recognition** — auxiliary classification head over the 14 procedure types.
- **Temporal query learning** — queries are propagated across frames within a clip.
- Built on **PyTorch Lightning** with a config-driven `LightningCLI` interface.

## Repository structure

```
.
├── main.py                  # LightningCLI entry point (fit / validate / test)
├── train.sh                 # SLURM + Apptainer training launcher
├── configs/                 # Experiment configs (one per backbone size)
│   ├── ATLAS_DINOv3_vits.yaml
│   ├── ATLAS_DINOv3_vitb.yaml
│   └── ATLAS_DINOv3_vitl.yaml
├── models/
│   ├── atlas.py             # ATLAS model (EoMT + phase/procedure/temporal heads)
│   ├── vit.py               # DINOv3 ViT encoder wrapper
│   └── scale_block.py       # Feature upscaling block
├── training/                # Lightning modules, losses, LR schedule
├── datasets/                # ATLAS dataset + Lightning data module + transforms
├── utils/
│   └── extract_model_weights.py   # Export weights from a Lightning checkpoint to .pth
├── Dockerfile               # Container definition
├── DOCKER.md                # How to build the container for train.sh
└── requirements.txt
```

## Installation

Requires Python 3.10 and a CUDA-capable GPU.

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.7.0`, `torchvision==0.22.0`, `lightning==2.5.1`,
`timm==1.0.15`, `transformers==4.56.1`, `wandb==0.19.10`.

For a reproducible environment (and for running on HPC via Apptainer), see
[DOCKER.md](DOCKER.md).

## Data & checkpoints

- **Dataset**: download and process the ATLAS-120k data with the
  [ATLAS](https://github.com/TimJaspers0801/atlas) repository. Training expects the
  packaged dataset (e.g. `atlas.zip`), passed via `--data.path`.
- **Backbone checkpoints**: SurgeNet-pretrained DINOv3 weights
  (e.g. `DINOv3-vits-256-surgenet2M.pth`), passed via `--model.ckpt_path`.

## Download the model weights here
You can download the model weights using the provided links below.

| Model   | Variant | Download |
|---------|---------|----------|
| **DINOv1** | ViT-b | [Download](https://huggingface.co/rlpddejong/ATLAS-pretraining-weights/resolve/main/DINOv1-vitb-224-surgenet2M.pth?download=true) |
| **DINOv2** | ViT-b | [Download](https://huggingface.co/rlpddejong/ATLAS-pretraining-weights/resolve/main/DINOv2-vitb-336-surgenet2M.pth?download=true) |
| **DINOv3** | ViT-s | [Download](https://huggingface.co/rlpddejong/ATLAS-pretraining-weights/resolve/main/DINOv3-vits-256-surgenet2M.pth?download=true) |
| **DINOv3** | ViT-b | [Download](https://huggingface.co/rlpddejong/ATLAS-pretraining-weights/resolve/main/DINOv3-vitb-256-surgenet2M.pth?download=true) |
| **DINOv3** | ViT-l | [Download](https://huggingface.co/rlpddejong/ATLAS-pretraining-weights/resolve/main/DINOv3-vitl-256-surgenet2M.pth?download=true)|

## Training

### Quick start

Training is driven by `main.py` and a YAML config. Pick the backbone size via the
config file:

```bash
python main.py fit \
  -c configs/ATLAS_DINOv3_vits.yaml \
  --data.path /path/to/atlas.zip \
  --model.ckpt_path /path/to/DINOv3-vits-256-surgenet2M.pth \
  --trainer.devices 1 \
  --data.batch_size 24 \
  --trainer.max_epochs 3
```

Any config value can be overridden from the command line (CLI args take precedence over
the YAML). Three backbone sizes are provided:

| Config | Backbone |
|--------|----------|
| `configs/ATLAS_DINOv3_vits.yaml` | DINOv3 ViT-Small |
| `configs/ATLAS_DINOv3_vitb.yaml` | DINOv3 ViT-Base |
| `configs/ATLAS_DINOv3_vitl.yaml` | DINOv3 ViT-Large |

### On a SLURM cluster

[`train.sh`](train.sh) wraps the command above for a SLURM + Apptainer setup. Edit the
user-defined paths at the top of the file (data, container `.sif`, backbone checkpoints,
results directory), set `MODEL_VARIANT` to `vits`, `vitb`, or `vitl`, and submit:

```bash
sbatch train.sh
```

It runs a single model variant with a single seed and logs to Weights & Biases.

## Exporting weights

To extract a plain `.pth` state dict from a Lightning `.ckpt` checkpoint (e.g. to reuse
the trained backbone elsewhere):

```bash
python utils/extract_model_weights.py /path/to/checkpoint.ckpt
```

## Finetuned weights
The weights of the ATLAS model trained on the ATLAS-120k dataset can be found below.

| Model | Download |
|-------|----------|
| **ATLAS ViT-s** | [Download](https://huggingface.co/rlpddejong/ATLAS-finetuned-weights/resolve/main/ATLAS_vits_val_iou_all%3D0.2806.ckpt?download=true) |
| **ATLAS ViT-b** | [Download](https://huggingface.co/rlpddejong/ATLAS-finetuned-weights/resolve/main/ATLAS_vitb_val_iou_all%3D0.3498.ckpt?download=true) |
| **ATLAS ViT-l** | [Download](https://huggingface.co/rlpddejong/ATLAS-finetuned-weights/resolve/main/ATLAS_vitl_val_iou_all%3D0.3837.ckpt?download=true) |

