#!/bin/bash
#SBATCH --nodes=1                               # Specify the amount of Nodes
#SBATCH --ntasks=1                              # Specify the number of tasks
#SBATCH --cpus-per-task=16                      # Specify the number of CPUs/task
#SBATCH --gpus=1                                # Specify the number of GPUs to use
#SBATCH --partition=gpu_h100                    # Specify the node partition
#SBATCH --time=120:00:00                        # Specify the maximum time the job can run

# ============================================================
# Configuration
# ============================================================

# # ---- User-defined paths (edit these) ----
OUTPUT_FOLDER=path/to/atlas_output   # base folder for W&B output
DATA_PATH=path/to/atlas_final.zip     # dataset archive
CONTAINER=path/to/atlas.sif            # apptainer image
MODELS_DIR=path/to/atlas_models       # pretrained checkpoints
RESULTS_DIR=path/to/atlas_results     # training results output

# Personal WANDB API Key (replace with your own)
export WANDB_API_KEY=...  # Weights & Biases API key

# ---- Run settings ----
# Model variant to train: one of "vits", "vitb", "vitl"
MODEL_VARIANT=vitl

# Single seed to train with
SEED=1

NUM_WORKERS=12
DEVICES=1
BATCH_SIZE=24
MAX_EPOCHS=5
PROJECT_NAME=ATLAS-EOMT

# Per-variant settings
MODEL_NAME=ATLAS_DINOv3_${MODEL_VARIANT}
CONFIG=configs/${MODEL_NAME}.yaml
CKPT_PATH=${MODELS_DIR}/DINOv3-${MODEL_VARIANT}-256-surgenet2M.pth
ROOT_DIR=${RESULTS_DIR}/${PROJECT_NAME}/${MODEL_NAME}

# Weights & Biases settings and login
export WANDB_DIR=$OUTPUT_FOLDER/wandb
export WANDB_CONFIG_DIR=$OUTPUT_FOLDER/wandb
export WANDB_CACHE_DIR=$OUTPUT_FOLDER/wandb
export WANDB_START_METHOD="thread"
wandb login

# ============================================================
# Run
# ============================================================
echo "Training DINOv3 ${MODEL_VARIANT} (Seed: $SEED)..."
apptainer exec --nv $CONTAINER python main.py fit \
  -c $CONFIG \
  --data.path $DATA_PATH \
  --data.check_empty_targets false \
  --data.num_workers $NUM_WORKERS \
  --trainer.devices $DEVICES \
  --data.batch_size $BATCH_SIZE \
  --trainer.max_epochs $MAX_EPOCHS \
  --seed_everything $SEED \
  --model.ckpt_path $CKPT_PATH \
  --trainer.default_root_dir $ROOT_DIR \
  --trainer.callbacks+=ModelCheckpoint \
  --trainer.callbacks.dirpath=${ROOT_DIR}/checkpoints_seed${SEED} \
  --trainer.callbacks.filename="{epoch:02d}-{metrics/val_iou_all:.4f}" \
  --trainer.callbacks.monitor="metrics/val_iou_all" \
  --trainer.callbacks.mode=max \
  --trainer.callbacks.save_top_k=3 \
  --trainer.callbacks.save_last=true \
  --trainer.logger.class_path lightning.pytorch.loggers.WandbLogger \
  --trainer.logger.init_args.project $PROJECT_NAME \
  --trainer.logger.init_args.group $MODEL_NAME \
  --trainer.logger.init_args.name ${MODEL_NAME}_seed${SEED}
