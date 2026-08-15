#!/usr/bin/env bash
set -euo pipefail

: "${EVEREST_DATASET:?Set EVEREST_DATASET to the SocioSeg dataset root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="./train"
ORIGINAL_CONFIG="$CONFIG_PATH/rlvr_megatron.yaml"
TIMESTAMP=$(date +"%m_%d_%H_%M")
DYNAMIC_CONFIG="$CONFIG_PATH/rlvr_megatron_${TIMESTAMP}.yaml"

sed '
  s|exp_name: "qwen2_5_vl_3B_socioseg"|exp_name: "qwen2_5_vl_3B_socioseg_'"${TIMESTAMP}"'"|g;
  s|logging_dir: ./output/train/logs|logging_dir: ./output/train/'"${TIMESTAMP}"'/logs|g;
  s|output_dir: ./output/train$|output_dir: ./output/train/'"${TIMESTAMP}"'|g;
  s|output_dir: ./output/train/checkpoint|output_dir: ./output/train/'"${TIMESTAMP}"'/checkpoint|g;
  s|log_dir: ./output/train/tensorboard|log_dir: ./output/train/'"${TIMESTAMP}"'/tensorboard|g;
' "$ORIGINAL_CONFIG" > "$DYNAMIC_CONFIG"

export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python start_rlvr_socioseg_pipeline.py --config_path "$CONFIG_PATH" --config_name "rlvr_megatron_${TIMESTAMP}"
