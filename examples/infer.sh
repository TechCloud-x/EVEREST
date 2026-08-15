#!/usr/bin/env bash
set -euo pipefail

: "${EVEREST_CHECKPOINT:?Set EVEREST_CHECKPOINT to a trained checkpoint path}"
: "${EVEREST_DATASET:?Set EVEREST_DATASET to the SocioSeg dataset root}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
python start_rlvr_socioseg_pipeline_infer.py --config_path "./infer" --config_name rlvr_megatron
