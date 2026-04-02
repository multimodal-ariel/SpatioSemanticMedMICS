#!/bin/bash
# results_final: Phase 1 — 1-shot. Default fusion (unnormalized weights); NO IO, late fusion. biag6 GPUs 0-3 (exps 1-4).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE=./results_final

# One process per GPU; all run in parallel
CUDA_VISIBLE_DEVICES=0 python main.py \
  -qp ./data_autoprompt/CHAOS_MR -sp ./data_autoprompt/CHAOS_MR -qmod mri -smod mri \
  --use_all_splits -ns 1 --pairs_per_query 5 --reg_net unigradicon --reg_finetune_steps 0 \
  --prompt_mode dense_only --save_path $BASE/intradataset/CHAOS_MR/1shot --device cuda:0 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python main.py \
  -qp ./data_autoprompt/BTCV -sp ./data_autoprompt/BTCV -qmod ct -smod ct \
  --use_all_splits -ns 1 --pairs_per_query 5 --reg_net unigradicon --reg_finetune_steps 0 \
  --prompt_mode dense_only --save_path $BASE/intradataset/BTCV/1shot --device cuda:0 &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python main.py \
  -qp ./data_autoprompt/MSD_Spleen -sp ./data_autoprompt/MSD_Spleen -qmod ct -smod ct \
  --use_all_splits -ns 1 --pairs_per_query 5 --reg_net unigradicon --reg_finetune_steps 0 \
  --prompt_mode dense_only --save_path $BASE/intradataset/MSD_Spleen/1shot --device cuda:0 &
PID3=$!

CUDA_VISIBLE_DEVICES=3 python main.py \
  -qp ./data_autoprompt/MSD_Spleen -sp ./data_autoprompt/BTCV -qmod ct -smod ct \
  --use_all_splits -ns 1 --pairs_per_query 5 --reg_net unigradicon --reg_finetune_steps 0 \
  --organs "spleen" --prompt_mode dense_only --save_path $BASE/crossdataset/unimodal/BTCV_support_MSD_Spleen_query/1shot --device cuda:0 &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "biag6 1-shot done (exps 1-4)."
