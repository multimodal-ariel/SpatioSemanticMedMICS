#!/bin/bash
# results_final: Phase 1 — 1-shot. Default fusion (unnormalized weights); NO IO, late fusion. biag5 GPUs 0-3 (exps 5-7). biag5 has 4 GPUs; exp 8 runs on biag0.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE=./results_final

CUDA_VISIBLE_DEVICES=0 python main.py \
  -qp ./data_autoprompt/BTCV -sp ./data_autoprompt/MSD_Spleen -qmod ct -smod ct \
  --use_all_splits -ns 1 --pairs_per_query 5 --reg_net unigradicon --reg_finetune_steps 0 \
  --organs "spleen" --prompt_mode dense_only --save_path $BASE/crossdataset/unimodal/MSD_Spleen_support_BTCV_query/1shot --device cuda:0 &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python main.py \
  -qp ./data_autoprompt/CHAOS_MR -sp ./data_autoprompt/BTCV -qmod mri -smod ct \
  --use_all_splits -ns 1 --pairs_per_query 5 --reg_net multigradicon --reg_finetune_steps 0 \
  --organs "liver left_kidney right_kidney spleen" --prompt_mode dense_only \
  --save_path $BASE/crossdataset/crossmodal/BTCV_support_CHAOS_MR_query/1shot --device cuda:0 &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python main.py \
  -qp ./data_autoprompt/BTCV -sp ./data_autoprompt/CHAOS_MR -qmod ct -smod mri \
  --use_all_splits -ns 1 --pairs_per_query 5 --reg_net multigradicon --reg_finetune_steps 0 \
  --organs "liver left_kidney right_kidney spleen" --prompt_mode dense_only \
  --save_path $BASE/crossdataset/crossmodal/CHAOS_MR_support_BTCV_query/1shot --device cuda:0 &
PID3=$!

wait $PID1 $PID2 $PID3
echo "biag5 1-shot done (exps 5-7)."
