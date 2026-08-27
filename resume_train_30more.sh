#!/bin/bash
# Continue 4 CaS-DETR ablation experiments (dairv2x) for +30 epochs each.
# Resumes from last.pth (epoch 99) -> epoches=130 -> trains epochs 100..129.
# Also merges existing D-FINE/DEIM stage1+stage2 logs into continuous history.
#
# Run this script and it will wait for any currently running train.py to
# finish before starting the CaS-DETR resume runs.  All 4 experiments run
# sequentially; does NOT auto-start when GPU frees — run it manually.
set -u

PY=/root/autodl-tmp/cas_trt_env/bin/python
ROOT=/root/autodl-tmp/CaS_DETR
LOGDIR=$ROOT/resume10_logs
mkdir -p "$LOGDIR"

POLL_INTERVAL=60   # seconds between GPU-busy checks

# Reduce dataloader workers + disable pin_memory to avoid system-RAM OOM.
# tuning= suppresses the base config's tuning field so resume doesn't clash
# with it (train.py raises RuntimeError if both -r and tuning are active).
DL_OVERRIDE="train_dataloader.num_workers=4 val_dataloader.num_workers=4 train_dataloader.pin_memory=False val_dataloader.pin_memory=False tuning="

# ── wait for any train.py to free the GPU ──
wait_for_gpu() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检查 GPU 是否被占用 (train.py) ..."
  while true; do
    if pgrep -f 'train\.py' > /dev/null 2>&1; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 仍被占用，等待 ${POLL_INTERVAL}s ..."
      sleep $POLL_INTERVAL
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 空闲，可以开始了！"
      break
    fi
  done
  sleep 10  # brief cooldown, same as run_fasterrcnn_after_dfine.sh
}

# ── merge old D-FINE/DEIM stage1+stage2 logs (idempotent) ──
merge_logs() {
  if [ -f "$LOGDIR/.logs_merged" ]; then
    echo "[merge_logs] already merged, skip."
    return
  fi
  for model in dfine deim; do
    local s1="$LOGDIR/${model}_dairv2x.log"
    local s2="$LOGDIR/${model}_dairv2x_20more.log"
    if [ -f "$s1" ] && [ -f "$s2" ]; then
      echo "[merge_logs] merging $model stage1+stage2 ..."
      cat "$s1" "$s2" > "$LOGDIR/${model}_dairv2x_merged.log"
      mv "$LOGDIR/${model}_dairv2x_merged.log" "$s1"
      echo "[merge_logs]   -> $s1 (continuous history)"
    else
      echo "[merge_logs] $model: missing one or both parts, skip ($s1 / $s2)"
    fi
  done
  touch "$LOGDIR/.logs_merged"
}

# ── reusable resume runner ──
run_one() {
  local framework=$1 cfg=$2 ckpt=$3 protocol_flag=$4 epoch_override=$5 logname=$6
  echo "===== START $logname $(date) =====" | tee -a "$LOGDIR/$logname.log"
  ( cd "$ROOT/experiments/$framework" && \
    $PY train.py -c "$cfg" -r "$ckpt" $protocol_flag \
        -u "$epoch_override" $DL_OVERRIDE vis_after_train=false 2>&1 ) | tee -a "$LOGDIR/$logname.log"
  echo "===== END $logname $(date) =====" | tee -a "$LOGDIR/$logname.log"
}

echo "=============================================="
echo "CaS-DETR ablation +30 epoch resume script"
echo "Started at $(date)"
echo "=============================================="

# ── 0. Wait for GPU ──
wait_for_gpu

# ── 1. Merge old logs ──
merge_logs

# ── 2. CaS-DETR: cap05x (epoch 100→129) ──
run_one CaS-DETR \
  configs/dataset/ablation/archive/cas_deim_moe4_cap05x_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml \
  "$ROOT/experiments/CaS-DETR/outputs/dairv2x/ablation/cas_deim_moe4_cap05x_cass_caip_base03_a10_hgnetv2_s_dairv2x/last.pth" \
  --dairv2x-vehicle8 "epoches=130" \
  cas_deim_moe4_cap05x_dairv2x

# ── 3. CaS-DETR: cap1x (epoch 100→129) ──
run_one CaS-DETR \
  configs/dataset/ablation/archive/cas_deim_moe4_cap1x_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml \
  "$ROOT/experiments/CaS-DETR/outputs/dairv2x/ablation/cas_deim_moe4_cap1x_cass_caip_base03_a10_hgnetv2_s_dairv2x/last.pth" \
  --dairv2x-vehicle8 "epoches=130" \
  cas_deim_moe4_cap1x_dairv2x

# ── 4. CaS-DETR: cap2x (epoch 100→129) ──
run_one CaS-DETR \
  configs/dataset/ablation/archive/cas_deim_moe4_cap2x_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml \
  "$ROOT/experiments/CaS-DETR/outputs/dairv2x/ablation/cas_deim_moe4_cap2x_cass_caip_base03_a10_hgnetv2_s_dairv2x/last.pth" \
  --dairv2x-vehicle8 "epoches=130" \
  cas_deim_moe4_cap2x_dairv2x

# ── 5. CaS-DETR: cass_caip (epoch 100→129) ──
run_one CaS-DETR \
  configs/dataset/ablation/archive/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x.yml \
  "$ROOT/experiments/CaS-DETR/outputs/dairv2x/ablation/cas_deim_moe4_cass_caip_base03_a10_hgnetv2_s_dairv2x/last.pth" \
  --dairv2x-vehicle8 "epoches=130" \
  cas_deim_moe4_cass_caip_dairv2x

echo ""
echo "=============================================="
echo "ALL DONE $(date)"
echo "=============================================="
