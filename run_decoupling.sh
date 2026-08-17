#!/usr/bin/env bash
# Decoupling experiments: can we keep DTR ~ 1 while cutting SCD?
#
# Runs sequentially on the single GPU. Each stage uses --skip-existing so the
# whole script is safe to re-run after an interruption.
set -u
cd "/home/owais/machine unlearning/ebm_unlearning" || exit 1
CFG=configs/config_domainnet_subset.yaml
RUN="conda run --no-capture-output -n myn_again python"
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 0. wait for the in-flight lambda=3 baseline curve to finish
log "waiting for lambda=3 steps sweep (baseline curve) to reach 18 checkpoints..."
while [ "$(ls outputs/checkpoints/steps/*.pt 2>/dev/null | wc -l)" -lt 18 ]; do sleep 60; done
log "baseline curve complete."

# 1. THE DIAGNOSTIC: same duration sweep with the propagation term OFF.
#    If SCD stays low while DTR -> 1, the coupling was self-inflicted by lambda_clip.
log "stage 1/2: lambda_clip=0 duration sweep (propagation removed)"
$RUN run_locality_sweep.py --config $CFG \
  --lambda-clip 0 --steps 500 1000 2250 3000 --seeds 0 1 2 \
  --tag steps_lam0 --skip-existing >> outputs/logs/sweep_steps_lam0.log 2>&1
log "stage 1 done: $(ls outputs/checkpoints/steps_lam0/*.pt 2>/dev/null | wc -l) checkpoints"

# 2. Head-only at the converged setting: removes backbone drift entirely, leaving
#    only head/energy reallocation. Completes the three-way damage decomposition.
log "stage 2/2: lambda_clip=0, finetune_stages=0 (head only), 3000 steps"
$RUN run_locality_sweep.py --config $CFG \
  --lambda-clip 0 --steps 3000 --seeds 0 1 2 --finetune-stages 0 \
  --tag headonly --skip-existing >> outputs/logs/sweep_headonly.log 2>&1
log "stage 2 done: $(ls outputs/checkpoints/headonly/*.pt 2>/dev/null | wc -l) checkpoints"

log "ALL DECOUPLING RUNS COMPLETE"
