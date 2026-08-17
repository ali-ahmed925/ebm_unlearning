#!/usr/bin/env bash
# Decoupling experiments: can we keep DTR ~ 1 while cutting SCD?
#
# The question: erasure strength drives BOTH style-invariance (DTR, wanted) and
# semantic collateral damage (SCD, hazardous). Is that coupling fundamental, or
# is it caused by the lambda_clip term, which by construction applies a
# similarity-weighted forget push to RETAIN samples?
#
# Three stages, all at matched duration and 3 seeds, isolating one source each:
#   1. lambda_clip=3, stages=1  -> baseline curve (the trade-off as measured)
#   2. lambda_clip=0, stages=1  -> removes propagation; leaves backbone interference
#   3. lambda_clip=0, stages=0  -> removes backbone drift; leaves head reallocation
#
# Every stage uses --skip-existing, so re-running after an interruption is cheap.
#
# Usage:
#   bash run_decoupling.sh                       # defaults (BS=32, good for >=24 GB)
#   BS=8 bash run_decoupling.sh                  # 6 GB card
#   CONDA_ENV=myenv BS=48 bash run_decoupling.sh
#   STAGES="2 3" bash run_decoupling.sh          # only the diagnostic stages
set -u

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

CFG=${CFG:-configs/config_domainnet_subset.yaml}
BS=${BS:-32}                       # configs are pinned to 8 for a 6 GB card
SEEDS=${SEEDS:-"0 1 2"}
STAGES=${STAGES:-"1 2 3"}
CONDA_ENV=${CONDA_ENV:-myn_again}

# Prefer the named conda env; fall back to whatever python has torch.
if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  RUN="conda run --no-capture-output -n $CONDA_ENV python"
else
  echo "!! conda env '$CONDA_ENV' not found; falling back to '$(command -v python)'"
  echo "!! override with: CONDA_ENV=<name> bash $0"
  RUN="python"
fi

mkdir -p outputs/logs
log() { echo "[$(date +%H:%M:%S)] $*"; }
have() { ls "$1"/*.pt >/dev/null 2>&1 && ls "$1"/*.pt | wc -l || echo 0; }

log "repo   : $(pwd)"
log "config : $CFG   batch=$BS   seeds=[$SEEDS]   stages=[$STAGES]"
$RUN -c "import torch;print('[env] torch',torch.__version__,'cuda',torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || exit 1

run_stage() {
  local tag=$1; shift
  log "=== $tag ==="
  $RUN run_locality_sweep.py --config "$CFG" --batch-size "$BS" --seeds $SEEDS \
       --tag "$tag" --skip-existing "$@" >> "outputs/logs/sweep_${tag}.log" 2>&1
  local rc=$?
  log "$tag finished (rc=$rc, $(have "outputs/checkpoints/$tag") checkpoints)"
  [ $rc -ne 0 ] && log "!! see outputs/logs/sweep_${tag}.log"
  return 0
}

# 1. baseline curve — the trade-off as currently measured
case " $STAGES " in *" 1 "*)
  run_stage steps --lambda-clip 3 --steps 250 500 1000 1500 2250 3000 ;;
esac

# 2. THE DIAGNOSTIC — propagation term off. If SCD stays low while DTR -> 1,
#    the coupling was self-inflicted and no new mechanism is needed.
case " $STAGES " in *" 2 "*)
  run_stage steps_lam0 --lambda-clip 0 --steps 500 1000 2250 3000 ;;
esac

# 3. head only — no backbone drift, completing the damage decomposition
case " $STAGES " in *" 3 "*)
  run_stage headonly --lambda-clip 0 --steps 3000 --finetune-stages 0 ;;
esac

log "ALL RUNS COMPLETE"
cat <<EOF

Next — score and analyse (each writes a JSON, then a table + frontier figure):

  P="\$RUN score_locality.py --config $CFG --split holdout --batch-size 64"
  $RUN score_locality.py --config $CFG --split holdout --batch-size 64 \\
      --out outputs/locality_steps_lam3.json  --unlearned outputs/checkpoints/steps/*.pt
  $RUN score_locality.py --config $CFG --split holdout --batch-size 64 \\
      --out outputs/locality_steps_lam0.json  --unlearned outputs/checkpoints/steps_lam0/*.pt
  $RUN score_locality.py --config $CFG --split holdout --batch-size 64 \\
      --out outputs/locality_headonly.json    --unlearned outputs/checkpoints/headonly/*.pt

  $RUN analyze_locality.py outputs/locality_steps_lam3.json --fig outputs/frontier_lam3.png
  $RUN analyze_locality.py outputs/locality_steps_lam0.json --fig outputs/frontier_lam0.png

The decision rule (fixed in advance):
  SCD <~ 0.15 at DTR >= 0.95  -> coupling was self-inflicted by lambda_clip
  SCD ~ 0.25-0.40 at DTR>=0.95 -> partly representational; build neighbour protection
  head-only also high          -> intrinsic to label-space reallocation; write the limit
EOF
