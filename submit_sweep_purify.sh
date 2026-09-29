#!/usr/bin/env bash
# Orchestrate DiffPure purification evals for every (dataset, method, mode, eps, sigma_star)
# combination -- the purify-pipeline counterpart to repro/sweep/submit_sweep.sh. This is what
# was missing: every log in repro/sweep/results/ from the undefended sweep exists because
# submit_sweep.sh auto-generates a unique, consistent EVAL_TAG per combination in a loop; the
# purify runs so far were hand-typed one sbatch command at a time, which is exactly how an
# EVAL_TAG/NOISE_TAG mismatch slipped through. This script removes that failure mode: EVAL_TAG
# is always derived from the SAME loop variables that pick NOISE_TAG, so they cannot diverge.
#
# REQUIRES stage-3 noise to already exist (run repro/sweep/submit_sweep.sh ONLY=stage3 first).
# This script only runs stage4_purify_mini.sbatch -- it does not train noise.
#
# Per (dataset, method) x seed x sigma_star: for each eps in EPSILONS:
#   2 purify evals: reckless(ours) / cautious(ours), + 1 uniform (reuses the reckless pkl,
#   same convention as stage4.sbatch/submit_sweep.sh -- only its delta matters for uniform)
#   + 1 clean eval per (dataset,method,seed,sigma) [not per eps -- clean has no eps]
#
# USAGE
#   ACCT=<acct> ./submit_sweep_purify.sh [dataset] [method]
# EXAMPLES
#   DRY_RUN=1 ACCT=gvresearch_crch_student ./submit_sweep_purify.sh mini ewc
#       # preview everything, submit nothing
#   ACCT=gvresearch_crch_student ./submit_sweep_purify.sh mini ewc
#       # launch reckless/cautious/uniform/clean at the default sigma_star=0.5, both eps
#   SIGMA_STARS="0 0.1 0.25 0.5 1.0 2.0" ACCT=... ./submit_sweep_purify.sh mini ewc
#       # the clean-accuracy sweep used to PICK sigma_star (run this before trusting any
#       # attacked-arm number -- see README_purify_mini.md "How to choose sigma_star")
#   PURIFY_CHUNK=32 GRES=gpu:a100:1 ACCT=... ./submit_sweep_purify.sh mini ewc
#       # confirmed full A100 -- larger batches
#   FP32=1 GRES=gpu:p100:1 ACCT=... ./submit_sweep_purify.sh mini ewc
#       # P100 (no tensor cores)
#
# ENV: ACCT(req) GRES(gpu:a100:1) PART(standard) TIME(03:00:00)
#      EPSILONS("0.1 0.3") SIGMA_STARS("0.5") SEEDS("0")
#      PURIFY_CHUNK(8) FP32(0) DRY_RUN(0|1)
#      ARMS("reckless,cautious,uniform,clean") -- restrict which arms get submitted, e.g.
#        ARMS=reckless,cautious ACCT=... ./submit_sweep_purify.sh mini ewc
#      skips uniform and clean entirely. Comma-separated, no spaces.
#
# RESUMABLE AT TWO LEVELS:
#   1. This script skips any EVAL_TAG whose repro/sweep/results_purify/<tag>.log already
#      contains "After BWT" -- safe to re-run after any partial launch.
#   2. A job that gets killed mid-purification (wall time, a dropped connection, anything)
#      is itself resumable via stage4_purify_mini.sbatch's checkpoint/resume mechanism -- just
#      re-run THIS script again with the same args, and it will re-submit exactly the
#      EVAL_TAGs that are still missing/incomplete, which will then pick up from their last
#      checkpoint rather than starting over.
set -eo pipefail
cd "$(dirname "$0")"   # -> repo root (this script lives there, next to stage4_purify_mini.sbatch)

: "${ACCT:?set ACCT=<your SLURM account, e.g. gvresearch_crch_student>}"
: "${GRES:=gpu:a100:1}"
: "${PART:=standard}"
: "${TIME:=03:00:00}"
: "${EPSILONS:=0.1 0.3}"
: "${SIGMA_STARS:=0.5}"
: "${SEEDS:=0}"
: "${PURIFY_CHUNK:=32}"
: "${FP32:=0}"
: "${DRY_RUN:=0}"
: "${ARMS:=reckless,cautious,uniform,clean}"
FILTER_DS="${1:-}"; FILTER_M="${2:-}"
CONF="repro/sweep/configs.tsv"
RESULTS_DIR="repro/sweep/results_purify"
mkdir -p logs "${RESULTS_DIR}"

[[ -f "${CONF}" ]] || { echo "FATAL: ${CONF} not found -- run this from the repo root (or check the path)"; exit 1; }
[[ -f "stage4_purify_mini.sbatch" ]] || { echo "FATAL: stage4_purify_mini.sbatch not found next to this script"; exit 1; }

has_arm() { [[ ",${ARMS}," == *",$1,"* ]]; }  # ARMS="reckless,cautious" -> has_arm uniform is false

submit() {  # prints jobid (or DRYRUN); under DRY_RUN prints the command to stderr
  if [[ "${DRY_RUN}" == "1" ]]; then echo "    + sbatch $*" >&2; echo "DRYRUN"; return 0; fi
  local out; out=$(sbatch "$@"); echo "${out##* }"   # "Submitted batch job N" -> N
}

submit_purify() {  # eval_tag variant noise_tag sigma seed  (reads loop vars: experiment method lamb emp lasttask tasknum)
  local et="$1" variant="$2" ntag="$3" sigma="$4" seed="$5"
  if grep -qs "After BWT" "${RESULTS_DIR}/${et}.log"; then
    echo "  [skip purify] ${et} (result already present)"
    return 0
  fi
  if ! compgen -G "repro/sweep/noise/${ntag}/*.pkl" >/dev/null 2>&1; then
    echo "  [SKIP purify] ${et} -- MISSING noise pkl in repro/sweep/noise/${ntag}/ (run submit_sweep.sh ONLY=stage3 first)" >&2
    return 0
  fi
  local jid
  jid=$(submit --account="${ACCT}" --partition="${PART}" --gres="${GRES}" --time="${TIME}" --job-name="s4p_${et}" \
    --export=ALL,EVAL_TAG="${et}",VARIANT="${variant}",EXPERIMENT="${experiment}",APPROACH="${method}",\
LAMB="${lamb}",LAMB_EMP="${emp}",LASTTASK="${lasttask}",TASKNUM="${tasknum}",NOISE_TAG="${ntag}",\
SIGMA_STAR="${sigma}",PURIFY_CHUNK="${PURIFY_CHUNK}",FP32="${FP32}",SEED="${seed}" \
    stage4_purify_mini.sbatch)
  echo "  [purify] ${et}  (variant=${variant}, noise=${ntag}, sigma_star=${sigma}, seed=${seed}) -> job ${jid}"
}

while IFS=$'\t' read -r dataset method dshort experiment lasttask tasknum lamb lamb_emp ckpt invdir; do
  [[ "$dataset" =~ ^# || -z "$dataset" ]] && continue
  [[ -n "$FILTER_DS" && "$dataset" != "$FILTER_DS" ]] && continue
  [[ -n "$FILTER_M"  && "$method"  != "$FILTER_M"  ]] && continue
  emp="${lamb_emp}"
  echo "===== ${dataset} / ${method} (lamb=${lamb}, lamb_emp=${emp}) seeds=[${SEEDS}] sigma_stars=[${SIGMA_STARS}] ====="

  for seed in ${SEEDS}; do
    if [[ "${seed}" == "0" ]]; then sfx=""; else sfx="_s${seed}"; fi

    for sigma in ${SIGMA_STARS}; do
      last_reck_tag=""
      for eps in ${EPSILONS}; do
        rtag="${dshort}_${method}_reckless_eps${eps}${sfx}"
        ctag="${dshort}_${method}_cautious_eps${eps}${sfx}"

        has_arm reckless && submit_purify "${rtag}_purify_sigma${sigma}" ours "$rtag" "$sigma" "$seed"
        has_arm cautious && submit_purify "${ctag}_purify_sigma${sigma}" ours "$ctag" "$sigma" "$seed"
        has_arm uniform   && submit_purify "${dshort}_${method}_uniform_eps${eps}${sfx}_purify_sigma${sigma}" uniform "$rtag" "$sigma" "$seed"

        last_reck_tag="$rtag"
      done
      has_arm clean && submit_purify "${dshort}_${method}_clean${sfx}_purify_sigma${sigma}" clean "$last_reck_tag" "$sigma" "$seed"
    done
  done
done < "${CONF}"

echo ""
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY RUN — nothing submitted. Re-run without DRY_RUN=1 to launch."
else
  echo "Submitted. Monitor: squeue -u \$USER"
  echo "Logs land in ${RESULTS_DIR}/<EVAL_TAG>.log ; results/cache in repro/artifacts/miniImagenet_files/purify_results/"
fi
