#!/usr/bin/env bash
# Orchestrate the BrainWash Table-1 sweep: submit stage-3 (noise training) and the dependent
# stage-4 (evaluation) jobs for every (dataset, method, mode, eps) in configs.tsv.
#
# Per (dataset, method): 4 stage-3 runs {reckless,cautious} x {eps in EPSILONS}
#                        7 stage-4 evals: reckless/cautious/uniform x each eps + 1 clean.
# uniform reuses the reckless noise pkl (only its delta matters); clean reuses it too (no noise added).
#
# USAGE
#   ACCT=<acct> repro/sweep/submit_sweep.sh [dataset] [method]
# EXAMPLES
#   DRY_RUN=1 ACCT=gvresearch_cr_default repro/sweep/submit_sweep.sh                 # preview ALL, submit nothing
#   ACCT=gvresearch_cr_default ONLY=stage3 repro/sweep/submit_sweep.sh cifar100      # just CIFAR-100 noise runs
#   ACCT=gvresearch_cr_default ONLY=stage4 repro/sweep/submit_sweep.sh cifar100      # evals (after stage3 done)
#   ACCT=gvresearch_cr_default repro/sweep/submit_sweep.sh cifar100 ewc              # one cell-group, deps wired
#   SEEDS="0 1 2" GRES=gpu:a100:1 ACCT=... repro/sweep/submit_sweep.sh cifar100 mas  # multi-seed MAS (adds seeds 1,2)
#
# ENV: ACCT(req) GRES(gpu:a100_3g:1) PART(standard) EPSILONS("0.1 0.3") NEPOCHS3(5000)
#      SEEDS("0") ONLY(both|stage3|stage4) DRY_RUN(0|1)
# Multi-seed: SEEDS lists seeds to run. seed 0 keeps the original un-suffixed tags (so existing
# seed-0 noise/results are reused); seeds>=1 get _s<seed> tags. Evals with an existing result log
# (containing 'After BWT') are skipped, so re-running is cheap and resumable.
set -eo pipefail
cd "$(dirname "$0")/../.."          # -> repo root

: "${ACCT:?set ACCT=<your SLURM account, e.g. gvresearch_cr_default>}"
: "${GRES:=gpu:a100_3g:1}"
: "${PART:=standard}"
: "${EPSILONS:=0.1 0.3}"
: "${NEPOCHS3:=5000}"
: "${ONLY:=both}"
: "${DRY_RUN:=0}"
: "${SEEDS:=0}"
FILTER_DS="${1:-}"; FILTER_M="${2:-}"
CONF="repro/sweep/configs.tsv"
mkdir -p logs repro/sweep/noise repro/sweep/results

submit() {  # prints jobid (or DRYRUN); under DRY_RUN prints the command to stderr
  if [[ "${DRY_RUN}" == "1" ]]; then echo "    + sbatch $*" >&2; echo "DRYRUN"; return 0; fi
  local out; out=$(sbatch "$@"); echo "${out##* }"   # "Submitted batch job N" -> N
}
dep() { [[ "$1" =~ ^[0-9]+$ ]] && echo "--dependency=afterok:$1" || true; }

submit_stage3() {  # tag mode eps seed  -> stdout: jobid|EXISTS|DRYRUN ; progress to stderr
  local tag="$1" mode="$2" eps="$3" seed="$4"
  if compgen -G "repro/sweep/noise/${tag}/*.pkl" >/dev/null 2>&1; then
    echo "  [skip stage3] ${tag} (noise already present)" >&2; echo "EXISTS"; return 0
  fi
  if [[ "${ONLY}" == "stage4" ]]; then echo "EXISTS"; return 0; fi
  local jid
  jid=$(submit --account="${ACCT}" --partition="${PART}" --gres="${GRES}" --job-name="s3_${tag}" \
    --export=ALL,TAG="${tag}",PKL="${ckpt}",INVDIR="${invdir}",MODE="${mode}",DELTA="${eps}",SEED="${seed}",NEPOCHS="${NEPOCHS3}" \
    repro/sweep/stage3.sbatch)
  echo "  [stage3] ${tag} (seed ${seed}) -> job ${jid}" >&2
  echo "${jid}"
}

submit_eval() {  # eval_tag variant noise_tag depjid seed   (reads loop vars: experiment method lamb emp lasttask tasknum)
  local et="$1" variant="$2" ntag="$3" depj="$4" seed="$5"
  if grep -qs "After BWT" "repro/sweep/results/${et}.log"; then
    echo "  [skip stage4] ${et} (result already present)"; return 0
  fi
  submit --account="${ACCT}" --partition="${PART}" --gres="${GRES}" --job-name="s4_${et}" $(dep "${depj}") \
    --export=ALL,EVAL_TAG="${et}",VARIANT="${variant}",EXPERIMENT="${experiment}",APPROACH="${method}",LAMB="${lamb}",LAMB_EMP="${emp}",LASTTASK="${lasttask}",TASKNUM="${tasknum}",NOISE_TAG="${ntag}",SEED="${seed}" \
    repro/sweep/stage4.sbatch >/dev/null
  echo "  [stage4] ${et}  (variant=${variant}, noise=${ntag}, seed=${seed})"
}

while IFS=$'\t' read -r dataset method dshort experiment lasttask tasknum lamb lamb_emp ckpt invdir; do
  [[ "$dataset" =~ ^# || -z "$dataset" ]] && continue
  [[ -n "$FILTER_DS" && "$dataset" != "$FILTER_DS" ]] && continue
  [[ -n "$FILTER_M"  && "$method"  != "$FILTER_M"  ]] && continue
  emp="${lamb_emp}"
  echo "===== ${dataset} / ${method} (lamb=${lamb}, lamb_emp=${emp}) seeds=[${SEEDS}] ====="

  for seed in ${SEEDS}; do
    if [[ "${seed}" == "0" ]]; then sfx=""; else sfx="_s${seed}"; fi
    last_reck_tag=""; last_reck_jid=""
    for eps in ${EPSILONS}; do
      rtag="${dshort}_${method}_reckless_eps${eps}${sfx}"; rjid=$(submit_stage3 "$rtag" reckless "$eps" "$seed")
      ctag="${dshort}_${method}_cautious_eps${eps}${sfx}"; cjid=$(submit_stage3 "$ctag" cautious "$eps" "$seed")
      if [[ "${ONLY}" != "stage3" ]]; then
        submit_eval "${dshort}_${method}_reckless_eps${eps}${sfx}" ours    "$rtag" "$rjid" "$seed"
        submit_eval "${dshort}_${method}_cautious_eps${eps}${sfx}" ours    "$ctag" "$cjid" "$seed"
        submit_eval "${dshort}_${method}_uniform_eps${eps}${sfx}"  uniform "$rtag" "$rjid" "$seed"
      fi
      last_reck_tag="$rtag"; last_reck_jid="$rjid"
    done
    if [[ "${ONLY}" != "stage3" ]]; then
      submit_eval "${dshort}_${method}_clean${sfx}" clean "$last_reck_tag" "$last_reck_jid" "$seed"
    fi
  done
done < "${CONF}"

echo ""
if [[ "${DRY_RUN}" == "1" ]]; then echo "DRY RUN — nothing submitted. Re-run without DRY_RUN=1 to launch."
else echo "Submitted. Monitor: squeue -u \$USER ; collect: python repro/sweep/collect_results.py"; fi
