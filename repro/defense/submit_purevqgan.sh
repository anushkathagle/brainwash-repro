#!/usr/bin/env bash
# Submit PureVQ-GAN-defended stage-4 evaluations (thesis RQ1, layer one: input purification).
#
# Re-uses the BrainWash noise already produced by repro/sweep (stage 3) and the same stage4.sbatch,
# with PURIFIER set, so the only thing that differs from the undefended baseline is the purifier.
# Results go to repro/defense/results/<base_tag>__pvq-<purifier_tag>.log ; compare with
#   python repro/defense/collect_defense.py
#
# USAGE
#   ACCT=<acct> PURIFIERS="purifiers/cifar10_K512/purifier.pt" repro/defense/submit_purevqgan.sh [dataset] [method]
# EXAMPLES
#   DRY_RUN=1 ACCT=gvresearch_cr_default PURIFIERS="purifiers/cifar10_K512/purifier.pt" \
#       repro/defense/submit_purevqgan.sh cifar100 ewc
#   # codebook-capacity sweep (RQ1 strength axis), cautious attacker only, 5 seeds:
#   PURIFIERS="$(ls purifiers/cifar10_K*/purifier.pt)" MODES="clean cautious" SEEDS="0 1 2 3 4" \
#       ACCT=... repro/defense/submit_purevqgan.sh cifar100
#
# ENV: ACCT(req) PURIFIERS(req, space-separated purifier.pt paths) GRES(gpu:a100:1) PART(standard)
#      EPSILONS("0.1 0.3") MODES("clean uniform cautious reckless") SEEDS("0") PURIFY_PASSES(1) DRY_RUN(0|1)
#      METHODS_DEFAULT: all rows of repro/sweep/configs.tsv matching [dataset] [method]
# The purifier tag is the name of the directory holding purifier.pt (e.g. cifar10_K512); passes!=1 adds _p<N>.
# Seed handling matches repro/sweep: seed 0 = un-suffixed noise/eval tags, seed N>=1 = _s<N>.
# NOTE: for seeds>=1 the poisoned modes need repro/sweep/noise/<tag>_s<N>/ (run the attack sweep with SEEDS first).
set -eo pipefail
cd "$(dirname "$0")/../.."          # -> repo root

: "${ACCT:?set ACCT=<your SLURM account>}"
: "${PURIFIERS:?set PURIFIERS=\"path/to/purifier.pt [...]\"}"
: "${GRES:=gpu:a100:1}"
: "${PART:=standard}"
: "${EPSILONS:=0.1 0.3}"
: "${MODES:=clean uniform cautious reckless}"
: "${SEEDS:=0}"
: "${PURIFY_PASSES:=1}"
: "${DRY_RUN:=0}"
FILTER_DS="${1:-}"; FILTER_M="${2:-}"
CONF="repro/sweep/configs.tsv"
RES="repro/defense/results"
mkdir -p logs "${RES}"

submit() {
  if [[ "${DRY_RUN}" == "1" ]]; then echo "    + sbatch $*" >&2; return 0; fi
  sbatch "$@" >/dev/null
}

eval_one() {  # base_tag variant noise_tag seed purifier ptag
  local base="$1" variant="$2" ntag="$3" seed="$4" pur="$5" ptag="$6"
  local et="${base}__pvq-${ptag}"
  if grep -qs "After BWT" "${RES}/${et}.log"; then echo "  [skip] ${et} (done)"; return 0; fi
  if ! compgen -G "repro/sweep/noise/${ntag}/*.pkl" >/dev/null 2>&1; then
    echo "  [MISSING noise] ${ntag} -> skip ${et}"; return 0
  fi
  submit --account="${ACCT}" --partition="${PART}" --gres="${GRES}" --job-name="pvq_${et}" \
    --export=ALL,EVAL_TAG="${et}",VARIANT="${variant}",EXPERIMENT="${experiment}",APPROACH="${method}",LAMB="${lamb}",LAMB_EMP="${lamb_emp}",LASTTASK="${lasttask}",TASKNUM="${tasknum}",NOISE_TAG="${ntag}",SEED="${seed}",PURIFIER="${pur}",PURIFY_PASSES="${PURIFY_PASSES}",RESULTS_DIR="${RES}" \
    repro/sweep/stage4.sbatch
  echo "  [submit] ${et}  (variant=${variant}, noise=${ntag})"
}

for pur in ${PURIFIERS}; do
  [[ -f "${pur}" ]] || { echo "MISSING purifier ${pur}"; exit 2; }
  ptag="$(basename "$(dirname "${pur}")")"
  [[ "${PURIFY_PASSES}" != "1" ]] && ptag="${ptag}_p${PURIFY_PASSES}"
  while IFS=$'\t' read -r dataset method dshort experiment lasttask tasknum lamb lamb_emp ckpt invdir; do
    [[ "$dataset" =~ ^# || -z "$dataset" ]] && continue
    [[ -n "$FILTER_DS" && "$dataset" != "$FILTER_DS" ]] && continue
    [[ -n "$FILTER_M"  && "$method"  != "$FILTER_M"  ]] && continue
    echo "===== ${dataset}/${method}  purifier=${ptag}  seeds=[${SEEDS}] ====="
    for seed in ${SEEDS}; do
      if [[ "${seed}" == "0" ]]; then sfx=""; else sfx="_s${seed}"; fi
      last_eps=""
      for eps in ${EPSILONS}; do
        for mode in ${MODES}; do
          case "${mode}" in
            cautious|reckless) eval_one "${dshort}_${method}_${mode}_eps${eps}${sfx}" ours "${dshort}_${method}_${mode}_eps${eps}${sfx}" "${seed}" "${pur}" "${ptag}" ;;
            uniform)           eval_one "${dshort}_${method}_uniform_eps${eps}${sfx}" uniform "${dshort}_${method}_reckless_eps${eps}${sfx}" "${seed}" "${pur}" "${ptag}" ;;
          esac
        done
        last_eps="${eps}"
      done
      # clean (no attack) + purifier = the utility cost of purification; any noise pkl of this method works
      if [[ " ${MODES} " == *" clean "* ]]; then
        eval_one "${dshort}_${method}_clean${sfx}" clean "${dshort}_${method}_reckless_eps${last_eps}${sfx}" "${seed}" "${pur}" "${ptag}"
      fi
    done
  done < "${CONF}"
done
[[ "${DRY_RUN}" == "1" ]] && echo "DRY RUN - nothing submitted." || echo "Submitted. Collect: python repro/defense/collect_defense.py"
