#!/usr/bin/env bash
# Submit PureVQ-GAN-defended stage-4 evaluations (thesis RQ1, layer one: input purification).
#
# Re-uses the BrainWash noise already produced by repro/sweep (stage 3) and repro/defense/stage4_layer1b.sbatch (a copy of
# stage4.sbatch that runs purify_layer1b.py), with PURIFIER set, so the only thing that differs from the baseline is the purifier.
# Results go to repro/sweep/purevq_results/<base_tag>__pvq-<purifier_tag>.log ; compare with
#   python repro/defense/collect_defense.py
#
# USAGE
#   ACCT=<acct> PURIFIERS="repro/artifacts/purifier/A_c10_K512_L8/purifier.pt" repro/defense/submit_purevqgan.sh [dataset] [method]
# EXAMPLES
#   DRY_RUN=1 ACCT=gvresearch_cr_default PURIFIERS="repro/artifacts/purifier/A_c10_K512_L8/purifier.pt" \
#       repro/defense/submit_purevqgan.sh cifar100 ewc
#   # codebook-capacity sweep (RQ1 strength axis), cautious attacker only, 5 seeds:
#   PURIFIERS="$(ls repro/artifacts/purifier/*/purifier.pt)" MODES="clean cautious" SEEDS="0 1 2 3 4" \
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
RES="repro/sweep/purevq_results"
mkdir -p logs "${RES}"

submit() {
  if [[ "${DRY_RUN}" == "1" ]]; then echo "    + sbatch $*" >&2; return 0; fi
  sbatch "$@" >/dev/null
}

noise_exists() { compgen -G "repro/sweep/noise/$1/*.pkl" >/dev/null 2>&1; }
pick_noise() {  # prints the first noise tag (of the arguments) whose noise dir holds a .pkl; fails with no output if none do
  local t first="$1"
  for t in "$@"; do
    if noise_exists "$t"; then
      [[ "$t" != "$first" ]] && echo "  [fallback] ${first} not found -> using ${t}" >&2
      echo "$t"; return 0
    fi
  done
  return 1
}

eval_one() {  # base_tag variant noise_tag seed purifier ptag
  local base="$1" variant="$2" ntag="$3" seed="$4" pur="$5" ptag="$6"
  local et="${base}__pvq-${ptag}"
  if grep -qs "After BWT" "${RES}/${et}.log"; then echo "  [skip] ${et} (done)"; return 0; fi
  if ! noise_exists "${ntag}"; then
    echo "  [MISSING noise] ${ntag} (no fallback available either) -> skip ${et}"; return 0
  fi
  submit --account="${ACCT}" --partition="${PART}" --gres="${GRES}" --job-name="pvq_${et}" \
    --export=ALL,EVAL_TAG="${et}",VARIANT="${variant}",EXPERIMENT="${experiment}",APPROACH="${method}",LAMB="${lamb}",LAMB_EMP="${lamb_emp}",LASTTASK="${lasttask}",TASKNUM="${tasknum}",NOISE_TAG="${ntag}",SEED="${seed}",PURIFIER="${pur}",PURIFY_PASSES="${PURIFY_PASSES}",RESULTS_DIR="${RES}" \
    repro/defense/stage4_layer1b.sbatch
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
            uniform)  # only the noise file's checkpoint and delta (= eps) matter: prefer reckless, else the cautious file of the SAME eps
                      utag=$(pick_noise "${dshort}_${method}_reckless_eps${eps}${sfx}" "${dshort}_${method}_cautious_eps${eps}${sfx}") || utag="${dshort}_${method}_reckless_eps${eps}${sfx}"
                      eval_one "${dshort}_${method}_uniform_eps${eps}${sfx}" uniform "${utag}" "${seed}" "${pur}" "${ptag}" ;;
          esac
        done
        last_eps="${eps}"
      done
      # clean (no attack) + purifier = the utility cost of purification. The noise pkl is only used for its victim checkpoint
      # (nothing is added to the images), so ANY noise file of this dataset/method/seed works. Preference: reckless (last eps,
      # then the other eps), then cautious (same order).
      if [[ " ${MODES} " == *" clean "* ]]; then
        clean_cands=()
        for kind in reckless cautious; do
          clean_cands+=("${dshort}_${method}_${kind}_eps${last_eps}${sfx}")
          for e in ${EPSILONS}; do
            if [[ "${e}" != "${last_eps}" ]]; then clean_cands+=("${dshort}_${method}_${kind}_eps${e}${sfx}"); fi
          done
        done
        ctag=$(pick_noise "${clean_cands[@]}") || ctag="${clean_cands[0]}"
        eval_one "${dshort}_${method}_clean${sfx}" clean "${ctag}" "${seed}" "${pur}" "${ptag}"
      fi
    done
  done < "${CONF}"
done
[[ "${DRY_RUN}" == "1" ]] && echo "DRY RUN - nothing submitted." || echo "Submitted. Collect: python repro/defense/collect_defense.py"
