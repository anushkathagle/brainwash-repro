#!/usr/bin/env bash
# Submit many stage4_purify_tiny.sbatch jobs at once: every combination of
# METHODS x MODES x EPSILONS x SEEDS x VARIANTS. Run from the repo root.
#
# NOISE_TAG is built as  ${PREFIX}_${method}_${mode}_eps${eps}${seed_suffix}
#   seed 0 -> no suffix, seed N -> _sN   (same convention as submit_sweep_purify.sh)
#   e.g. tiny20_rwalk_cautious_eps0.3_s2
# APPROACH / LAMB / LAMB_EMP / EVAL_TAG are filled in by stage4_purify_tiny.sbatch
# (and explicitly blanked on submit, so stray exported values in your shell cannot leak in)
# from the noise pkl, so they are never typed here.
#
# EXAMPLES
#   DRY_RUN=1 METHODS=rwalk MODES=cautious EPSILONS="0.1 0.3" SEEDS="0 1 2" ./submit_purify_tiny.sh
#   METHODS=rwalk MODES=cautious EPSILONS="0.1 0.3" SEEDS="0 1 2" ./submit_purify_tiny.sh
#   VARIANTS="clean ours uniform" METHODS="ewc rwalk" ./submit_purify_tiny.sh
#
# ENV: ACCT(open) PREFIX(tiny20) METHODS(ewc) MODES("reckless cautious") EPSILONS("0.1 0.3")
#      SEEDS(0) VARIANTS(ours) TIME(06:00:00) SIGMA_STAR(0.5) DRY_RUN(0|1)
#      Anything else (ENV_PREFIX, ID_PATH, PURIFY_CKPT, RESPACE, ...) is passed through
#      to the job unchanged -- export it before running this script.
# VARIANTS: ours = poison + purify; uniform = uniform noise + purify; clean = Arm D.
#   uniform/clean use the noise pkl only for its delta / bookkeeping, so they can share
#   any mode's pkl; they are submitted once per (method, eps, seed) / (method, seed).
set -eo pipefail
cd "$(dirname "$0")"

: "${ACCT:=open}"
: "${PREFIX:=tiny20}"
: "${METHODS:=ewc}"
: "${MODES:=reckless cautious}"
: "${EPSILONS:=0.1 0.3}"
: "${SEEDS:=0}"
: "${VARIANTS:=ours}"
: "${TIME:=06:00:00}"
: "${SIGMA_STAR:=0.5}"
: "${DRY_RUN:=0}"
RESULTS_DIR="repro/sweep/results_purify"
mkdir -p logs "${RESULTS_DIR}"

declare -A done_key=()
n_sub=0; n_skip=0

submit_one() {  # variant noise_tag seed
  local variant="$1" ntag="$2" seed="$3"
  local tag="${ntag}_${variant}_purify64_sigma${SIGMA_STAR}${RESPACE:+_sp${RESPACE}}"
  if grep -qs "After BWT" "${RESULTS_DIR}/${tag}.log"; then
    echo "  [done]    ${tag}"; n_skip=$((n_skip+1)); return
  fi
  if ! compgen -G "repro/sweep/noise/${ntag}/*.pkl" >/dev/null; then
    echo "  [MISSING] ${tag}  -- no pkl in repro/sweep/noise/${ntag}/ (stage 3 not finished?)"; n_skip=$((n_skip+1)); return
  fi
  local cmd=(sbatch --account="${ACCT}" --time="${TIME}" --job-name="p64_${tag:0:40}"
             --export=ALL,EVAL_TAG=,APPROACH=,LAMB=,LAMB_EMP=,VARIANT="${variant}",NOISE_TAG="${ntag}",SEED="${seed}",SIGMA_STAR="${SIGMA_STAR}"
             stage4_purify_tiny.sbatch)
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [dry-run] ${cmd[*]}"
  else
    echo "  [submit]  ${tag} -> $("${cmd[@]}" | awk '{print $NF}')"
  fi
  n_sub=$((n_sub+1))
}

for method in ${METHODS}; do
  for seed in ${SEEDS}; do
    sfx=""; [[ "${seed}" != "0" ]] && sfx="_s${seed}"
    for eps in ${EPSILONS}; do
      for mode in ${MODES}; do
        ntag="${PREFIX}_${method}_${mode}_eps${eps}${sfx}"
        for variant in ${VARIANTS}; do
          case "${variant}" in
            ours)    submit_one ours "${ntag}" "${seed}" ;;
            uniform) k="u_${method}_${eps}_${seed}"
                     [[ -z "${done_key[$k]:-}" ]] && { done_key[$k]=1; submit_one uniform "${ntag}" "${seed}"; } ;;
            clean)   k="c_${method}_${seed}"
                     [[ -z "${done_key[$k]:-}" ]] && { done_key[$k]=1; submit_one clean "${ntag}" "${seed}"; } ;;
            *) echo "bad VARIANT ${variant} (ours|uniform|clean)"; exit 2 ;;
          esac
        done
      done
    done
  done
done

[[ "${DRY_RUN}" == "1" ]] && echo "DRY RUN -- nothing submitted"
echo "submitted: ${n_sub}   skipped: ${n_skip}"
