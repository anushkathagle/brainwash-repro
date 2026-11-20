#!/usr/bin/env bash
# Submit PureVQ-GAN-defended stage-4 evaluations (thesis RQ1, layer one: input purification).
#
# Re-uses the BrainWash noise already produced by repro/sweep (stage 3) and repro/defense/stage4_layer1b.sbatch (a copy of
# stage4.sbatch that runs purify_layer1b.py), with PURIFIER set, so the only thing that differs from the baseline is the purifier.
# Results go to repro/sweep/purevq_results/<base_tag>__pvq-<purifier_tag>.log ; compare with
#   python repro/defense/collect_defense.py
#
# Two ways the runs are found:
#   cifar100 / mini : one row per (dataset, method) in repro/sweep/configs.tsv (experiment, lasttask, tasknum, lambda).
#   tiny            : NO configs.tsv row. The noise folders repro/sweep/noise/${PREFIX}_<method>_<mode>_eps<eps>[_s<N>] are
#                     scanned (PREFIX defaults to tiny20, as in submit_purify_tiny.sh); lambda / lambda_emp are read from the
#                     noise pkl inside the job (LAMB=auto), so they are never typed here.
#
# USAGE
#   ACCT=<acct> PURIFIERS="repro/artifacts/purifier/<variant>/purifier.pt" repro/defense/submit_purevqgan.sh <cifar100|mini|tiny> [method]
# EXAMPLES
#   DRY_RUN=1 ACCT=gvresearch_cr_default PURIFIERS="repro/artifacts/purifier/A_c10_K512_L8/purifier.pt" \
#       repro/defense/submit_purevqgan.sh cifar100 ewc
#   # Tiny-ImageNet with a 64x64 purifier, every method that has noise folders, cautious + clean, seeds 0-2:
#   DRY_RUN=1 ACCT=open PURIFIERS="repro/artifacts/purifier/F_stl10_K512_L16/purifier.pt" MODES="clean cautious" SEEDS="0 1 2" \
#       repro/defense/submit_purevqgan.sh tiny
#   # codebook-capacity sweep (RQ1 strength axis), cautious attacker only, 5 seeds:
#   PURIFIERS="$(ls repro/artifacts/purifier/*/purifier.pt)" MODES="clean cautious" SEEDS="0 1 2 3 4" \
#       ACCT=... repro/defense/submit_purevqgan.sh cifar100
#
# ENV: ACCT(req) PURIFIERS(req, space-separated purifier.pt paths) GRES(gpu:a100:1) PART(standard) TIME(unset = sbatch default)
#      EPSILONS("0.1 0.3") MODES("clean uniform cautious reckless") SEEDS("0") PURIFY_PASSES(1) DRY_RUN(0|1)
#      tiny only: PREFIX(tiny20) METHODS(all found in the noise folders) TINY_EXPERIMENT(split_tiny_imagenet)
#                 TINY_LASTTASK(19) TINY_TASKNUM(20)
#      Anything else you export (ENV_PREFIX, NEPOCHS, ...) is passed through to the job unchanged.
# The purifier tag is the name of the directory holding purifier.pt (the variant); passes!=1 adds _p<N>.
# A purifier whose training resolution does not match the dataset (32 cifar100, 84 mini, 64 tiny) is skipped.
# Seed handling matches repro/sweep: seed 0 = un-suffixed noise/eval tags, seed N>=1 = _s<N>.
# clean/uniform only use a noise pkl for its victim checkpoint (and eps): reckless is preferred, else cautious.
set -eo pipefail
cd "$(dirname "$0")/../.."          # -> repo root

: "${ACCT:?set ACCT=<your SLURM account>}"
: "${PURIFIERS:?set PURIFIERS=\"path/to/purifier.pt [...]\"}"
: "${GRES:=gpu:a100:1}"
: "${PART:=standard}"
: "${TIME:=}"
: "${EPSILONS:=0.1 0.3}"
: "${MODES:=clean uniform cautious reckless}"
: "${SEEDS:=0}"
: "${PURIFY_PASSES:=1}"
: "${DRY_RUN:=0}"
: "${PREFIX:=tiny20}"
: "${METHODS:=}"
: "${TINY_EXPERIMENT:=split_tiny_imagenet}"
: "${TINY_LASTTASK:=19}"
: "${TINY_TASKNUM:=20}"
FILTER_DS="${1:-}"; FILTER_M="${2:-}"
CONF="repro/sweep/configs.tsv"
RES="repro/sweep/purevq_results"
mkdir -p logs "${RES}"
n_sub=0; n_skip=0

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

# purifier resolution must match the dataset (the VQ-GAN is fully convolutional, so a mismatch would run silently)
expected_size() { case "$1" in split_cifar100) echo 32 ;; split_mini_imagenet) echo 84 ;; split_tiny_imagenet) echo 64 ;; *) echo "" ;; esac; }
declare -A PUR_SIZE=()
cache_purifier_size() {  # fills PUR_SIZE[path] with img_size, or "?" if it cannot be read here (e.g. no torch on this node)
  local p="$1"
  if [[ -z "${PUR_SIZE[$p]:-}" ]]; then
    PUR_SIZE[$p]=$(python -c "import torch,sys; ck=torch.load(sys.argv[1], map_location='cpu'); print(ck['config']['img_size'])" "$p" 2>/dev/null || echo "?")
  fi
}

eval_one() {  # base_tag variant noise_tag seed purifier ptag   (reads: experiment method lamb lamb_emp lasttask tasknum)
  local base="$1" variant="$2" ntag="$3" seed="$4" pur="$5" ptag="$6"
  local et="${base}__pvq-${ptag}"
  if grep -qs "After BWT" "${RES}/${et}.log"; then echo "  [skip] ${et} (done)"; n_skip=$((n_skip+1)); return 0; fi
  if ! noise_exists "${ntag}"; then
    echo "  [MISSING noise] ${ntag} (no fallback available either) -> skip ${et}"; n_skip=$((n_skip+1)); return 0
  fi
  submit --account="${ACCT}" --partition="${PART}" --gres="${GRES}" ${TIME:+--time="${TIME}"} --job-name="pvq_${et:0:60}" \
    --export=ALL,EVAL_TAG="${et}",VARIANT="${variant}",EXPERIMENT="${experiment}",APPROACH="${method}",LAMB="${lamb}",LAMB_EMP="${lamb_emp}",LASTTASK="${lasttask}",TASKNUM="${tasknum}",NOISE_TAG="${ntag}",SEED="${seed}",PURIFIER="${pur}",PURIFY_PASSES="${PURIFY_PASSES}",RESULTS_DIR="${RES}" \
    repro/defense/stage4_layer1b.sbatch
  echo "  [submit] ${et}  (variant=${variant}, noise=${ntag})"
  n_sub=$((n_sub+1))
}

run_group() {  # dshort  (reads globals: experiment method lamb lamb_emp lasttask tasknum pur ptag)
  local dshort="$1" seed sfx eps mode utag ctag kind e
  local want; want=$(expected_size "${experiment}"); cache_purifier_size "${pur}"; local have="${PUR_SIZE[$pur]}"
  if [[ "${have}" == "?" ]]; then
    echo "  [warn] could not read ${pur} here (no torch?) -- not checking its resolution (expected ${want})"
  elif [[ -n "${want}" && "${have}" != "${want}" ]]; then
    echo "  [SKIP purifier] ${ptag} was trained at ${have}x${have}, but ${experiment} needs ${want}x${want}"; return 0
  fi
  for seed in ${SEEDS}; do
    if [[ "${seed}" == "0" ]]; then sfx=""; else sfx="_s${seed}"; fi
    local last_eps=""
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
      local clean_cands=()
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
}

tiny_methods() {  # methods that have at least one noise folder ${PREFIX}_<method>_<cautious|reckless>_eps<eps>[_s<N>] with a pkl
  local d m
  for d in repro/sweep/noise/"${PREFIX}"_*; do
    [[ -d "$d" ]] && compgen -G "$d/*.pkl" >/dev/null 2>&1 || continue
    m=$(basename "$d" | sed -nE "s/^${PREFIX}_(.+)_(cautious|reckless)_eps[0-9.]+(_s[0-9]+)?$/\1/p")
    [[ -n "$m" ]] && echo "$m"
  done | sort -u
}

for pur in ${PURIFIERS}; do
  [[ -f "${pur}" ]] || { echo "MISSING purifier ${pur}"; exit 2; }
  ptag="$(basename "$(dirname "${pur}")")"
  [[ "${PURIFY_PASSES}" != "1" ]] && ptag="${ptag}_p${PURIFY_PASSES}"

  if [[ "${FILTER_DS}" == "tiny" ]]; then
    # ---- Tiny-ImageNet: discover runs from the noise folders (no configs.tsv row) ----
    experiment="${TINY_EXPERIMENT}"; lasttask="${TINY_LASTTASK}"; tasknum="${TINY_TASKNUM}"
    lamb="auto"; lamb_emp="auto"          # read from the noise pkl inside the job
    methods="${METHODS:-$(tiny_methods | tr '\n' ' ')}"
    [[ -n "${FILTER_M}" ]] && methods="${FILTER_M}"
    [[ -n "${methods// /}" ]] || { echo "no noise folders matching repro/sweep/noise/${PREFIX}_<method>_<mode>_eps<eps>[_sN]/*.pkl"; exit 2; }
    for method in ${methods}; do
      echo "===== tiny (${PREFIX}) / ${method}  purifier=${ptag}  seeds=[${SEEDS}] ====="
      run_group "${PREFIX}"
    done
  else
    # ---- CIFAR-100 / miniImageNet: one row per (dataset, method) in configs.tsv ----
    while IFS=$'\t' read -r dataset method dshort experiment lasttask tasknum lamb lamb_emp ckpt invdir; do
      [[ "$dataset" =~ ^# || -z "$dataset" ]] && continue
      [[ -n "$FILTER_DS" && "$dataset" != "$FILTER_DS" ]] && continue
      [[ -n "$FILTER_M"  && "$method"  != "$FILTER_M"  ]] && continue
      echo "===== ${dataset}/${method}  purifier=${ptag}  seeds=[${SEEDS}] ====="
      run_group "${dshort}"
    done < "${CONF}"
  fi
done
[[ "${DRY_RUN}" == "1" ]] && echo "DRY RUN - nothing submitted."
echo "submitted: ${n_sub}   skipped: ${n_skip}"
[[ "${DRY_RUN}" == "1" ]] || echo "Collect: python repro/defense/collect_defense.py"
