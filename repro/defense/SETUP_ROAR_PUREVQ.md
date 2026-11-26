# Setting up PureVQ-GAN defense runs on Roar Collab (new user)

Goal: run BrainWash stage 4 with PureVQ-GAN purification (and the matching undefended references) on Roar Collab.
You do **not** need to redo stages 1–3: a BrainWash noise `.pkl` already contains the victim checkpoint. You need
(a) the code + conda env, (b) the datasets, (c) the noise folders, (d) a trained purifier.

Replace `<ACCT>` with your SLURM allocation, `<you>` with your Roar user id, and `<owner>` with the user whose noise and
purifiers you are copying.

## 0. Before you start
- A Roar Collab account and a SLURM allocation name (`<ACCT>`), e.g. your group's `..._default`.
- GitHub access to `anushkathagle/brainwash-repro` (ask to be added as a collaborator if it is private), plus a token or SSH
  key for git on Roar (see "Authentication" at the end).
- **Home is only 16 GB.** Keep the repo, conda env, data, noise and results under `/storage/work/<you>/`.

## 1. Storage layout and conda location (login node)
```bash
mkdir -p /storage/work/$USER/{bw_data,conda_envs,conda_pkgs}
ln -sfn /storage/work/$USER/bw_data ~/data            # the code reads ~/data/miniImagenet and ~/data/tiny-imagenet-200
module load anaconda3                                   # if it fails: module spider anaconda
conda config --add envs_dirs /storage/work/$USER/conda_envs
conda config --add pkgs_dirs /storage/work/$USER/conda_pkgs
```

## 2. Get the code (login node)
```bash
cd /storage/work/$USER
git clone -b claude/adoring-clarke-97rxkd https://github.com/anushkathagle/brainwash-repro.git
cd brainwash-repro
git log --oneline -3                                    # confirm you are on the latest commit
```

## 3. Conda env (login node, needs internet; ~10–20 min)
```bash
bash repro/setup_env.sh                                 # creates env "brainwash": py3.10 + torch 1.12.1+cu113
conda activate brainwash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"   # 1.12.1+cu113 0.13.1+cu113
```
PureVQ needs nothing beyond this env. (PureGen-EBM would also need `pip install huggingface_hub safetensors`.)

## 4. Datasets (login node)
```bash
python repro/download_artifacts.py --what datasets      # miniImagenet.zip + tiny-imagenet-200.zip (~1.3 GB)
bash repro/setup_data.sh                                # -> ~/data/miniImagenet, ~/data/tiny-imagenet-200, ./data (CIFAR-100)
python repro/prepare_tiny_imagenet.py                   # only for Tiny-ImageNet: builds ./data/tiny_imagenet.npz
ls ~/data/miniImagenet                                  # train_x.npy train_y.npy test_x.npy test_y.npy
```
(Add `cifar100 miniimagenet` to `--what` only if you will also re-run stages 1–3; stage 4 does not need them.)

## 5. Noise folders (copy, don't regenerate)
Stage 3 takes days of GPU time, so copy the existing noise folders. `<owner>` must give you read access
(`chmod -R g+rX` on a group-shared folder, or copy via a shared group directory / Globus).
```bash
mkdir -p repro/sweep/noise
cp -r /storage/work/<owner>/<path-to-their-repo>/repro/sweep/noise/mini_ewc_* repro/sweep/noise/   # pick what you need
ls repro/sweep/noise/                                    # e.g. mini_ewc_cautious_eps0.1, mini_ewc_cautious_eps0.1_s1, tiny20_ewc_...
```
Naming: `<dataset>_<method>_<mode>_eps<eps>` with `_s<N>` for seeds ≥ 1 (`cifar100_`, `mini_`, `tiny20_`).

## 6. Purifier
Copy the trained purifier folder (keep the variant name as the folder name):
```bash
mkdir -p repro/artifacts/purifier
cp -r /storage/work/<owner>/<path-to-their-repo>/repro/artifacts/purifier/S_stl10_K512_L21_84 repro/artifacts/purifier/
```
(Or train one: `repro/defense/README_purevqgan.md`, "Training a purifier on STL-10". Resolution must match the dataset:
32 = CIFAR-100, 84 = miniImageNet, 64 = Tiny-ImageNet.)

## 7. Checks before submitting anything
On the login node:
```bash
conda activate brainwash
python -c "from purevqgan import load_purifier; print(load_purifier('repro/artifacts/purifier/S_stl10_K512_L21_84/purifier.pt','cpu').config)"
#   -> img_size 84 for miniImageNet. This also proves the purifier loads under torch 1.12 (it was trained on torch 2.x).
```
Then one short GPU test (~10 min) that exercises CUDA + data + noise + purifier together:
```bash
salloc --account=<ACCT> --partition=standard --gres=gpu:a100:1 --time=00:30:00
conda activate brainwash
python -m purevqgan.purify --purifier repro/artifacts/purifier/S_stl10_K512_L21_84/purifier.pt \
    --experiment split_mini_imagenet --noise_pkl repro/sweep/noise/mini_ewc_cautious_eps0.1
exit
```
Expect a JSON report: `psnr_poisoned` above `input_psnr_poisoned` means the purifier moves poisoned images toward clean.

## 8. Submit (dry run first)
```bash
P84=repro/artifacts/purifier/S_stl10_K512_L21_84/purifier.pt
# defended runs
DRY_RUN=1 ACCT=<ACCT> PURIFIERS="$P84" MODES="clean cautious" SEEDS="0 1 2" TIME=04:00:00 repro/defense/submit_purevqgan.sh mini ewc
# matching undefended references through the SAME seeded code (no purifier needed)
DRY_RUN=1 ACCT=<ACCT> UNDEFENDED=1     MODES="clean cautious" SEEDS="0 1 2" TIME=04:00:00 repro/defense/submit_purevqgan.sh mini ewc
```
Read the dry-run output: no `[SKIP purifier]`, and `[MISSING noise]` only for folders you really do not have.
Then run the same two commands without `DRY_RUN=1`. Tiny-ImageNet: replace `mini ewc` with `tiny` (all methods found in
`repro/sweep/noise/tiny20_*`) or `tiny ewc`, with a 64×64 purifier.

## 9. Monitor and collect
```bash
squeue -u $USER                                          # job state
tail -f logs/<job-name>_<jobid>.out                      # live log of one job
ls repro/sweep/purevq_results/                           # one .log per finished run
python repro/defense/collect_defense.py --purifier S_stl10_K512_L21_84 --csv mini_ewc_pvq.csv
```
Re-running a submit command skips runs that already have a result, so it is safe to resubmit after failures.

## Common problems
| symptom | cause / fix |
|---|---|
| `No module named torch` in a job | conda not activated in batch: the scripts auto-detect the env; set `ENV_PREFIX=/storage/work/$USER/conda_envs/brainwash` |
| `Disk quota exceeded` | something is writing to home: check `du -sh ~/.conda ~/.cache ~/data`; move to `/storage/work` |
| job killed at the time limit | raise `TIME` (miniImageNet stage 4 is ~4× slower than CIFAR) |
| `[SKIP purifier] ... trained at 32x32` | wrong purifier for the dataset (see step 6) |
| `MISSING noise pkl` | copy that noise folder (step 5) |
| git push asks for a password | GitHub needs a token or SSH key (below) |

## Authentication (git push from Roar)
GitHub does not accept your account password. Either:
- **Token:** github.com → Settings → Developer settings → Personal access tokens → Fine-grained → only this repository,
  Contents: Read and write. Paste it as the password at `git push`. `git config --global credential.helper 'cache --timeout=28800'`
  remembers it for 8 h.
- **SSH key:** `ssh-keygen -t ed25519`, add `~/.ssh/id_ed25519.pub` on github.com (Settings → SSH keys),
  `git remote set-url origin git@github.com:anushkathagle/brainwash-repro.git`, test with `ssh -T git@github.com`.
  If it hangs, GitHub also accepts SSH on port 443 (`Hostname ssh.github.com`, `Port 443` in `~/.ssh/config`).
Never `git add .` here: `data` and `repro/sweep/noise` may be symlinks; add files by name.
