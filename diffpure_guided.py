"""
diffpure_guided.py

DiffPure (SDEdit-style) purification backed by OpenAI's *guided-diffusion*
ImageNet 256x256 UNCONDITIONAL model (256x256_diffusion_uncond.pt).

This is the miniImageNet / tinyImageNet counterpart to diffpure_edm.py (which
wraps an NVlabs EDM network for CIFAR-100). The two expose the SAME notation:

    diffpure_edm.py      ->  load_edm_net(ckpt)     / purify_in_chunks(net, x, sigma_star, ...)
    diffpure_guided.py   ->  load_guided_net(ckpt)  / purify_in_chunks(net, x, sigma_star, ...)

Why a different BACKBONE at all: there is no public diffusion model trained on
miniImageNet (84x84) or tinyImageNet (64x64). The only released *unconditional*
ImageNet diffusion models are OpenAI's ImageNet-64 (improved-diffusion, 100M)
and ImageNet-256 (guided-diffusion, 552M). DiffPure's own ImageNet experiments
use the 256x256 unconditional model, so that is what this module targets, with
a bicubic resize into and out of 256x256. Purification needs an *unconditional*
model because no trustworthy label exists at purification time.

WHY --sigma_star NEEDS A CONVERSION, NOT JUST A RENAME
    EDM (diffpure_edm.py) is a variance-EXPLODING (VE) model: x_sigma = x0 + sigma*eps,
    parameterized directly by a noise standard deviation sigma.
    guided-diffusion is a variance-PRESERVING (VP) DDPM: x_t = sqrt(abar_t) x0 +
    sqrt(1-abar_t) eps, parameterized by a discrete timestep t in [0, 1000).
    These are different SDEs. A bare rename (calling the timestep fraction
    "sigma_star") would silently mislabel a fraction as a noise level -- e.g. a
    previous "t*=0.15" would show up as "sigma_star=0.15" even though its actual
    noise level is sigma~0.52, not 0.15. That is worse than two flag names: it
    looks numerically comparable across CIFAR/mini when it is not.

    Instead this module takes a real --sigma_star (same physical meaning as
    diffpure_edm.py's: the standard deviation of noise added relative to a
    unit-variance signal) and converts it to the closest matching DDPM timestep
    via the standard VP<->sigma correspondence (Song et al. 2021 VP-SDE; Karras
    et al. 2022 EDM paper, Appendix B):

        sigma_t = sqrt((1 - abar_t) / abar_t)

    where abar_t is this model's cumulative-alpha schedule (linear, T=1000).
    sigma_to_t() inverts this by nearest-neighbor search over the 1000 discrete
    steps and returns the resolved sigma alongside, which is ALWAYS printed and
    logged -- never trust the requested sigma_star alone, check the resolved one.

    Caveat, stated plainly: matching sigma_star numerically across the CIFAR/EDM
    and mini/guided-diffusion backbones makes the *noise-injection step*
    comparable, but the two UNets differ in architecture, training resolution,
    and training data, so "same sigma_star" is not "same purification strength"
    in outcome -- only in input corruption. Still worth doing for a consistent
    reporting notation, but say so in the thesis methodology rather than implying
    exact equivalence. Because of this, don't assume a CIFAR-tuned sigma_star is
    optimal for mini -- run the same clean-accuracy sweep independently per
    dataset (see README_purify_mini.md).

WHAT PURIFICATION DOES HERE (the SDEdit / DiffPure recipe):
    1. map x from [0,1] to [-1,1]
    2. bicubic resize 84x84 -> 256x256
    3. resolve sigma_star -> nearest DDPM timestep t, forward-diffuse to t (adds
       Gaussian noise via q_sample)
    4. run the DDPM reverse chain from t-1 down to 0 (removes it, and with it
       most of the adversarial perturbation)
    5. bicubic resize 256x256 -> 84x84, map back to [0,1], clamp

sigma_star = 0.0 is a SPECIAL, DELIBERATE CASE: steps 3-4 are skipped and only
the resize round trip happens. That is the "resize-only" control arm. You need
it: a bicubic 84->256->84 round trip is itself a low-pass filter and removes a
meaningful chunk of an L-inf perturbation, so without this control you cannot
attribute a robustness gain to diffusion rather than to resampling.

REQUIREMENTS
    The `guided_diffusion` PACKAGE must be importable. It is source code, not a
    checkpoint dependency -- the .pt file holds only weights, and the UNet class
    that those weights are loaded into lives in that package. Either:
        git clone https://github.com/openai/guided-diffusion.git
        export PYTHONPATH=$PYTHONPATH:/path/to/guided-diffusion
    or `pip install -e /path/to/guided-diffusion`. See README_purify_mini.md.
"""

import hashlib
import math
import os

import torch
import torch.nn.functional as F


# Official MODEL_FLAGS for 256x256_diffusion_uncond.pt, from the guided-diffusion
# README. These MUST match the checkpoint exactly or load_state_dict will fail
# (or, worse, silently mismatch and produce garbage).
GUIDED_256_UNCOND_CONFIG = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    num_heads=4,
    num_heads_upsample=-1,
    num_head_channels=64,
    attention_resolutions="32,16,8",
    channel_mult="",
    dropout=0.0,
    class_cond=False,
    use_checkpoint=False,
    use_scale_shift_norm=True,
    resblock_updown=True,
    use_fp16=True,
    use_new_attention_order=False,
    learn_sigma=True,
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
)


def load_guided_net(ckpt_path, device="cuda", use_fp16=True):
    """Load the guided-diffusion 256x256 unconditional model + its diffusion.

    Returns a dict {'model', 'diffusion', 'image_size', 'sigma_schedule'} that
    purify_in_chunks consumes. Mirrors load_edm_net's role in diffpure_edm.py.
    """
    try:
        from guided_diffusion.script_util import create_model_and_diffusion
    except ImportError as e:
        raise ImportError(
            "Could not import `guided_diffusion`. It is source code, not part of "
            "the .pt checkpoint. Clone it and put it on PYTHONPATH:\n"
            "    git clone https://github.com/openai/guided-diffusion.git\n"
            "    export PYTHONPATH=$PYTHONPATH:$PWD/guided-diffusion\n"
            f"(original error: {e})"
        )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"purifier checkpoint not found: {ckpt_path}")

    cfg = dict(GUIDED_256_UNCOND_CONFIG)
    cfg["use_fp16"] = bool(use_fp16)

    model, diffusion = create_model_and_diffusion(**cfg)

    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Loud on purpose: a silent key mismatch here yields plausible-looking
        # but meaningless purified images, which would quietly corrupt results.
        raise RuntimeError(
            f"state_dict mismatch loading {ckpt_path}\n"
            f"  missing keys   : {list(missing)[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected keys: {list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}\n"
            "This usually means the MODEL_FLAGS do not match the checkpoint."
        )

    model.to(device)
    if use_fp16:
        model.convert_to_fp16()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[diffpure_guided] loaded {os.path.basename(ckpt_path)} "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
          f"fp16={use_fp16}, T={diffusion.num_timesteps})")

    return {"model": model, "diffusion": diffusion, "image_size": cfg["image_size"],
            "sigma_schedule": _sigma_schedule(diffusion)}


def _sigma_schedule(diffusion):
    """EDM-style sigma equivalent to each DDPM timestep under this diffusion's
    variance-preserving (VP) schedule: sigma_t = sqrt((1 - abar_t) / abar_t).
    Pure Python (no numpy dependency) since diffusion.alphas_cumprod is a small
    (T,) array. Monotonically increasing in t.
    """
    abar = list(diffusion.alphas_cumprod)
    return [math.sqrt((1.0 - a) / max(a, 1e-12)) for a in abar]


def sigma_to_t(net, sigma_star):
    """Map a target EDM-style sigma to the closest DDPM timestep.

    Returns (steps, resolved_sigma): `steps` is how many forward/reverse DDPM
    steps to run (0 means resize-only -- the control arm), `resolved_sigma` is
    the ACTUAL sigma that timestep corresponds to (report this, not the input,
    since the match is nearest-neighbor over 1000 discrete steps).
    """
    if sigma_star <= 0:
        return 0, 0.0

    sched = net["sigma_schedule"]
    # nearest-neighbor search; sched is monotonic so this is safe and cheap (T=1000)
    idx = min(range(len(sched)), key=lambda i: abs(sched[i] - sigma_star))
    resolved = sched[idx]

    if sigma_star > sched[-1] * 1.05:
        print(f"[diffpure_guided] WARNING: sigma_star={sigma_star} exceeds this "
              f"schedule's max reachable sigma ({sched[-1]:.2f}); clamped to t={idx}.")
    elif abs(resolved - sigma_star) > 0.2 * sigma_star:
        print(f"[diffpure_guided] NOTE: sigma_star={sigma_star} resolved to nearest "
              f"timestep sigma={resolved:.4f} ({100*abs(resolved-sigma_star)/sigma_star:.0f}% off "
              f"-- the 1000-step grid is coarse in this range).")

    return idx + 1, resolved  # steps = idx+1 forward steps, matching q_sample(t=idx)


def _resize(x, size):
    """Bicubic resize with antialiasing on downsample. No-op if already `size`."""
    if size is None or x.shape[-1] == size:
        return x
    downsampling = size < x.shape[-1]
    try:
        return F.interpolate(x, size=(size, size), mode="bicubic",
                             align_corners=False, antialias=downsampling)
    except TypeError:
        # torch < 1.11 has no `antialias=` kwarg.
        return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)


@torch.no_grad()
def _purify_batch(net, x01, steps):
    """Purify one batch of images given in [0,1]. Returns [0,1], same shape.

    `steps` is an already-resolved DDPM step count (from sigma_to_t), not a raw
    sigma -- this function is the shared inner loop, kept backbone-parameterization
    agnostic.
    """
    model, diffusion = net["model"], net["diffusion"]

    native = x01.shape[-1]
    work = net["image_size"]

    # [0,1] -> [-1,1], then up to the model's native resolution.
    x = _resize(x01, work) * 2.0 - 1.0
    x = x.clamp(-1.0, 1.0)

    if steps > 0:
        n = x.shape[0]
        # Diffuse to timestep index steps-1 (0-indexed), i.e. `steps` forward steps.
        t_batch = torch.full((n,), steps - 1, device=x.device, dtype=torch.long)
        x = diffusion.q_sample(x, t_batch)

        # Reverse DDPM chain steps-1 -> 0.
        for i in reversed(range(steps)):
            i_batch = torch.full((n,), i, device=x.device, dtype=torch.long)
            x = diffusion.p_sample(model, x, i_batch, clip_denoised=True)["sample"]

    # Back to [0,1] at the dataset's native resolution.
    x = (x + 1.0) / 2.0
    x = _resize(x, native)
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def purify_in_chunks(net, x, sigma_star, chunk_size=32, device="cuda", seed=None,
                     verbose=True, resume_path=None, checkpoint_every=1):
    """Purify a whole tensor of images in [0,1], shape [N,3,H,W].

    Returns a CPU tensor of the same shape and dtype. `sigma_star` is an EDM-style
    noise standard deviation (same notation as diffpure_edm.py's CIFAR arm) --
    internally converted to the nearest matching DDPM timestep under this model's
    VP schedule (see sigma_to_t / the module docstring for why this conversion,
    not a rename, is required). sigma_star == 0 performs the resize round trip
    only (the control arm).

    RESUMABILITY: if `resume_path` is given, progress is checkpointed to that file
    every `checkpoint_every` chunks (default: every chunk). On the next call with
    the same `resume_path`, already-purified chunks are skipped and the run
    continues from where it left off -- built for exactly this situation: a wall
    time limit or a dropped `salloc` session killing the job mid-purification,
    where re-submitting the SAME command should make forward progress instead of
    starting over. The checkpoint write is atomic (write to a .tmp file, then
    os.replace onto the real path) so a kill mid-write never corrupts it -- worst
    case you lose the last unwritten chunk, never the whole file. The checkpoint
    is deleted once purification completes; a stale one whose shape/sigma_star
    doesn't match the current call is ignored (a mismatch means a different task
    or setting, not a resumable state) and purification restarts from 0 for it.
    """
    assert x.dim() == 4 and x.shape[1] == 3, f"expected [N,3,H,W], got {tuple(x.shape)}"

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    n = x.shape[0]
    out = torch.empty_like(x, device="cpu")
    start_lo = 0

    if resume_path and os.path.isfile(resume_path):
        state = torch.load(resume_path, map_location="cpu")
        if state.get("n") == n and state.get("sigma_star") == sigma_star and \
           tuple(state.get("shape", ())) == tuple(x.shape):
            out = state["out"]
            start_lo = state["next_lo"]
            print(f"[diffpure_guided] RESUMING from checkpoint: {start_lo}/{n} images "
                  f"already done -> {resume_path}")
        else:
            print(f"[diffpure_guided] found {resume_path} but it doesn't match this "
                  f"input/sigma_star -- ignoring it and starting from 0")

    steps, resolved_sigma = sigma_to_t(net, sigma_star)
    if verbose:
        T = net["diffusion"].num_timesteps
        what = ("resize round trip only (control arm)" if steps == 0 else
                f"{steps} DDPM reverse steps (t={steps}/{T}, resolved sigma={resolved_sigma:.4f})")
        print(f"[diffpure_guided] purifying {n} images ({start_lo} already done) at "
              f"{x.shape[-1]}x{x.shape[-1]} via {net['image_size']}x{net['image_size']}: "
              f"sigma_star={sigma_star} -> {what}; chunk={chunk_size}")

    if start_lo >= n:
        if resume_path and os.path.isfile(resume_path):
            os.remove(resume_path)
        return out

    for i, lo in enumerate(range(start_lo, n, chunk_size)):
        hi = min(lo + chunk_size, n)
        chunk = x[lo:hi].to(device=device, dtype=torch.float32)
        out[lo:hi] = _purify_batch(net, chunk, steps).to("cpu", dtype=out.dtype)

        if resume_path and (i % checkpoint_every == 0 or hi >= n):
            tmp = resume_path + ".tmp"
            torch.save({"out": out, "next_lo": hi, "n": n, "sigma_star": sigma_star,
                       "shape": tuple(x.shape)}, tmp)
            os.replace(tmp, resume_path)  # atomic on POSIX -- never a half-written checkpoint

        if verbose and (lo // chunk_size) % 10 == 0:
            print(f"  [diffpure_guided] {hi}/{n}", flush=True)

    if resume_path and os.path.isfile(resume_path):
        os.remove(resume_path)  # fully done -- clean up the partial-progress marker

    return out


def cache_key(x, sigma_star, resize, seed):
    """Stable key for an on-disk purified-tensor cache.

    Hashes the *input pixels* so the key automatically distinguishes the clean,
    uniform and attacked arms without having to thread arm names through.
    """
    h = hashlib.sha1()
    h.update(x.detach().cpu().numpy().tobytes())
    h.update(repr((float(sigma_star), int(resize), int(seed))).encode())
    return h.hexdigest()[:16]
