"""
diffpure_imagenet64.py

DiffPure (SDEdit-style) purification backed by OpenAI's *improved-diffusion*
UNCONDITIONAL ImageNet-64 model (imagenet64_uncond_100M_1500K.pt), for
**Tiny-ImageNet (64x64)**.

This is the Tiny-ImageNet counterpart of diffpure_guided.py (the miniImageNet
purifier). It only adds a LOADER: once loaded, the network is handed back in
the exact dict format diffpure_guided.py already consumes, so the sigma ->
timestep conversion, the DDPM reverse chain, chunking, resumability and the
cache key are all the SAME code as on miniImageNet (re-exported below). Nothing
in diffpure_guided.py is modified.

    diffpure_edm.py        ->  load_edm_net(ckpt)          (CIFAR-100, 32x32, native)
    diffpure_guided.py     ->  load_guided_net(ckpt)       (miniImageNet, 84 -> 256 -> 84)
    diffpure_imagenet64.py ->  load_imagenet64_net(ckpt)   (Tiny-ImageNet, 64x64, native)

WHY THIS MODEL (the point of the Tiny-ImageNet experiment)
    Tiny-ImageNet is 64x64 and drawn from ImageNet, and improved-diffusion ships
    an unconditional ImageNet-64 model. Using it means:
      * no resize, so no bicubic blur: the diffusion noise and the poison live
        at the same pixel scale, as on CIFAR-100;
      * a direct test of the resolution explanation for miniImageNet's results:
        same kind of content as miniImageNet, but at native resolution;
      * unconditional, so the purifier never relies on the poisoned set's labels
        (do NOT swap in imagenet64_cond_270M_250K.pt).

CHECKPOINT AND FLAGS (from the improved-diffusion README)
    https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt
    MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
    DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
    Everything else is improved-diffusion's default. Two of those defaults
    matter and differ from guided-diffusion's defaults:
      * rescale_timesteps=True  -- the model was trained seeing t * 1000/4000,
        not raw t. Getting this wrong does NOT raise an error; it silently feeds
        the network the wrong noise level and produces garbage. This is the main
        reason the config below is spelled out in full.
      * attention_resolutions="16,8", num_heads=4, channel_mult=(1,2,3,4).
    Use the L_hybrid checkpoint above, not imagenet64_uncond_vlb_100M_1500K.pt.

BACKEND
    The checkpoint was produced by `improved_diffusion`, so that package is the
    exact match and is tried first. If it is not importable, `guided_diffusion`
    (a descendant of improved-diffusion, already on PYTHONPATH for the mini
    runs) is tried with flags chosen to reproduce the same network. Either way
    load_state_dict(strict) failures are fatal -- a mismatch never passes
    silently. Force one with backend="improved" / "guided".

SIGMA_STAR ON THIS SCHEDULE (cosine, T=4000; check the "resolved sigma" log line)
    sigma*=0.1 -> 226 steps   0.25 -> 598   0.5 -> 1159   1.0 -> 1984
    With respace=1000 (optional, ~4x faster):
    sigma*=0.1 -> 57 steps    0.25 -> 150   0.5 -> 290    1.0 -> 497

SIGMA_STAR = 0
    There is no resize at native 64x64, so sigma_star=0 is an exact identity
    (diffpure_guided's "resize round trip" is a no-op here). The miniImageNet
    "Arm E" resize-only control therefore does not exist for Tiny-ImageNet: it
    is identical to the no-purify arm. That is the intended property of this
    experiment, not a gap.

REQUIREMENTS
    git clone https://github.com/openai/improved-diffusion.git
    export PYTHONPATH=$PYTHONPATH:/path/to/improved-diffusion
    (it imports mpi4py via improved_diffusion/dist_util.py only for training;
    script_util / gaussian_diffusion / unet used here do not need it.)
"""

import inspect
import os

import torch

# Reuse, unchanged, everything that does not depend on which network is loaded.
from diffpure_guided import (  # noqa: F401  (re-exported for callers)
    _sigma_schedule,
    cache_key,
    purify_in_chunks,
    sigma_to_t,
)


IMAGENET64_UNCOND_CKPT_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/march-2021/"
    "imagenet64_uncond_100M_1500K.pt"
)

# Full improved-diffusion config for imagenet64_uncond_100M_1500K.pt:
# README MODEL_FLAGS/DIFFUSION_FLAGS on top of improved-diffusion's
# model_and_diffusion_defaults().
IMAGENET64_UNCOND_CONFIG = dict(
    image_size=64,
    num_channels=128,
    num_res_blocks=3,
    num_heads=4,
    num_heads_upsample=-1,
    attention_resolutions="16,8",
    dropout=0.0,
    learn_sigma=True,
    sigma_small=False,
    class_cond=False,
    diffusion_steps=4000,
    noise_schedule="cosine",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=True,       # improved-diffusion default -- REQUIRED, see docstring
    rescale_learned_sigmas=True,  # improved-diffusion default (only affects training loss)
    use_checkpoint=False,
    use_scale_shift_norm=True,
)

# Extra flags guided-diffusion's create_model_and_diffusion needs to build the
# SAME network as improved-diffusion did (guided added these knobs later; the
# values below are the ones that reproduce improved-diffusion's UNet).
_GUIDED_EXTRA = dict(
    channel_mult="",               # -> (1,2,3,4) for image_size=64, same as improved
    num_head_channels=-1,          # use num_heads=4, as improved does
    resblock_updown=False,         # improved uses separate Up/Downsample layers
    use_fp16=False,
    use_new_attention_order=False, # improved's QKV split order
)


def _import_backend(backend):
    """Return (create_model_and_diffusion, backend_name)."""
    errors = []
    order = {"auto": ["improved", "guided"], "improved": ["improved"],
             "guided": ["guided"]}[backend]
    for name in order:
        try:
            if name == "improved":
                from improved_diffusion.script_util import create_model_and_diffusion
            else:
                from guided_diffusion.script_util import create_model_and_diffusion
            return create_model_and_diffusion, name
        except ImportError as e:
            errors.append(f"{name}_diffusion: {e}")
    raise ImportError(
        "Could not import a diffusion backend for the ImageNet-64 purifier. "
        "It is source code, not part of the .pt checkpoint:\n"
        "    git clone https://github.com/openai/improved-diffusion.git\n"
        "    export PYTHONPATH=$PYTHONPATH:$PWD/improved-diffusion\n"
        "(errors: " + "; ".join(errors) + ")"
    )


def load_imagenet64_net(ckpt_path, device="cuda", use_fp16=False, respace="",
                        backend="auto"):
    """Load the improved-diffusion ImageNet-64 unconditional model + its diffusion.

    Returns the same dict diffpure_guided.purify_in_chunks consumes:
    {'model', 'diffusion', 'image_size', 'sigma_schedule', ...}.

    respace: "" (default) = full 4000-step cosine schedule. "1000" (or any
    improved-diffusion timestep_respacing string) evenly respaces it; sigma_star
    is then matched on the respaced grid, and the reverse chain is ~4x shorter.
    use_fp16: off by default -- the 64x64 model is small and fp32 avoids any
    precision difference from the CIFAR arm. Turn on only if speed requires it.
    """
    create_model_and_diffusion, used = _import_backend(backend)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"purifier checkpoint not found: {ckpt_path}\n"
            f"download it with:  wget {IMAGENET64_UNCOND_CKPT_URL}")

    cfg = dict(IMAGENET64_UNCOND_CONFIG)
    cfg["timestep_respacing"] = str(respace or "")
    if used == "guided":
        cfg.update(_GUIDED_EXTRA)
        cfg["use_fp16"] = bool(use_fp16)

    # Keep only kwargs this backend's create_model_and_diffusion accepts
    # (improved has sigma_small; guided does not). Refuse to silently drop any
    # setting that changes the network or schedule.
    accepted = set(inspect.signature(create_model_and_diffusion).parameters)
    dropped = sorted(set(cfg) - accepted)
    harmless = {"sigma_small"}  # False == guided's implicit default
    if set(dropped) - harmless:
        raise RuntimeError(f"{used}_diffusion's create_model_and_diffusion does not accept "
                           f"{sorted(set(dropped) - harmless)}; cannot build the ImageNet-64 net "
                           "faithfully with this backend. Use backend='improved'.")
    cfg = {k: v for k, v in cfg.items() if k in accepted}

    model, diffusion = create_model_and_diffusion(**cfg)

    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Loud on purpose (same policy as diffpure_guided.load_guided_net).
        raise RuntimeError(
            f"state_dict mismatch loading {ckpt_path} with backend={used}_diffusion\n"
            f"  missing keys   : {list(missing)[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  unexpected keys: {list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}\n"
            "If this was guided_diffusion, retry with improved_diffusion on PYTHONPATH "
            "(the exact package the checkpoint was trained with)."
        )

    if not getattr(diffusion, "rescale_timesteps", False):
        raise RuntimeError("rescale_timesteps is False -- this checkpoint was trained with "
                           "rescaled timesteps and would receive the wrong noise level.")

    model.to(device)
    if use_fp16:
        model.convert_to_fp16()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    sched = _sigma_schedule(diffusion)
    print(f"[diffpure_imagenet64] loaded {os.path.basename(ckpt_path)} via {used}_diffusion "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, fp16={use_fp16}, "
          f"T={diffusion.num_timesteps}{' respaced from 4000' if respace else ''}, cosine, "
          f"max sigma={sched[-1]:.1f})")

    return {"model": model, "diffusion": diffusion, "image_size": cfg["image_size"],
            "sigma_schedule": sched, "backend": used, "respace": str(respace or "")}
