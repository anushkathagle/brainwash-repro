"""
diffpure_edm.py

EDM-native implementation of DiffPure-style input purification, for use as
Layer 1 in the BrainWash defense pipeline.

Why this exists instead of porting the original DiffPure repo:
DiffPure's published code assumes a variance-preserving (VP-SDE) score model
with a discrete timestep schedule. The CIFAR-100 checkpoint you fine-tuned
with NVIDIA's EDM library uses EDM's continuous noise-level (sigma)
parameterization instead. Rather than converting your checkpoint into the
VP-SDE API (error-prone, and a second thing to validate), this module
implements purification directly against EDM's own sampler: add Gaussian
noise up to a chosen sigma_star, then run a truncated version of the EDM
deterministic (Heun, 2nd order) sampler from sigma_star down to sigma_min.
This is the mechanical equivalent of DiffPure's forward-noise-then-partial-
reverse process, just expressed in the noise parameterization your model
actually uses.

REQUIREMENTS
- torch_utils and dnnlib from the NVlabs/edm repo must be importable
  (they are required to unpickle the network checkpoint; the class
  definitions are stored inside the pickle itself). Add the edm repo root
  to PYTHONPATH, e.g.:
      export PYTHONPATH=$PYTHONPATH:/path/to/edm
- The checkpoint is assumed to be a standard EDM training-run pickle with
  an 'ema' key holding the EMA network, i.e. what NVIDIA's train.py writes
  out to training-state-*.pkl / network-snapshot-*.pkl.

ASSUMPTION TO VERIFY BEFORE TRUSTING RESULTS
EDM's reference CIFAR checkpoints are trained on images scaled to [-1, 1].
This module assumes your fine-tuned checkpoint follows that same
convention (data_range='pm1' below). If your fine-tuning script rescaled
CIFAR-100 differently, change data_range to 'zero_one' or adjust
img_to_model / model_to_img. Getting this wrong will not crash anything,
it will just silently purify garbage, so sanity-check on a handful of
CLEAN images first (see sanity_check() at the bottom) before running this
against the poisoned task.
"""

import pickle
import numpy as np
import torch

try:
    import dnnlib  # noqa: F401  (required so the pickle can deserialize the network class)
except ImportError as e:
    raise ImportError(
        "dnnlib not importable. Add the NVlabs/edm repo root to PYTHONPATH "
        "(export PYTHONPATH=$PYTHONPATH:/path/to/edm) before importing this module."
    ) from e

# ---------------------------------------------------------------------------
# numpy >=2.0 renamed its internal numpy.core -> numpy._core. Checkpoints
# pickled under a newer numpy (e.g. Colab's numpy 2.1.3) reference the new
# name internally; older numpy installs paired with torch 1.12.1 only have
# the old name. Confirmed via direct test: checkpoint fine-tuned under
# numpy 2.1.3 / torch 2.11 loads and runs correctly under torch 1.12.1 once
# this alias is in place -- this is the only compatibility gap that showed
# up in practice, nothing deeper. Aliasing here means every caller of
# load_edm_net() gets this for free without needing to remember it.
import numpy as np  # noqa: E402
import numpy.core  # noqa: E402
import numpy.core.multiarray  # noqa: E402
import numpy.core.numeric  # noqa: E402
import sys as _sys  # noqa: E402

_sys.modules.setdefault("numpy._core", numpy.core)
_sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
_sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)


# ---------------------------------------------------------------------------
# Network loading
# ---------------------------------------------------------------------------

def load_edm_net(pkl_path, device="cuda", use_ema=True):
    """Load a fine-tuned EDM checkpoint and return the eval-mode network.

    pkl_path: path to the .pkl produced by NVIDIA's EDM training script.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    key = "ema" if (use_ema and "ema" in data) else "net"
    if key not in data:
        # some snapshot formats store the network directly
        net = data
    else:
        net = data[key]
    net = net.to(device)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Truncated EDM (Heun 2nd order) sampler, starting from sigma_star instead
# of sigma_max, and starting from a noised real image instead of pure noise.
# ---------------------------------------------------------------------------

def _build_truncated_schedule(sigma_star, sigma_min, num_steps, rho, device):
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_star ** (1 / rho)
        + step_indices / max(num_steps - 1, 1) * (sigma_min ** (1 / rho) - sigma_star ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # final step lands at sigma=0
    return t_steps


def img_to_model(x01, data_range="pm1"):
    """[0,1] batch -> whatever range the diffusion model expects."""
    if data_range == "pm1":
        return x01 * 2 - 1
    return x01


def model_to_img(x_model, data_range="pm1"):
    """Model-range batch -> [0,1], clamped."""
    if data_range == "pm1":
        x_model = (x_model + 1) / 2
    return x_model.clamp(0, 1)


@torch.no_grad()
def diffpure_edm_purify(
    net,
    x01,
    sigma_star,
    num_steps=15,
    sigma_min=0.002,
    rho=7,
    S_churn=0.0,
    S_min=0.0,
    S_max=float("inf"),
    S_noise=1.0,
    data_range="pm1",
    device="cuda",
):
    """
    Purify a batch of images already in [0,1].

    x01: (B, C, H, W) tensor in [0,1]. This is exactly the format xtrain is
         in immediately after `xtrain = torch.clamp(xtrain + all_noise, 0, 1)`
         in main_baselines.py.
    sigma_star: noise level to diffuse to before reverse denoising. This is
         your calibration hyperparameter (the analog of DiffPure's t*).
         Sweep this per dataset / per epsilon, do not assume a fixed value.
    num_steps: number of reverse steps between sigma_star and sigma_min.
         DiffPure-style purification uses a short partial trajectory, not a
         full generation; start with something small (10-20) and increase
         only if the purified clean-accuracy cost is unacceptable.

    Returns a (B, C, H, W) tensor in [0,1].
    """
    x01 = x01.to(device)
    x_in = img_to_model(x01, data_range).to(torch.float64)

    t_steps = _build_truncated_schedule(sigma_star, sigma_min, num_steps, rho, device)

    # Forward step: add Gaussian noise up to sigma_star (this is the part
    # that is supposed to wash out the bounded BrainWash perturbation).
    x_next = x_in + sigma_star * torch.randn_like(x_in)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0.0
        t_hat = t_cur + gamma * t_cur
        noise_add = (t_hat ** 2 - t_cur ** 2).clamp(min=0).sqrt()
        x_hat = x_cur + noise_add * S_noise * torch.randn_like(x_cur)

        denoised = net(x_hat.to(torch.float32), t_hat.to(torch.float32), None).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat.clamp(min=1e-8)
        x_next = x_hat + (t_next - t_hat) * d_cur

        # 2nd order correction, skipped on the final (sigma -> 0) step
        if i < len(t_steps) - 2:
            denoised = net(x_next.to(torch.float32), t_next.to(torch.float32), None).to(torch.float64)
            d_prime = (x_next - denoised) / t_next.clamp(min=1e-8)
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return model_to_img(x_next.to(torch.float32), data_range)


def purify_in_chunks(net, x01, sigma_star, chunk_size=256, device="cuda", **kwargs):
    """Memory-safe wrapper: purifies a large task tensor in mini-batches."""
    out_chunks = []
    for i in range(0, x01.shape[0], chunk_size):
        chunk = x01[i : i + chunk_size]
        purified = diffpure_edm_purify(net, chunk, sigma_star, device=device, **kwargs)
        out_chunks.append(purified.cpu())
    return torch.cat(out_chunks, dim=0)


# ---------------------------------------------------------------------------
# Sanity check: run this BEFORE running against the poisoned task.
# ---------------------------------------------------------------------------

def sanity_check(net, x01_clean_sample, sigma_star=0.5, device="cuda"):
    """
    Purifies a small batch of CLEAN images and reports how much the
    purification itself perturbs them (should be small and visually the
    same class/content). If output is unrecognizable noise, data_range is
    almost certainly wrong -- flip it and re-check before running anything
    else.
    """
    purified = purify_in_chunks(net, x01_clean_sample, sigma_star, chunk_size=64, device=device)
    l2 = (purified - x01_clean_sample.to(device)).pow(2).mean().sqrt().item()
    print(f"[sanity_check] sigma_star={sigma_star} mean per-pixel L2 shift: {l2:.4f}")
    print("[sanity_check] visually inspect purified vs. original before proceeding.")
    return purified
