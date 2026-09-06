"""Demo-site step F: the mirror-equivariance identity, checked numerically on
the demo scenes and rendered as data the viewer can overlay.

The architecture claims two algebraic identities for the left-right mirror M
(x -> -x; azimuth -> -azimuth; mirror-paired tokens swapped):

    encode(mirror(scene))               = rho(M) encode(scene)
    denoise(flip_x(x_t), t, rho(M) z)   = flip_x(denoise(x_t, t, z))

and therefore, for the eta=0 DDIM sampler started from a mirrored noise
tensor, sample(mirror(scene), flip_x(x_T)) = flip_x(sample(scene, x_T)).

This script measures all three on every demo scene (max absolute error, for
the full system and the independent ablation, both of which are built from
the same equivariant blocks) and stores the mirrored-prompt sample so the
page can overlay it on the mirror of the original: if the identity holds the
two curves coincide.

Output: work/mirror_check.json, work/mirror_trajs.npz
Usage:  python -B build/mirror_check.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from _common import DATA, OUTPUTS, WORK

from dataset_torch import (load_split, strip_anchor_tokens,
                           strip_cross_source_edges)
from diffusion_decoder import (K, DiffusionSchedule, denormalize_kf,
                               kf_to_dense)
from eval_strict import load_model
from generator import build_vocab
from parity import Rep
from symmetry import build_mirror_perm, flip_x, mirror_scene_tensor
from train_diffusion import _to_device

CKPT = {"REL2TRAJ": OUTPUTS / "d3_csteer.pt",
        "INDEP": OUTPUTS / "e1b_indep_noanchor.pt"}
STEPS = 20


def scene_seed(scene_id: str) -> int:
    import zlib
    return 2000 + zlib.crc32(scene_id.encode()) % 100000


def ddim_from(den, sched, z, x, steps=STEPS):
    """eta=0 DDIM from a GIVEN x_T (ddim_sample draws its own noise, which is
    fine for sampling but not for a paired identity check)."""
    ts = torch.linspace(sched.T - 1, 0, steps, device=x.device).long()
    for i, t in enumerate(ts):
        eps = den(x, t.view(1), z)
        ab_t = sched.alphas_bar[t]
        x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt().clamp(min=1e-4)
              ).clamp(-1.5, 1.5)
        if i + 1 < len(ts):
            ab_p = sched.alphas_bar[ts[i + 1]]
            x = ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps
        else:
            x = x0
    return denormalize_kf(x)


def rho(rep: Rep, z: torch.Tensor) -> torch.Tensor:
    """rho(M) on the feature space: +1 on the symmetric block, -1 on the
    antisymmetric block (parity.Rep layout: [sym | anti])."""
    s = torch.ones(rep.dim, device=z.device)
    s[rep.d_plus:] = -1.0
    return z * s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    vocab = build_vocab()
    perm = build_mirror_perm(vocab)
    sched = DiffusionSchedule(device=args.device)

    manifest = json.loads((WORK / "scene_manifest.json").read_text("utf-8"))
    wanted = {c["scene"]: c for c in manifest["scenes"]}

    report, arrays = {"scenes": []}, {}
    for cond, ckpt in CKPT.items():
        enc, den, _, independent, no_anchors = load_model(str(ckpt), device,
                                                          vocab)
        ck = torch.load(str(ckpt), weights_only=False, map_location="cpu")
        rep = Rep(ck["dim"] // 2, ck["dim"] // 2)
        recs = []
        for split in ("val_iid", "test_comp"):
            rs = [r for r in load_split(DATA, split, vocab)
                  if r.row["id"] in wanted]
            if independent:
                strip_cross_source_edges(rs)
            if no_anchors:
                strip_anchor_tokens(rs, vocab["<unk>"])
            recs += rs

        for rec in recs:
            sid = rec.row["id"]
            st = _to_device(rec.tensor, device)
            st_m = mirror_scene_tensor(st, perm)
            N = len(rec.tensor.names)
            T = st.gt_az.shape[1]
            with torch.no_grad():
                z, z_m = enc.encode(st), enc.encode(st_m)
                e_enc = (z_m - rho(rep, z)).abs().max().item()

                # denoiser identity at a spread of timesteps
                gen = torch.Generator(device=device.type).manual_seed(
                    scene_seed(sid))
                x = torch.randn(N, K, 3, device=device, generator=gen)
                e_den = 0.0
                for tt in (999, 750, 500, 250, 50, 0):
                    t = torch.tensor([tt], device=device)
                    lhs = den(flip_x(x), t, z_m)
                    rhs = flip_x(den(x, t, z))
                    e_den = max(e_den, (lhs - rhs).abs().max().item())

                # full sampler: mirrored prompt + mirrored noise
                kf = ddim_from(den, sched, z, x.clone())
                kf_m = ddim_from(den, sched, z_m, flip_x(x.clone()))
                e_samp = (kf_m - flip_x(kf)).abs().max().item()
                scale = kf.abs().max().item()

                az, el, dist = (a.cpu().numpy() for a in kf_to_dense(kf, T))
                az_m, el_m, dist_m = (a.cpu().numpy()
                                      for a in kf_to_dense(kf_m, T))
            for i, n in enumerate(rec.tensor.names):
                arrays[f"{cond}|orig|{sid}|{n}"] = np.stack(
                    [rec.t, az[i], el[i], dist[i]]).astype(np.float32)
                arrays[f"{cond}|mirror|{sid}|{n}"] = np.stack(
                    [rec.t, az_m[i], el_m[i], dist_m[i]]).astype(np.float32)
            report["scenes"].append({
                "scene": sid, "cond": cond,
                "err_encoder": e_enc, "err_denoiser": e_den,
                "err_sampler_m": e_samp, "sample_scale_m": scale})
            print(f"  {sid} {cond:9} enc {e_enc:.1e}  den {e_den:.1e}  "
                  f"sampler {e_samp:.1e} (scale {scale:.2f} m)")

    worst = {c: max(s["err_sampler_m"] for s in report["scenes"]
                    if s["cond"] == c) for c in CKPT}
    report["worst_sampler_err_m"] = worst
    report["timesteps_checked"] = [999, 750, 500, 250, 50, 0]
    (WORK / "mirror_check.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    np.savez_compressed(WORK / "mirror_trajs.npz", **arrays)
    print(f"\nworst full-sampler error: {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
