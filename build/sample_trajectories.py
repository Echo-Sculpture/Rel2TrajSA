"""Demo-site step B: trajectories for all five demo conditions.

Conditions (page column order):
  GT        frozen ground-truth trajectories from the dataset npz
  REL2TRAJ  the full system (relation graph + anchors, equivariant keyframe
            diffusion)                        -- outputs/d3_csteer.pt
  INDEP     the same model with cross-source information flow removed and
            anchor tokens withheld -- a single-variable control, NOT a
            reimplementation of any published system (main-table row 4)
                                              -- outputs/e1b_indep_noanchor.pt
  RULE      grammar constraint solver (plan_rule_v2): motions + relations,
            no anchors
  LLM       LLM-direct (qwen3:8b via Ollama, temp 0, thinking off, 2 retries)

Sampling discipline mirrors the listening test: PER-SCENE deterministic seed
(2000 + crc32(scene_id) % 100000 + draw index), no guidance, no selection by
any relation metric. Each per-scene RSR is recorded so the page prints the
real number next to the audio instead of a claim.

Both diffusion conditions produce M draws (default 10, the protocol the
paper's tables use). Draw 1 is the FIRST draw -- it is the one that gets
rendered to audio, so nothing on the page is a hand-picked sample; the other
draws are trajectory-only and let the viewer see the one-to-many spread.

The learned baseline is sampled in its OWN training regime (cross-source
edges stripped, anchor tokens -> unk), exactly as eval_strict.py does it.

Reads the research repo read-only; writes only under Rel2TrajSA/work.

Output: work/demo_trajs.npz   ("{cond}|{scene}|{source}" -> [4, T])
        work/demo_trajs_report.json
Usage:  python -B build/sample_trajectories.py [--device cuda] [--skip-llm]
"""

from __future__ import annotations

import argparse
import json
import time
import zlib

import numpy as np
import torch

from _common import DATA, OUTPUTS, WORK

from dataset import EvalItem
from dataset_torch import (load_split, strip_anchor_tokens,
                           strip_cross_source_edges)
from diffusion_decoder import DiffusionSchedule, ddim_sample, kf_to_dense
from eval_strict import load_model
from generator import build_vocab
from relations import scene_rsr
from run_rule_splits import _relations_of
from scene import Scene, SourceEvent, Trajectory
from train_diffusion import _to_device

STEPS = 20
CKPT = {"REL2TRAJ": OUTPUTS / "d3_csteer.pt",
        "INDEP": OUTPUTS / "e1b_indep_noanchor.pt"}
MODELS: dict = {}


def scene_seed(scene_id: str) -> int:
    """Same pre-registered per-scene seed rule as the listening test."""
    return 2000 + zlib.crc32(scene_id.encode()) % 100000


def as_scene(names, t, az, el, dist, prompt="") -> Scene:
    sc = Scene(prompt=prompt)
    for i, n in enumerate(names):
        sc.add(SourceEvent(n, Trajectory(t, az[i], el[i], dist[i])))
    return sc


def rsr_of(scene: Scene, rels) -> dict:
    r = scene_rsr(scene, rels)
    per = {}
    for pr, rel in zip(r["per_relation"], rels):
        key = f"{pr['type']}|{rel.subject}|{rel.reference}"
        per[key] = None if pr["rsr"] != pr["rsr"] else round(float(pr["rsr"]), 3)
    mean = r["mean_rsr"]
    return {"mean_rsr": None if mean != mean else round(float(mean), 3),
            "per_relation": per}


def sample_model(cond, rec, device, sched, draw: int = 0) -> tuple:
    enc, den, *_ = MODELS[cond]
    st = _to_device(rec.tensor, device)
    with torch.no_grad():
        z = enc.encode(st)
    gen = torch.Generator(device=device.type).manual_seed(
        scene_seed(rec.row["id"]) + draw)
    kf = ddim_sample(den, sched, z, n_sources=len(rec.tensor.names),
                     steps=STEPS, generator=gen)
    az, el, dist = kf_to_dense(kf, rec.tensor.gt_az.shape[1])
    return az.cpu().numpy(), el.cpu().numpy(), dist.cpu().numpy()


def stack_scene(sc: Scene, names) -> tuple:
    return (np.stack([sc.get(n).trajectory.azimuth for n in names]),
            np.stack([sc.get(n).trajectory.elevation for n in names]),
            np.stack([sc.get(n).trajectory.distance for n in names]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--m", type=int, default=10,
                    help="draws per diffusion condition (draw 0 = audio)")
    ap.add_argument("--maxiter", type=int, default=200)
    args = ap.parse_args()

    manifest = json.loads((WORK / "scene_manifest.json").read_text("utf-8"))
    scenes = manifest["scenes"]
    device = torch.device(args.device)
    vocab = build_vocab()
    sched = DiffusionSchedule(device=args.device)

    rows = {}
    for split in ("val_iid", "test_comp"):
        for r in (json.loads(l) for l in
                  open(DATA / f"scenes_{split}.jsonl", encoding="utf-8")):
            rows[r["id"]] = r
    gt_npz = {s: np.load(DATA / f"trajs_{s}.npz")
              for s in ("val_iid", "test_comp")}

    wanted = {c["scene"] for c in scenes}
    recs = {}
    for cond in ("REL2TRAJ", "INDEP"):
        MODELS[cond] = load_model(str(CKPT[cond]), device, vocab)
        _, _, _, independent, no_anchors = MODELS[cond]
        print(f"[{cond}] {CKPT[cond].name}: independent={independent} "
              f"no_anchors={no_anchors}")
        for split in ("val_iid", "test_comp"):
            rs = [r for r in load_split(DATA, split, vocab)
                  if r.row["id"] in wanted]
            if independent:
                strip_cross_source_edges(rs)
            if no_anchors:
                strip_anchor_tokens(rs, vocab["<unk>"])
            for r in rs:
                recs[(cond, r.row["id"])] = r
    assert not MODELS["REL2TRAJ"][3] and MODELS["INDEP"][3], \
        "checkpoint roles mixed up (REL2TRAJ joint, INDEP independent)"

    llm_cfg = None
    if not args.skip_llm:
        from llm import LLMConfig, is_available
        llm_cfg = LLMConfig()
        if not is_available(llm_cfg):
            print(f"WARNING: {llm_cfg.model} unavailable -- LLM column skipped")
            llm_cfg = None

    from rule_planner import plan_rule_v2
    arrays, report = {}, []
    for c in scenes:
        sid, split = c["scene"], c["split"]
        row = rows[sid]
        names = [s["name"] for s in row["sources"]]
        rels = _relations_of(row)
        t = gt_npz[split][f"{sid}__{names[0]}"][0]

        conds: dict[tuple[str, int], tuple] = {}
        gt = np.stack([gt_npz[split][f"{sid}__{n}"] for n in names])
        conds[("GT", 0)] = (gt[:, 1], gt[:, 2], gt[:, 3], 0.0)

        for cond in ("REL2TRAJ", "INDEP"):
            for j in range(args.m):
                t0 = time.time()
                az, el, dist = sample_model(cond, recs[(cond, sid)], device,
                                            sched, draw=j)
                conds[(cond, j)] = (az, el, dist, time.time() - t0)

        t0 = time.time()
        sc = plan_rule_v2(row["prompt"], row["sources"], rels,
                          float(row["duration"]), maxiter=args.maxiter)
        conds[("RULE", 0)] = (*stack_scene(sc, names), time.time() - t0)

        if llm_cfg is not None:
            from llm import LLMError
            from llm_direct import plan_llm_direct
            item = EvalItem(id=sid, prompt=row["prompt"],
                            duration=float(row["duration"]),
                            sources={s["name"]: s.get("motion", "static")
                                     for s in row["sources"]},
                            relations=[], raw=row)
            t0, err = time.time(), "unknown"
            for _ in range(3):
                try:
                    sc = plan_llm_direct(item, llm_cfg)
                    conds[("LLM", 0)] = (*stack_scene(sc, names),
                                         time.time() - t0)
                    break
                except (LLMError, ValueError, KeyError, TypeError) as e:
                    err = f"{type(e).__name__}: {e}"
            else:
                print(f"  {sid} LLM FAILED after 3 attempts: {err[:80]}")

        line = [f"{sid} [{split[:4]}]"]
        for (cond, draw), (az, el, dist, secs) in conds.items():
            for i, n in enumerate(names):
                arrays[f"{cond}|{draw}|{sid}|{n}"] = np.stack(
                    [t, az[i], el[i], dist[i]]).astype(np.float32)
            info = rsr_of(as_scene(names, t, az, el, dist, row["prompt"]), rels)
            report.append({"scene": sid, "cond": cond, "draw": draw,
                           "plan_s": round(secs, 2), **info})
            if draw == 0:
                line.append(f"{cond} {info['mean_rsr']}")
        print("  " + "  ".join(line), flush=True)

    np.savez_compressed(WORK / "demo_trajs.npz", **arrays)
    (WORK / "demo_trajs_report.json").write_text(json.dumps(
        {"seed_rule": "2000 + crc32(scene_id) % 100000 + draw, "
                      "no guidance, no selection; draw 0 is rendered to audio",
         "ddim_steps": STEPS, "m_draws": args.m,
         "checkpoints": {k: v.name for k, v in CKPT.items()},
         "llm": None if llm_cfg is None else llm_cfg.model,
         "rows": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {len(arrays)} trajectories -> {WORK / 'demo_trajs.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
