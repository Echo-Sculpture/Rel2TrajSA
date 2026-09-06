"""Demo-site step E: results per held-out combination cell, all five systems.

The paper reports per-type means over the whole compositional split. The page
goes one level finer: for each pre-registered held-out cell (H1 follow x
right-to-left, H2 opposite with right-to-left lead, H3 behind with a rear-
sector leader, H4 approach from back-right, H6 counter-clockwise half circle)
it reports, per system, the RSR of the cell's defining relation and the scene
mean, over exactly the test_comp scenes that carry that cell.

Protocol matches the main table: per-type mean, zero selection; the diffusion
systems draw M samples per scene with the SAME seeds eval_diffusion.py uses
(2000 + j), so their per-type numbers over the whole split reproduce the
paper's E6-2 column -- that reproduction is written out as a cross-check.

Systems:
  GT        frozen dataset trajectories
  REL2TRAJ  outputs/d3_csteer.pt, full conditioning
  INDEP     outputs/e1b_indep_noanchor.pt, own regime (edges stripped,
            anchors -> unk)
  RULE      plan_rule_v2, motions + relations, maxiter 200 (as in E6)
  LLM       outputs/llm_direct_v11/test_comp.jsonl (the E6 run, per scene;
            a scene that failed all retries is scored 0, the strict protocol)

Per-scene records are cached under work/cells/ so a re-run is free.

Output: work/cells_results.json
Usage:  python -B build/eval_cells.py [--device cuda] [--m 10] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import torch

from _common import DATA, OUTPUTS, WORK

from dataset_torch import (load_split, strip_anchor_tokens,
                           strip_cross_source_edges)
from diffusion_decoder import DiffusionSchedule, ddim_sample, kf_to_dense
from eval_strict import load_model
from generator import build_vocab
from relations import scene_rsr
from run_rule_splits import _relations_of
from scene import Scene, SourceEvent, Trajectory
from train_diffusion import _to_device

CACHE = WORK / "cells"
CKPT = {"REL2TRAJ": OUTPUTS / "d3_csteer.pt",
        "INDEP": OUTPUTS / "e1b_indep_noanchor.pt"}
CELL_RELATION = {"H1": "follow", "H2": "opposite_direction", "H3": "behind",
                 "H4": "approach", "H6": "counterclockwise"}
STEPS = 20


def rows_test_comp() -> list[dict]:
    return [json.loads(l) for l in
            open(DATA / "scenes_test_comp.jsonl", encoding="utf-8")]


def score(scene: Scene, rels) -> dict:
    """One record: scene mean + per relation instance (type, inter flag, rsr)."""
    r = scene_rsr(scene, rels)
    per = [{"type": pr["type"], "inter": pr["reference"] != "listener",
            "rsr": None if pr["rsr"] != pr["rsr"] else float(pr["rsr"])}
           for pr in r["per_relation"]]
    m = r["mean_rsr"]
    return {"mean": None if m != m else float(m), "per": per}


def scene_from(names, t, az, el, dist) -> Scene:
    sc = Scene(prompt="")
    for i, n in enumerate(names):
        sc.add(SourceEvent(n, Trajectory(t, az[i], el[i], dist[i])))
    return sc


# ------------------------------------------------------------------ systems
def run_gt(rows) -> dict:
    npz = np.load(DATA / "trajs_test_comp.npz")
    out = {}
    for row in rows:
        names = [s["name"] for s in row["sources"]]
        arr = np.stack([npz[f"{row['id']}__{n}"] for n in names])
        sc = scene_from(names, arr[0, 0], arr[:, 1], arr[:, 2], arr[:, 3])
        out[row["id"]] = [score(sc, _relations_of(row))]
    return out


def _rule_one(args):
    row, maxiter = args
    from rule_planner import plan_rule_v2
    rels = _relations_of(row)
    sc = plan_rule_v2(row["prompt"], row["sources"], rels,
                      float(row["duration"]), maxiter=maxiter)
    return row["id"], [score(sc, rels)]


def run_rule(rows, workers: int, maxiter: int = 200) -> dict:
    with Pool(workers) as pool:
        res = pool.map(_rule_one, [(r, maxiter) for r in rows], chunksize=4)
    return dict(res)


def run_llm(rows) -> dict:
    """Per-scene records of the E6 LLM-direct run. Strict protocol: a scene
    that failed every retry counts as 0 on each of its relations."""
    recs = {}
    for line in open(OUTPUTS / "llm_direct_v11" / "test_comp.jsonl",
                     encoding="utf-8"):
        rec = json.loads(line)
        recs[rec["id"]] = rec
    out = {}
    for row in rows:
        rec = recs.get(row["id"])
        rels = row["relations"]
        if rec is None or not rec.get("ok"):
            out[row["id"]] = [{"mean": 0.0, "per": [
                {"type": r["type"], "inter": r["reference"] != "listener",
                 "rsr": 0.0} for r in rels]}]
            continue
        per = []
        for pr, r in zip(rec["per_relation"], rels):
            v = pr["rsr"]
            per.append({"type": r["type"], "inter": r["reference"] != "listener",
                        "rsr": None if v != v else float(v)})
        m = rec["mean_rsr"]
        out[row["id"]] = [{"mean": None if m != m else float(m), "per": per}]
    return out


def run_model(cond, rows, device, m: int) -> dict:
    vocab = build_vocab()
    enc, den, _, independent, no_anchors = load_model(str(CKPT[cond]), device,
                                                      vocab)
    sched = DiffusionSchedule(device=device.type)
    recs = load_split(DATA, "test_comp", vocab)
    if independent:
        strip_cross_source_edges(recs)
    if no_anchors:
        strip_anchor_tokens(recs, vocab["<unk>"])
    print(f"[{cond}] independent={independent} no_anchors={no_anchors} "
          f"{len(recs)} scenes x {m} draws", flush=True)
    out, t0 = {}, time.time()
    for k, rec in enumerate(recs):
        st = _to_device(rec.tensor, device)
        with torch.no_grad():
            z = enc.encode(st)
        T = st.gt_az.shape[1]
        draws = []
        for j in range(m):
            gen = torch.Generator(device=device.type).manual_seed(2000 + j)
            kf = ddim_sample(den, sched, z, n_sources=len(rec.tensor.names),
                             steps=STEPS, generator=gen)
            az, el, dist = (a.cpu().numpy() for a in kf_to_dense(kf, T))
            sc = scene_from(rec.tensor.names, rec.t, az, el, dist)
            draws.append(score(sc, rec.relations))
        out[rec.row["id"]] = draws
        if (k + 1) % 100 == 0:
            el_s = time.time() - t0
            print(f"  [{cond}] {k + 1}/{len(recs)} "
                  f"({el_s / (k + 1):.2f} s/scene)", flush=True)
    return out


# ---------------------------------------------------------------- aggregate
def per_type_mean(records: dict, ids, rel_type: str, inter_only: bool):
    vals = []
    for sid in ids:
        for draw in records[sid]:
            for pr in draw["per"]:
                if pr["type"] == rel_type and pr["rsr"] is not None and \
                        (pr["inter"] or not inter_only):
                    vals.append(pr["rsr"])
    return (round(float(np.mean(vals)), 3), len(vals)) if vals else (None, 0)


def scene_mean(records: dict, ids):
    vals = [d["mean"] for sid in ids for d in records[sid]
            if d["mean"] is not None]
    return round(float(np.mean(vals)), 3) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--systems", nargs="*",
                    default=["GT", "REL2TRAJ", "INDEP", "RULE", "LLM"])
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    rows = rows_test_comp()
    device = torch.device(args.device)

    records: dict[str, dict] = {}
    for sysname in args.systems:
        f = CACHE / f"{sysname}.json"
        if f.exists():
            records[sysname] = json.loads(f.read_text(encoding="utf-8"))
            print(f"[{sysname}] cached ({len(records[sysname])} scenes)")
            continue
        t0 = time.time()
        if sysname == "GT":
            rec = run_gt(rows)
        elif sysname == "RULE":
            rec = run_rule(rows, args.workers)
        elif sysname == "LLM":
            rec = run_llm(rows)
        else:
            rec = run_model(sysname, rows, device, args.m)
        f.write_text(json.dumps(rec), encoding="utf-8")
        records[sysname] = rec
        print(f"[{sysname}] done in {time.time() - t0:.0f} s", flush=True)

    # cells: scene ids per held-out cell
    cells: dict[str, list] = {}
    for row in rows:
        for c in row.get("heldout_cells", []):
            cells.setdefault(c.split(":")[0], []).append(row["id"])
    all_ids = [r["id"] for r in rows]

    result = {"m_draws": args.m, "seeds": "2000 + j (eval_diffusion protocol)",
              "n_test_comp": len(rows), "cells": [], "per_type_comp": {}}
    for cell in sorted(cells):
        ids = cells[cell]
        rel = CELL_RELATION[cell]
        inter = rel in ("follow", "opposite_direction", "behind")
        entry = {"cell": cell, "relation": rel, "inter": inter,
                 "n_scenes": len(ids), "systems": {}}
        for sysname, rec in records.items():
            v, n = per_type_mean(rec, ids, rel, inter_only=inter)
            entry["systems"][sysname] = {"target": v, "n_instances": n,
                                         "scene_mean": scene_mean(rec, ids)}
        result["cells"].append(entry)
        print(f"  {cell} {rel:20} n={len(ids):>4}  " + "  ".join(
            f"{s} {entry['systems'][s]['target']}" for s in records))

    # cross-check against the paper's E6-2 comp column (inter-source types)
    for rel in ("follow", "same_direction", "opposite_direction", "behind",
                "closer_than", "left_of", "right_of"):
        result["per_type_comp"][rel] = {
            s: per_type_mean(rec, all_ids, rel, inter_only=True)[0]
            for s, rec in records.items()}
    (WORK / "cells_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {WORK / 'cells_results.json'}")
    print("E6-2 comp cross-check:", json.dumps(result["per_type_comp"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
