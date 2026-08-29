"""Demo-site scene selection (Rel2TrajSA project page).

Picks the scenes shown on the Rel2TrajSA demo site. Selection rules follow the
user's brief (2026-08-29): every scene must be MULTI-SOURCE, must carry at
least one inter-source relation, and the set as a whole must span the motion
vocabulary -- a viewer should be able to hear every relation family and see
most motion primitives.

Filters:
- 2-3 sources, at least one inter-source relation (reference != listener);
- the two sources of the target relation come from DIFFERENT timbre
  categories (ESC-50 category via LABEL_TO_CATEGORY) so a listener can tell
  which source is which in the mix -- same rule as the E2 stimulus screen;
- clean ground truth (row worst_rsr >= 0.95) so the GT column is a fair
  reference, not a borderline case;
- duration 8-14 s (the dataset range).

Greedy coverage: relation families first (all 7 instantiated inter-source
types must appear), then motion primitives, with test_comp scenes preferred
for the held-out combination cells (follow / behind / opposite_direction) --
those are where the baselines break and the demo has something to show.

Deterministic: candidates sorted by id, no randomness.

Output: work/scene_manifest.json
Usage:  python -B build/select_scenes.py [--n 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import DATA, WORK
from esc50 import LABEL_TO_CATEGORY

OUT = WORK

INTER = {"follow", "opposite_direction", "same_direction", "behind",
         "closer_than", "left_of", "right_of"}
# families whose held-out cells are the paper's compositional showcase
COMP_PREFERRED = {"follow", "behind", "opposite_direction"}


def category(name: str) -> str | None:
    base = name.lower().rstrip("0123456789")
    return LABEL_TO_CATEGORY.get(base, base)


def candidates(split: str) -> list[dict]:
    rows = [json.loads(l) for l in
            open(DATA / f"scenes_{split}.jsonl", encoding="utf-8")]
    out = []
    for row in rows:
        n_src = len(row["sources"])
        if not 2 <= n_src <= 3:
            continue
        if row.get("worst_rsr", 0.0) < 0.95:
            continue
        if not 8.0 <= float(row["duration"]) <= 14.0:
            continue
        inter = [r for r in row["relations"] if r["reference"] != "listener"
                 and r["type"] in INTER]
        if not inter:
            continue
        motions = {s.get("motion", "static") for s in row["sources"]}
        # timbre-distinguishable pair for every inter-source relation shown
        keep = []
        for r in inter:
            ca, cb = category(r["subject"]), category(r["reference"])
            if ca is not None and cb is not None and ca != cb:
                keep.append(r)
        if not keep:
            continue
        out.append({
            "scene": row["id"], "split": split, "prompt": row["prompt"],
            "duration": float(row["duration"]), "n_sources": n_src,
            "sources": [{"name": s["name"], "motion": s.get("motion", "static")}
                        for s in row["sources"]],
            "inter": [{"type": r["type"], "subject": r["subject"],
                       "reference": r["reference"]} for r in keep],
            "motions": sorted(motions),
            "heldout_cells": row.get("heldout_cells", []),
        })
    return sorted(out, key=lambda c: c["scene"])


def score(c: dict, rel_have: dict, mot_have: dict) -> float:
    """Coverage gain: unseen relation families dominate, then unseen motions."""
    s = 0.0
    for r in c["inter"]:
        t = r["type"]
        s += 10.0 if rel_have.get(t, 0) == 0 else 1.0 / (1 + rel_have[t])
    for m in c["motions"]:
        s += 3.0 if mot_have.get(m, 0) == 0 else 0.2 / (1 + mot_have[m])
    if c["split"] == "test_comp" and any(r["type"] in COMP_PREFERRED
                                         for r in c["inter"]):
        s += 2.0                      # held-out combination showcase
    if c["heldout_cells"]:
        s += 1.0
    s += 0.5 * (c["n_sources"] - 2)   # richer scenes read better in 3D
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    pool = candidates("test_comp") + candidates("val_iid")
    print(f"candidates: {len(pool)} "
          f"(test_comp {sum(c['split'] == 'test_comp' for c in pool)}, "
          f"val_iid {sum(c['split'] == 'val_iid' for c in pool)})")

    rel_have: dict[str, int] = {}
    mot_have: dict[str, int] = {}
    picked: list[dict] = []
    while len(picked) < args.n and pool:
        best = max(pool, key=lambda c: (score(c, rel_have, mot_have),
                                        -ord(c["scene"][1]), c["scene"]))
        pool.remove(best)
        picked.append(best)
        for r in best["inter"]:
            rel_have[r["type"]] = rel_have.get(r["type"], 0) + 1
        for m in best["motions"]:
            mot_have[m] = mot_have.get(m, 0) + 1

    picked.sort(key=lambda c: (c["split"] != "test_comp", c["scene"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scene_manifest.json").write_text(json.dumps(
        {"rule": "multi-source + >=1 inter-source relation + distinct timbres "
                 "+ GT worst_rsr>=0.95; greedy relation/motion coverage",
         "relation_coverage": dict(sorted(rel_have.items())),
         "motion_coverage": dict(sorted(mot_have.items())),
         "scenes": picked}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\npicked {len(picked)} scenes")
    print(f"{'scene':>8} {'split':>10} {'n':>2}  {'relations':<34} motions")
    for c in picked:
        rels = ",".join(f"{r['type']}({r['subject']}>{r['reference']})"
                        for r in c["inter"])
        print(f"{c['scene']:>8} {c['split']:>10} {c['n_sources']:>2}  "
              f"{rels[:34]:<34} {','.join(c['motions'])}")
    print(f"\nrelation coverage: {dict(sorted(rel_have.items()))}")
    print(f"motion coverage:   {dict(sorted(mot_have.items()))}")
    missing = INTER - set(rel_have)
    print(f"MISSING relations: {sorted(missing)}" if missing
          else "all 7 inter-source relation types covered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
