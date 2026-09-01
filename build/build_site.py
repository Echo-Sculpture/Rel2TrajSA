"""Demo-site step D: emit the published page.

Writes docs/data/{scene}.json (trajectories, per-relation RSR and geometry
stats for every condition and draw) and docs/index.html, by filling the
abstract into build/template.html -- the page copy lives in that file, so
it can be rewritten without touching this script. The viewer's playhead is
driven by the audio clock, so what you see is what you hear.

The viewer keeps the playhead when you switch condition, which is what makes
the page an A/B test rather than five separate players: the same moment of the
same scene, same source clips, same renderer, only the trajectory changes.

Usage: python -B build/build_site.py
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from _common import DATA, HERE, OUTPUTS, RESEARCH, SITE, WORK

from naturalness import trajectory_naturalness
from relations import Relation, _scope_mask, satisfied
from scene import Scene, SourceEvent, Trajectory

KEEP = 96            # frames per trajectory in the page payload
COLORS = ["#5b9cff", "#39c07c", "#ffab3f", "#e05c5c", "#b07cf0"]
COND_LABEL = {
    "GT": ("Ground truth", "dataset trajectory"),
    "REL2TRAJ": ("Rel2Traj", "relation graph + anchors, equivariant "
                             "keyframe diffusion"),
    # Named for what it IS, not for whose paper it evokes: this is our own
    # model with cross-source information flow removed and anchor tokens
    # withheld -- a single-variable control, not a reimplementation of anyone.
    "INDEP": ("Independent per-source", "same model, cross-source information "
                                        "flow removed, anchors withheld"),
    "RULE": ("Rule solver", "constraint optimisation over the declared "
                            "relations"),
    "LLM": ("LLM-direct", "qwen3:8b emits keyframes from the prompt text"),
}
COND_ORDER = ["GT", "REL2TRAJ", "INDEP", "RULE", "LLM"]


def abstract_html() -> str:
    """The abstract shown on the page is the frozen one from the paper draft,
    verbatim -- the page must never drift from the submission. LaTeX quoting
    (``...'') becomes typographic quotes."""
    src = (RESEARCH / "paper" / "abstract_v1.md").read_text(encoding="utf-8")
    body = [p.strip() for p in src.split("\n\n")
            if p.strip() and not p.startswith("#") and not p.startswith("---")]
    text = body[0].replace("``", "&ldquo;").replace("''", "&rdquo;")
    return " ".join(text.split())


SECTORS = ["at_front", "at_front_left", "at_left", "at_back_left", "at_back",
           "at_back_right", "at_right", "at_front_right"]
HELDOUT_DOC = {
    "H1": ("follow &times; right-to-left motion",
           "training keeps follow &times; left-to-right"),
    "H2": ("opposite direction with the right-to-left source leading",
           "training keeps the mirrored lead"),
    "H3": ("behind-pair whose leader sits in a rear sector",
           "training keeps frontal and side leaders"),
    "H4": ("approach starting from the back-right sector",
           "training keeps every other approach sector"),
    "H6": ("half circle swept counter-clockwise",
           "training keeps clockwise sweeps"),
}


def vocabulary() -> dict:
    """The grammar as the dataset actually instantiates it: which relation
    types and motion primitives occur, and how many scenes each held-out
    combination cell contributes to the compositional test split."""
    listener, inter, motion, cells = {}, {}, {}, {}
    for split in ("train", "val_iid", "test_comp"):
        f = DATA / f"scenes_{split}.jsonl"
        if not f.exists():
            continue
        for row in (json.loads(l) for l in open(f, encoding="utf-8")):
            for s in row["sources"]:
                m = s.get("motion", "static")
                motion[m] = motion.get(m, 0) + 1
            for r in row["relations"]:
                d = inter if r["reference"] != "listener" else listener
                d[r["type"]] = d.get(r["type"], 0) + 1
            for c in row.get("heldout_cells", []):
                cells[c] = cells.get(c, 0) + 1
    return {
        "sectors": [k for k in SECTORS if k in listener],
        "listener_motion": {k: v for k, v in sorted(listener.items())
                            if k not in SECTORS},
        "inter": dict(sorted(inter.items(), key=lambda kv: -kv[1])),
        "motion": dict(sorted(motion.items(), key=lambda kv: -kv[1])),
        "heldout": [{"id": c.split(":")[0], "n": n,
                     "held": HELDOUT_DOC.get(c.split(":")[0], ("", ""))[0],
                     "kept": HELDOUT_DOC.get(c.split(":")[0], ("", ""))[1]}
                    for c, n in sorted(cells.items())],
    }


def parser_info() -> dict:
    """The LLM in the pipeline: a fine-tuned parser that turns one sentence
    into the relation graph and anchors. Numbers are read from the evaluation
    JSONs so the page cannot quote a stale figure."""
    def load(name):
        p = OUTPUTS / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    ft, few, para = (load("parser_v2_ablate60_lora.json"),
                     load("parser_v2_ablate60.json"),
                     load("e7b_paraphrase_eval.json"))
    return {
        "base": "Qwen3-4B",
        "adapter": "QLoRA, 4-bit NF4 base, LoRA r=16 alpha=32 on the "
                   "attention and MLP projections",
        "train": "4891 sentence / graph pairs from the training split",
        "rows": [
            ("fine-tuned parser", ft.get("motion_acc"), ft.get("relation_F1"),
             ft.get("e2e_rsr_parsed_bestofM")),
            ("same model, few-shot prompting only", few.get("motion_acc"),
             few.get("relation_F1"), few.get("e2e_rsr_parsed_bestofM")),
            ("fine-tuned parser, paraphrased sentences",
             para.get("motion_acc"), para.get("relation_F1"),
             para.get("e2e_rsr_parsed_bestofM")),
        ],
        "oracle": ft.get("e2e_rsr_oracle_bestofM"),
        "n_scenes": ft.get("n_scenes"),
    }


def spherical_to_xyz(az, el, dist):
    azr, elr = np.deg2rad(az), np.deg2rad(el)
    return (-dist * np.cos(elr) * np.sin(azr),
            dist * np.cos(elr) * np.cos(azr),
            dist * np.sin(elr))


def frame_state(names, arrs, target, sel) -> list[int]:
    """Per-frame state of the target relation: 1 satisfied, 0 violated,
    -1 out of scope. This is what RSR aggregates -- showing it on the timeline
    turns the metric into something you can watch break."""
    sc = Scene(prompt="")
    for n in names:
        t, az, el, dist = arrs[n]
        sc.add(SourceEvent(n, Trajectory(t, az, el, dist)))
    rel = Relation(target["type"], target["subject"], target["reference"], None)
    ok = satisfied(sc, rel)
    scope = _scope_mask(sc, rel)
    state = np.where(scope, ok.astype(int), -1)
    return [int(v) for v in state[sel]]


def geometry_stats(arrs) -> dict:
    """What RSR cannot see. A planner that optimises the declared relations
    can satisfy them while walking a source through the listener's head, so
    the page reports smoothness and head clearance next to the relation
    score."""
    jerk, head, dmin = [], [], []
    for t, az, el, dist in arrs.values():
        n = trajectory_naturalness(Trajectory(t, az, el, dist))
        jerk.append(n["mean_jerk"])
        head.append(n["head_pass_rate"])
        dmin.append(float(np.min(dist)))
    return {"jerk": round(float(np.mean(jerk)), 1),
            "head": round(float(np.mean(head)), 3),
            "min_dist": round(float(np.min(dmin)), 2)}


def main() -> int:
    manifest = json.loads((WORK / "scene_manifest.json").read_text("utf-8"))
    report = json.loads((WORK / "demo_trajs_report.json").read_text("utf-8"))
    render = json.loads((WORK / "render_manifest.json").read_text("utf-8"))
    trajs = np.load(WORK / "demo_trajs.npz")

    rsr = {(r["scene"], r["cond"], r["draw"]): r for r in report["rows"]}
    audio = {i["scene"]: i["files"] for i in render["items"]}

    rows = {}
    for split in ("val_iid", "test_comp"):
        for r in (json.loads(l) for l in
                  open(DATA / f"scenes_{split}.jsonl", encoding="utf-8")):
            rows[r["id"]] = r

    (SITE / "data").mkdir(parents=True, exist_ok=True)
    index = []
    for c in manifest["scenes"]:
        sid = c["scene"]
        row = rows[sid]
        names = [s["name"] for s in row["sources"]]
        target = c["inter"][0]
        tkey = f"{target['type']}|{target['subject']}|{target['reference']}"

        t_full = trajs[f"GT|0|{sid}|{names[0]}"][0]
        step = max(1, len(t_full) // KEEP)
        sel = np.arange(0, len(t_full), step)
        payload = {
            "id": sid, "split": row.get("split", c["split"]),
            "prompt": row["prompt"], "duration": float(row["duration"]),
            "t": np.round(t_full[sel], 2).tolist(),
            "heldout": c.get("heldout_cells", []),
            "target": target,
            "target_idx": [names.index(target["subject"]),
                           names.index(target["reference"])],
            "relations": [{"type": r["type"], "subject": r["subject"],
                           "reference": r["reference"],
                           "inter": r["reference"] != "listener"}
                          for r in row["relations"]],
            "sources": [{"name": s["name"], "motion": s.get("motion", "static"),
                         "color": COLORS[i % len(COLORS)]}
                        for i, s in enumerate(row["sources"])],
            "conds": {},
        }
        for cond in COND_ORDER:
            draws = []
            for j in range(20):
                key = f"{cond}|{j}|{sid}|{names[0]}"
                if key not in trajs:
                    break
                src, arrs = [], {}
                for n in names:
                    arr = trajs[f"{cond}|{j}|{sid}|{n}"]
                    arrs[n] = arr
                    _, az, el, dist = arr
                    x, y, z = spherical_to_xyz(az[sel], el[sel], dist[sel])
                    src.append({"x": np.round(x, 2).tolist(),
                                "y": np.round(y, 2).tolist(),
                                "z": np.round(z, 2).tolist()})
                rec = rsr.get((sid, cond, j), {})
                draws.append({
                    "src": src,
                    "state": frame_state(names, arrs, target, sel),
                    "geom": geometry_stats(arrs),
                    "mean": rec.get("mean_rsr"),
                    "target": (rec.get("per_relation") or {}).get(tkey),
                    "per_relation": rec.get("per_relation") or {},
                    "plan_s": rec.get("plan_s"),
                })
            if not draws:
                continue
            label, note = COND_LABEL[cond]
            payload["conds"][cond] = {
                "label": label, "note": note, "draws": draws,
                "audio": audio.get(sid, {}).get(cond),
            }
        (SITE / "data" / f"{sid}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        index.append({
            "id": sid, "split": c["split"], "prompt": row["prompt"],
            "target": target, "motions": c["motions"],
            "n_sources": c["n_sources"], "heldout": bool(c["heldout_cells"]),
            "rsr": {cond: (rsr.get((sid, cond, 0), {}) or {}).get("mean_rsr")
                    for cond in COND_ORDER},
            "target_rsr": {
                cond: ((rsr.get((sid, cond, 0), {}) or {}).get("per_relation")
                       or {}).get(tkey) for cond in COND_ORDER},
        })

    site_meta = {
        "scenes": index,
        "conds": [{"key": k, "label": COND_LABEL[k][0],
                   "note": COND_LABEL[k][1]} for k in COND_ORDER],
        "protocol": {
            "sampling": report["seed_rule"],
            "ddim_steps": report["ddim_steps"],
            "m_draws": report.get("m_draws"),
            "llm": report.get("llm"),
            "hrtf": render["hrtf_set"],
            "renderer": render["renderer"],
            "loudness": render["loudness"],
            "material": render["material"],
            "level": render["level"],
            "selection": manifest["rule"],
            "relation_coverage": manifest["relation_coverage"],
            "motion_coverage": manifest["motion_coverage"],
        },
        "vocab": vocabulary(),
        "parser": parser_info(),
    }
    meta_json = json.dumps(site_meta, ensure_ascii=False, indent=1)
    (SITE / "data" / "index.json").write_text(meta_json, encoding="utf-8")

    html = (HERE / "template.html").read_text(encoding="utf-8")
    assert "{{ABSTRACT}}" in html, "template lost its abstract placeholder"
    # data files are fetched with ?v=<stamp>; without it a browser (or the
    # Pages CDN) happily serves yesterday's JSON against today's page
    stamp = hashlib.md5(meta_json.encode("utf-8")).hexdigest()[:10]
    (SITE / "index.html").write_text(
        html.replace("{{ABSTRACT}}", abstract_html()).replace("{{BUILD}}", stamp),
        encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    payload_kb = sum(f.stat().st_size for f in (SITE / "data").iterdir()) / 1024
    print(f"wrote {len(index)} scenes, data payload {payload_kb:.0f} KB")
    print(f"page -> {SITE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
