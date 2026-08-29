"""Demo-site step C: render every scene x condition to binaural audio.

Rendering protocol (identical for all five conditions -- the whole point of
the page is that ONLY the trajectory differs):

- material: the same audited generated clip per source in every condition,
  picked by the E2 rule (per-class best screened clip);
- loudness: gated LUFS -28 (BS.1770) then the ESC-50 class power prior;
- renderer: renderer_v2 final (per-sample fractional delay + Doppler,
  minimum-phase interpolated HRTF, full shoebox reflections, rear shading,
  overhead band, near-field ILD);
- HRTF: one set for the page (Neumann KU100 dummy head by default);
- level: ONE gain per scene, shared by all conditions (peak-normalising each
  condition separately would erase the level differences that distance
  errors actually produce -- a comparison page must not launder them).

Only draw 0 of the diffusion conditions is rendered: it is the first draw
under the pre-registered seed, so no clip on the page is a hand-picked sample.

Output: docs/audio/{scene}_{cond}.flac  (lossless: lossy stereo coding can
        smear the ITD/ILD cues this page exists to demonstrate)
        work/render_manifest.json
Usage:  python -B build/render_audio.py [--hrtf ku100] [--scenes t04110 ...]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import soundfile as sf

from _common import DATA, SITE, WORK

import esc50
from audio import FS
from e2_render_stimuli import build_gen_picks, gen_mono
from renderer_v2 import render_source_v2_auto
from scene import Trajectory

AUDIO = SITE / "audio"
CONDS = ("GT", "REL2TRAJ", "INDEP", "RULE", "LLM")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hrtf", default="ku100", choices=["ku100", "kemar",
                                                        "hutubs"])
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--draw", type=int, default=0)
    args = ap.parse_args()

    AUDIO.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((WORK / "scene_manifest.json").read_text("utf-8"))
    trajs = np.load(WORK / "demo_trajs.npz")
    scenes = [c for c in manifest["scenes"]
              if args.scenes is None or c["scene"] in args.scenes]

    rows = {}
    for split in ("val_iid", "test_comp"):
        for r in (json.loads(l) for l in
                  open(DATA / f"scenes_{split}.jsonl", encoding="utf-8")):
            rows[r["id"]] = r

    esc50.CLASS_GAIN_ENABLED = True      # realism prior on, one protocol
    picks = build_gen_picks()
    out_items, n_files, total_bytes = [], 0, 0

    for c in scenes:
        sid = c["scene"]
        row = rows[sid]
        names = [s["name"] for s in row["sources"]]
        dur = float(row["duration"])
        n_samp = int(dur * FS)

        clips = {n: gen_mono(picks[n.lower().rstrip("0123456789")], dur,
                             n.lower().rstrip("0123456789")) for n in names}

        mixes = {}
        for cond in CONDS:
            draw = args.draw if cond in ("REL2TRAJ", "INDEP") else 0
            key0 = f"{cond}|{draw}|{sid}|{names[0]}"
            if key0 not in trajs:
                continue                  # condition missing (e.g. LLM failed)
            mix = np.zeros((n_samp, 2), dtype=np.float64)
            for n in names:
                t, az, el, dist = trajs[f"{cond}|{draw}|{sid}|{n}"]
                mix += render_source_v2_auto(clips[n],
                                             Trajectory(t, az, el, dist), n,
                                             FS, hrtf_set=args.hrtf)[:n_samp]
            mixes[cond] = mix

        # one gain for the whole scene: conditions stay comparable in level
        peak = max(float(np.max(np.abs(m))) for m in mixes.values()) + 1e-9
        files = {}
        for cond, mix in mixes.items():
            fn = f"{sid}_{cond}.flac"
            sf.write(AUDIO / fn, (0.89 * mix / peak).astype(np.float32), FS,
                     format="FLAC", subtype="PCM_16")
            files[cond] = f"audio/{fn}"
            n_files += 1
            total_bytes += (AUDIO / fn).stat().st_size
        out_items.append({"scene": sid, "duration": dur, "files": files})
        print(f"  {sid}: {len(files)} conditions, peak {peak:.3f}", flush=True)

    (WORK / "render_manifest.json").write_text(json.dumps(
        {"hrtf_set": args.hrtf, "fs": FS, "format": "FLAC PCM_16",
         "material": "GenA generated clips (per-class audited picks), "
                     "identical per source across conditions",
         "loudness": "gated LUFS -28 (BS.1770) x ESC-50 class power prior",
         "renderer": "renderer_v2 final (Doppler, min-phase interp HRTF, "
                     "shoebox reflections, rear shading, overhead band)",
         "level": "one peak gain per scene, shared by all conditions",
         "draw_rendered": args.draw, "items": out_items},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{n_files} files, {total_bytes / 1e6:.1f} MB -> {AUDIO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
