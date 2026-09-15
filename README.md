# Rel2TrajSA — demo page

Audio-visual supplement for **Inter-Source Relation-Aware Trajectory Generation
for Text-To-Spatial-Audio**. The page pairs a drag-orbit 3D trajectory viewer
with binaural audio: the same scene is rendered from five different trajectory
sources (ground truth, Rel2Traj, an independent per-source baseline, a rule
solver, and an LLM-direct baseline) while the source clips, the renderer and
the playhead stay fixed, so the only thing that changes between them is the
geometry.

Every scene on the page is multi-source and carries at least one inter-source
relation; the twelve scenes together cover all seven instantiated inter-source
relation types and every motion primitive in the grammar.

**Live page:** https://echo-sculpture.github.io/Rel2TrajSA/

## Layout

```
docs/            the published page (GitHub Pages serves this directory)
  index.html     viewer
  data/          per-scene trajectories + relation satisfaction rates
  audio/         binaural renders, FLAC
build/           scripts that generate docs/ from the research repo
work/            build intermediates (not published)
```

## Rebuilding

The build reads the research repository read-only; set `RESEARCH_REPO` if it
is not the sibling directory `../TTSA`.

```bash
python -B build/select_scenes.py --n 12        # pick scenes (coverage rule)
python -B build/sample_trajectories.py --m 10  # five conditions, ten draws
python -B build/render_audio.py --hrtf ku100   # binaural renders
python -B build/build_site.py                  # emit docs/
```

Local preview (the page fetches its data, so it needs a server, not `file://`):

```bash
python -m http.server 8765 --directory docs
```
