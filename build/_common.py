"""Shared paths for the Rel2TrajSA demo-site build scripts.

This project is deliberately SEPARATE from the research repo: nothing here
writes into it. The research repo is imported read-only (model code, renderer,
grammar, frozen dataset) via TTSA on sys.path, and every artefact this build
produces lands under Rel2TrajSA/work (intermediates) or Rel2TrajSA/site
(the published page).

Override the research-repo location with the RESEARCH_REPO env var.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # Rel2TrajSA/
WORK = ROOT / "work"                    # intermediates (not published)
SITE = ROOT / "docs"                    # the GitHub Pages site (served from /docs)
RESEARCH = Path(os.environ.get(
    "RESEARCH_REPO", ROOT.parent / "TTSA")).resolve()
DATA = RESEARCH / "data" / "train_v1"
OUTPUTS = RESEARCH / "outputs"

WORK.mkdir(parents=True, exist_ok=True)
SITE.mkdir(parents=True, exist_ok=True)

if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
# the research code resolves data/ and outputs/ relative to its own cwd in a
# few places; run its imports from there
os.chdir(RESEARCH)
