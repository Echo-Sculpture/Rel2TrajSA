"""Demo-site step D: emit the published page.

Writes docs/data/{scene}.json (trajectories + per-relation RSR for every
condition and draw) and docs/index.html -- a drag-orbit 3D viewer whose
playhead is driven by the audio clock, so what you see is what you hear.

The viewer keeps the playhead when you switch condition, which is what makes
the page an A/B test rather than five separate players: the same moment of the
same scene, same source clips, same renderer, only the trajectory changes.

Usage: python -B build/build_site.py
"""

from __future__ import annotations

import json

import numpy as np

from _common import DATA, SITE, WORK

from relations import Relation, _scope_mask, satisfied
from scene import Scene, SourceEvent, Trajectory

KEEP = 96            # frames per trajectory in the page payload
COLORS = ["#5b9cff", "#39c07c", "#ffab3f", "#e05c5c", "#b07cf0"]
COND_LABEL = {
    "GT": ("Ground truth", "dataset trajectory"),
    "REL2TRAJ": ("Rel2Traj", "relation graph + anchors, equivariant "
                             "keyframe diffusion"),
    "INDEP": ("Independent per-source", "Text2Move-style: no cross-source "
                                        "flow, no anchors"),
    "RULE": ("Rule solver", "constraint optimisation over the declared "
                            "relations"),
    "LLM": ("LLM-direct", "qwen3:8b emits keyframes from the prompt text"),
}
COND_ORDER = ["GT", "REL2TRAJ", "INDEP", "RULE", "LLM"]


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
    }
    (SITE / "data" / "index.json").write_text(
        json.dumps(site_meta, ensure_ascii=False, indent=1), encoding="utf-8")

    (SITE / "index.html").write_text(TEMPLATE, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    payload_kb = sum(f.stat().st_size for f in (SITE / "data").iterdir()) / 1024
    print(f"wrote {len(index)} scenes, data payload {payload_kb:.0f} KB")
    print(f"page -> {SITE}")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rel2Traj - inter-source relation-aware trajectory generation</title>
<style>
  :root { --bg:#0d1016; --panel:#151a23; --card:#1b212c; --fg:#e9ecf1;
          --mut:#98a1b0; --line:#28303d; --acc:#5b9cff; --good:#39c07c;
          --warn:#ffab3f; --bad:#e05c5c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); line-height:1.55;
         font-family:system-ui,"Segoe UI","Helvetica Neue",sans-serif; }
  a { color:var(--acc); }
  header { padding:28px 20px 18px; border-bottom:1px solid var(--line);
           background:linear-gradient(180deg,#141a26,#0d1016); }
  .wrap { max-width:1180px; margin:0 auto; }
  h1 { margin:0 0 6px; font-size:1.5rem; letter-spacing:.2px; }
  h2 { font-size:1.1rem; margin:34px 0 10px; }
  .sub { color:var(--mut); font-size:.95rem; max-width:80ch; }
  .tag { display:inline-block; padding:2px 9px; border-radius:99px;
         border:1px solid var(--line); background:#1d2431; color:var(--mut);
         font-size:.78rem; margin-right:6px; }
  .tag.inter { border-color:#3b5c8f; color:#a8c8ff; background:#16233a; }
  .tag.hold { border-color:#7a5a20; color:#ffc978; background:#2b2210; }
  .note { color:var(--mut); font-size:.86rem; }
  main { padding:0 20px 60px; }
  .scenebar { display:flex; gap:8px; overflow-x:auto; padding:14px 0 4px; }
  .sc { flex:0 0 auto; padding:8px 12px; border-radius:10px; cursor:pointer;
        border:1px solid var(--line); background:var(--card); font-size:.84rem;
        white-space:nowrap; }
  .sc.on { border-color:var(--acc); background:#16243c; }
  .sc small { display:block; color:var(--mut); font-size:.74rem; }
  .stage { display:grid; grid-template-columns:minmax(0,1fr) 330px; gap:16px;
           margin-top:10px; }
  @media (max-width:900px){ .stage { grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:14px; padding:14px; }
  #cv { display:block; width:100%; height:460px; cursor:grab;
        touch-action:none; border-radius:10px; background:#0a0d13; }
  #cv.drag { cursor:grabbing; }
  .transport { display:flex; align-items:center; gap:10px; margin-top:10px; }
  button, select { font:inherit; background:#212936; color:var(--fg);
        border:1px solid var(--line); border-radius:9px; padding:7px 13px;
        cursor:pointer; }
  button:hover { border-color:#3b475c; }
  input[type=range] { flex:1; accent-color:var(--acc); }
  .conds { display:flex; flex-direction:column; gap:8px; }
  .cond { text-align:left; padding:10px 12px; border-radius:11px;
          border:1px solid var(--line); background:var(--card); cursor:pointer; }
  .cond.on { border-color:var(--acc); background:#16243c; }
  .cond .row1 { display:flex; justify-content:space-between; gap:10px;
                align-items:baseline; }
  .cond b { font-size:.95rem; }
  .cond .note { font-size:.78rem; }
  .rsr { font-variant-numeric:tabular-nums; font-weight:600; font-size:.95rem; }
  .rsr.hi { color:var(--good); } .rsr.mid { color:var(--warn); }
  .rsr.lo { color:var(--bad); }
  .legend { display:flex; flex-wrap:wrap; gap:12px; margin:10px 0 0;
            font-size:.85rem; color:var(--mut); }
  .dot { display:inline-block; width:10px; height:10px; border-radius:5px;
         margin-right:6px; vertical-align:middle; }
  table { border-collapse:collapse; width:100%; font-size:.86rem;
          margin-top:8px; }
  th, td { border-bottom:1px solid var(--line); padding:7px 9px;
           text-align:left; }
  th { color:var(--mut); font-weight:600; }
  td.num { font-variant-numeric:tabular-nums; text-align:right; }
  .best { color:var(--good); font-weight:600; }
  details { margin-top:10px; } summary { cursor:pointer; color:var(--mut); }
  code { background:#1b212c; padding:1px 5px; border-radius:5px;
         font-size:.85em; }
</style>
</head>
<body>
<header><div class="wrap">
  <h1>Inter-Source Relation-Aware Trajectory Generation for Text-to-Spatial-Audio</h1>
  <p class="sub">Audio-visual supplement for the submission. Every scene below
  has <b>several moving sources whose relation to each other</b> is what the
  text asks for &mdash; one sound following another, two passing in opposite
  directions, one staying behind or closer than another. Drag the view to
  orbit, press play, and switch between systems: the source clips, the
  renderer and the playhead stay fixed, so the only thing that changes is the
  trajectory.</p>
  <p class="note">Headphones required (binaural rendering).
  The system is named <b>Rel2Traj</b>; ground truth is the dataset trajectory,
  not another model.</p>
</div></header>

<main class="wrap">
  <div class="scenebar" id="scenebar"></div>

  <div id="prompt" class="panel" style="margin-top:6px"></div>

  <div class="stage">
    <div class="panel">
      <canvas id="cv"></canvas>
      <div class="transport">
        <button id="play">&#9654; Play</button>
        <span id="tlabel" class="note" style="min-width:92px">0.0 / 0.0 s</span>
        <input type="range" id="time" min="0" max="1000" value="0">
        <label class="note"><input type="checkbox" id="ghost"> GT overlay</label>
        <button id="reset">Reset view</button>
      </div>
      <canvas id="tl" style="width:100%;height:16px;display:block;
              margin-top:8px;border-radius:4px"></canvas>
      <p class="note" id="tlnote" style="margin:6px 0 0"></p>
      <div class="legend" id="legend"></div>
      <p class="note" style="margin:8px 0 0">Drag to orbit &middot; wheel to
      zoom &middot; the grid is 1 m, the head at the origin faces
      <b>front</b>.</p>
    </div>

    <div>
      <div class="conds" id="conds"></div>
      <div class="panel" style="margin-top:12px">
        <div class="note" id="drawnote"></div>
        <div style="display:flex;gap:8px;align-items:center;margin-top:8px">
          <span class="note">Draw</span>
          <select id="draw"></select>
        </div>
      </div>
    </div>
  </div>

  <h2>Relation satisfaction in this scene</h2>
  <div class="panel"><div id="rsrtable"></div>
    <p class="note" style="margin:10px 0 0">RSR = relation satisfaction rate,
    the fraction of in-scope frames whose signed geometric margin is
    satisfied. Ground truth sits near 1.0 but not at it: the dataset quality
    gate accepts a relation at 0.95, so speed transitions and boundary frames
    are legitimately shaved.</p>
  </div>

  <h2>How these clips were made</h2>
  <div class="panel" id="protocol"></div>
</main>

<audio id="snd" preload="none"></audio>
<script>
const state = {meta:null, scene:null, cond:"REL2TRAJ", draw:0,
               tNorm:0, playing:false, ghost:false};
let yaw=-0.6, pitch=0.5, dist=15;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const snd=document.getElementById('snd');

const fmt = v => (v===null||v===undefined) ? '&mdash;' : v.toFixed(3);
const cls = v => (v===null||v===undefined) ? '' :
                 v>=0.95 ? 'hi' : v>=0.8 ? 'mid' : 'lo';

async function boot(){
  state.meta = await (await fetch('data/index.json')).json();
  const bar = document.getElementById('scenebar');
  state.meta.scenes.forEach((s,i)=>{
    const b=document.createElement('button'); b.className='sc'; b.dataset.i=i;
    b.innerHTML = `<b>${s.target.type}</b><small>${s.motions.join(' + ')}</small>`;
    b.onclick = ()=> loadScene(i);
    bar.appendChild(b);
  });
  renderProtocol();
  loadScene(0);
}

async function loadScene(i){
  const meta = state.meta.scenes[i];
  document.querySelectorAll('.sc').forEach(b=>
    b.classList.toggle('on', +b.dataset.i===i));
  state.scene = await (await fetch(`data/${meta.id}.json`)).json();
  state.idx = i;
  state.draw = 0; state.tNorm = 0; stop();
  if(!state.scene.conds[state.cond]) state.cond = 'GT';
  renderPrompt(); renderConds(); renderDraws(); renderTable();
  setAudio(true); draw();
}

function renderPrompt(){
  const S = state.scene;
  const rel = S.relations.map(r =>
    `<span class="tag ${r.inter?'inter':''}">${r.type}: ${r.subject}
     ${r.reference==='listener'?'':'&rarr; '+r.reference}</span>`).join('');
  const held = S.heldout.length ?
    `<span class="tag hold">held-out cell ${S.heldout.join(', ')}</span>` : '';
  document.getElementById('prompt').innerHTML =
    `<div style="font-size:1.02rem">&ldquo;${S.prompt}&rdquo;</div>
     <div style="margin-top:8px">${rel}${held}</div>`;
  document.getElementById('legend').innerHTML = S.sources.map(s =>
    `<span><span class="dot" style="background:${s.color}"></span>
     ${s.name} <span class="note">(${s.motion})</span></span>`).join('');
  document.getElementById('tlnote').innerHTML =
    `Timeline of <b>${S.target.type}</b>
     (${S.target.subject} &rarr; ${S.target.reference}):
     <span style="color:var(--good)">green</span> = constraint holds,
     <span style="color:var(--bad)">red</span> = violated,
     grey = out of scope. RSR is the green share of the coloured frames.`;
}

function renderConds(){
  const S = state.scene, box = document.getElementById('conds');
  box.innerHTML = '';
  for(const key of state.meta.conds.map(c=>c.key)){
    const c = S.conds[key]; if(!c) continue;
    const d = c.draws[Math.min(state.draw, c.draws.length-1)];
    const b = document.createElement('button');
    b.className = 'cond' + (key===state.cond ? ' on':'');
    b.innerHTML =
      `<div class="row1"><b>${c.label}</b>
        <span class="rsr ${cls(d.target)}">${fmt(d.target)}</span></div>
       <div class="note">${c.note}</div>
       <div class="note">scene mean ${fmt(d.mean)}</div>`;
    b.onclick = ()=>{ state.cond=key; renderConds(); renderDraws();
                      setAudio(false); draw(); };
    box.appendChild(b);
  }
}

function renderDraws(){
  const c = state.scene.conds[state.cond];
  const sel = document.getElementById('draw');
  sel.innerHTML = '';
  c.draws.forEach((d,j)=>{
    const o=document.createElement('option'); o.value=j;
    o.textContent = `${j+1} / ${c.draws.length}` +
      (d.target===null||d.target===undefined ? '' : `  (relation ${d.target.toFixed(2)})`);
    sel.appendChild(o);
  });
  sel.value = Math.min(state.draw, c.draws.length-1);
  sel.disabled = c.draws.length < 2;
  sel.onchange = ()=>{ state.draw = +sel.value; renderConds(); draw(); };
  document.getElementById('drawnote').innerHTML = c.draws.length > 1
    ? `This system is generative: ${c.draws.length} independent draws from the
       same text are shown. <b>Draw 1 is the one you hear</b> &mdash; the first
       draw under the pre-registered seed, no sample was picked by score.`
    : `Deterministic system: one solution per scene.`;
}

function renderTable(){
  const S = state.scene, keys = state.meta.conds.map(c=>c.key)
    .filter(k=>S.conds[k]);
  const rels = S.relations.map(r=>`${r.type}|${r.subject}|${r.reference}`);
  const vals = k => S.conds[k].draws[0].per_relation;
  let html = `<table><tr><th>relation</th>` +
    keys.map(k=>`<th style="text-align:right">${S.conds[k].label}</th>`).join('') +
    `</tr>`;
  for(const key of rels){
    const [type,subj,ref] = key.split('|');
    const row = keys.map(k=>vals(k)[key]);
    const best = Math.max(...row.filter(v=>v!==null&&v!==undefined));
    html += `<tr><td>${type} <span class="note">${subj}${ref==='listener'?'':' &rarr; '+ref}</span></td>` +
      row.map(v=>`<td class="num ${v===best?'best':''}">${fmt(v)}</td>`).join('') +
      `</tr>`;
  }
  html += `<tr><td><b>scene mean</b></td>` + keys.map(k=>
    `<td class="num">${fmt(S.conds[k].draws[0].mean)}</td>`).join('') +
    `</tr></table>`;
  document.getElementById('rsrtable').innerHTML = html;
}

function renderProtocol(){
  const p = state.meta.protocol;
  document.getElementById('protocol').innerHTML = `
    <table>
      <tr><th>scene selection</th><td>${p.selection}</td></tr>
      <tr><th>sampling</th><td>${p.sampling}, ${p.ddim_steps} DDIM steps,
          ${p.m_draws} draws shown</td></tr>
      <tr><th>source material</th><td>${p.material}</td></tr>
      <tr><th>loudness</th><td>${p.loudness}</td></tr>
      <tr><th>renderer</th><td>${p.renderer}</td></tr>
      <tr><th>HRTF</th><td>${p.hrtf}</td></tr>
      <tr><th>level</th><td>${p.level}</td></tr>
      <tr><th>LLM baseline</th><td>${p.llm || 'n/a'}, temperature 0,
          thinking off, 2 retries</td></tr>
    </table>
    <details><summary>coverage of the 12 scenes</summary>
      <p class="note">inter-source relations:
      ${Object.entries(p.relation_coverage).map(([k,v])=>`${k} &times;${v}`).join(', ')}<br>
      motion primitives:
      ${Object.entries(p.motion_coverage).map(([k,v])=>`${k} &times;${v}`).join(', ')}</p>
    </details>`;
}

// ---- audio
function setAudio(reload){
  const c = state.scene.conds[state.cond];
  const pos = state.tNorm * state.scene.duration;
  if(c.audio && (reload || !snd.src.endsWith(c.audio))){
    snd.src = c.audio;
    snd.addEventListener('loadedmetadata', ()=>{
      try { snd.currentTime = Math.min(pos, state.scene.duration-0.05); } catch(e){}
      if(state.playing) snd.play();
    }, {once:true});
  } else if(c.audio){
    try { snd.currentTime = pos; } catch(e){}
  }
}
function stop(){ state.playing=false; snd.pause();
  document.getElementById('play').innerHTML='&#9654; Play'; }

document.getElementById('play').onclick = ()=>{
  state.playing = !state.playing;
  document.getElementById('play').innerHTML =
    state.playing ? '&#10073;&#10073; Pause' : '&#9654; Play';
  if(state.playing){
    if(state.tNorm >= 0.999) state.tNorm = 0;
    setAudio(false); snd.play().catch(()=>{});
    requestAnimationFrame(tick);
  } else snd.pause();
};
snd.addEventListener('timeupdate', ()=>{
  if(state.playing){ state.tNorm = snd.currentTime / state.scene.duration; draw(); }
});
snd.addEventListener('ended', ()=>{ state.tNorm = 1; stop(); draw(); });
function tick(){
  if(!state.playing) return;
  if(!snd.paused) state.tNorm = snd.currentTime / state.scene.duration;
  draw(); requestAnimationFrame(tick);
}
document.getElementById('time').oninput = e=>{
  state.tNorm = +e.target.value/1000;
  try { snd.currentTime = state.tNorm * state.scene.duration; } catch(err){}
  draw();
};
document.getElementById('ghost').onchange = e=>{
  state.ghost = e.target.checked; draw(); };
document.getElementById('reset').onclick = ()=>{
  yaw=-0.6; pitch=0.5; dist=15; draw(); };

// ---- 3D projection (world: x right, y front, z up)
function project(p){
  const sy=Math.sin(yaw), cy=Math.cos(yaw), sp=Math.sin(pitch), cp=Math.cos(pitch);
  const x = p[0]*cy - p[1]*sy, y = p[0]*sy + p[1]*cy, z = p[2];
  const y2 = y*cp - z*sp, z2 = y*sp + z*cp;
  const d = dist/(dist+y2), scale = Math.min(cv.width,cv.height)*0.085;
  return [cv.width/2 + x*d*scale, cv.height/2 - z2*d*scale, d];
}
function seg(a,b,color,w,alpha,dash){
  const A=project(a), B=project(b);
  ctx.setLineDash(dash||[]);
  ctx.strokeStyle=color; ctx.globalAlpha=alpha; ctx.lineWidth=w;
  ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(B[0],B[1]); ctx.stroke();
  ctx.globalAlpha=1; ctx.setLineDash([]);
}
function drawPath(src, color, tMax, T, opts){
  const a = opts.alpha === undefined ? 1 : opts.alpha;
  for(let i=1;i<src.x.length;i++){
    const done = T[i] <= tMax;
    seg([src.x[i-1],src.y[i-1],src.z[i-1]], [src.x[i],src.y[i],src.z[i]],
        color, done?2.8:1.7, (done?0.95:0.5)*a, opts.dash);
    if(i % 8 === 0 && Math.abs(src.z[i]) > 0.2)
      seg([src.x[i],src.y[i],src.z[i]], [src.x[i],src.y[i],0],
          color, 1, 0.2*a);
  }
  if(opts.markers === false) return;
  const P0=project([src.x[0],src.y[0],src.z[0]]);
  ctx.fillStyle=color; ctx.globalAlpha=a;
  ctx.beginPath(); ctx.arc(P0[0],P0[1],4,0,7); ctx.fill();
  let k=0; while(k<T.length-1 && T[k+1]<=tMax) k++;
  const M=project([src.x[k],src.y[k],src.z[k]]);
  ctx.beginPath(); ctx.arc(M[0],M[1],7,0,7); ctx.fill();
  ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.setLineDash([]); ctx.stroke();
  ctx.globalAlpha=1;
}
function draw(){
  const S=state.scene; if(!S) return;
  const w=cv.clientWidth, h=cv.clientHeight;
  if(cv.width!==w||cv.height!==h){ cv.width=w; cv.height=h; }
  ctx.clearRect(0,0,w,h);
  for(let i=-8;i<=8;i++){
    seg([i,-8,0],[i,8,0],'#243043',1,i===0?0.8:0.3);
    seg([-8,i,0],[8,i,0],'#243043',1,i===0?0.8:0.3);
  }
  ctx.fillStyle='#7d879a'; ctx.font='12px system-ui';
  const f=project([0,7.7,0]); ctx.fillText('front', f[0]+4, f[1]);
  const r=project([7.7,0,0]); ctx.fillText('right', r[0]+4, r[1]);
  seg([0,0,0],[0,0,3.6],'#33405a',1,0.7);
  const hd=project([0,0,0]);
  ctx.fillStyle='#e9ecf1';
  ctx.beginPath(); ctx.arc(hd[0],hd[1],7*hd[2],0,7); ctx.fill();
  const nose=project([0,0.5,0]), eL=project([-0.2,0.1,0]), eR=project([0.2,0.1,0]);
  ctx.beginPath(); ctx.moveTo(nose[0],nose[1]); ctx.lineTo(eL[0],eL[1]);
  ctx.lineTo(eR[0],eR[1]); ctx.closePath(); ctx.fill();

  const T=S.t, tMax=S.duration*state.tNorm;
  if(state.ghost && state.cond!=='GT'){
    S.conds.GT.draws[0].src.forEach((s,i)=>
      drawPath(s, S.sources[i].color, tMax, T,
               {alpha:0.28, dash:[5,5], markers:false}));
  }
  const c=S.conds[state.cond];
  const d=c.draws[Math.min(state.draw, c.draws.length-1)];
  d.src.forEach((s,i)=> drawPath(s, S.sources[i].color, tMax, T, {}));

  // the relation itself: a link between the two related sources at this
  // instant, green while the geometric constraint holds, red while it fails
  let k=0; while(k<T.length-1 && T[k+1]<=tMax) k++;
  const [ia,ib] = S.target_idx, st = d.state[k];
  const A=d.src[ia], B=d.src[ib];
  if(st >= 0)
    seg([A.x[k],A.y[k],A.z[k]], [B.x[k],B.y[k],B.z[k]],
        st===1 ? '#39c07c' : '#e05c5c', 2.2, 0.9, st===1 ? [] : [6,4]);
  drawTimeline(d.state, k);

  document.getElementById('tlabel').textContent =
    tMax.toFixed(1)+' / '+S.duration.toFixed(1)+' s';
  document.getElementById('time').value = Math.round(state.tNorm*1000);
}

function drawTimeline(stateArr, k){
  const el=document.getElementById('tl'), w=el.clientWidth, h=16;
  if(el.width!==w){ el.width=w; } el.height=h;
  const c2=el.getContext('2d'), n=stateArr.length, bw=w/n;
  c2.clearRect(0,0,w,h);
  for(let i=0;i<n;i++){
    c2.fillStyle = stateArr[i]===1 ? '#2f7f57'
                 : stateArr[i]===0 ? '#a8393c' : '#2b3341';
    c2.fillRect(i*bw, 0, Math.ceil(bw)+0.5, h);
  }
  c2.fillStyle='#e9ecf1';
  c2.fillRect(Math.max(0, k*bw-1), 0, 2, h);
}
let dragging=false, px=0, py=0;
cv.addEventListener('pointerdown', e=>{ dragging=true; px=e.clientX; py=e.clientY;
  cv.classList.add('drag'); cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointermove', e=>{
  if(!dragging) return;
  yaw += (e.clientX-px)*0.008;
  pitch = Math.max(-0.1, Math.min(1.45, pitch + (e.clientY-py)*0.008));
  px=e.clientX; py=e.clientY; draw(); });
cv.addEventListener('pointerup', ()=>{ dragging=false; cv.classList.remove('drag'); });
cv.addEventListener('wheel', e=>{ e.preventDefault();
  dist = Math.max(6, Math.min(38, dist*(e.deltaY>0?1.1:0.9))); draw(); },
  {passive:false});
window.addEventListener('resize', draw);
boot();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
