# Renders the gsm8k-rl-rank-sweep results as a self-contained HTML report.
#
# Charts are inline SVG built from the data -- no chart library, no network
# fetches -- so the file can be published or emailed as one artifact.
#
# Usage:
#   uv run python rank_sweep_html.py --runs-dir <dir> --out report.html

import argparse
import html
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
from rank_sweep_report import config_order, load_runs, noise_floor, summarize, summarize_configs

# Validated categorical palette, light / dark pairs. Slots are handed out in
# configuration order, so a configuration keeps its hue across every chart and
# across reports that contain different subsets.
PALETTE = [
  ("#2a78d6", "#3987e5"),
  ("#eb6834", "#d95926"),
  ("#1baf7a", "#199e70"),
  ("#9457c9", "#a06fd4"),
  ("#c9a227", "#d9b53a"),
  ("#3aa3a3", "#48b5b5"),
]
# Replicates of one configuration share a colour and separate by dash pattern.
DASHES = ["", "5 3", "1 3", "7 3 2 3"]


def config_label(config: str) -> str:
  if config == "fullft":
    return "Full fine-tuning"
  m = re.match(r"lora-r(\d+)$", config)
  return f"LoRA rank {m.group(1)}" if m else config


def run_label(config: str, replicate) -> str:
  base = config_label(config)
  return base if replicate in (None, "") or _isnan(replicate) else f"{base} ({replicate})"


def _isnan(v) -> bool:
  return v != v


def palette_for(configs) -> dict[str, tuple[str, str]]:
  return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(config_order(configs))}


def dash_for(replicates) -> dict:
  present = sorted({r for r in replicates if r is not None and not _isnan(r)})
  out = {r: DASHES[i % len(DASHES)] for i, r in enumerate(present)}
  out[None] = ""
  return out


def _dash_attr(pattern: str) -> str:
  return f' stroke-dasharray="{pattern}"' if pattern else ""


# Head-room past the plot so end labels overflowing the viewBox are not cropped.
LABEL_OVERFLOW = 260


def reveal(marks: str, key: str, x: float, width: float, height: float, dur: float = 1.8) -> str:
  """Wrap plotted marks in a clip rect that widens, so the chart draws itself.

  A clip is used rather than the usual stroke-dashoffset trick because the
  replicate lines already carry a dash pattern for identity, and animating
  dashoffset would destroy it. Clipping also sweeps every series together,
  including the raw underlay and the end labels, which is the point: the
  reveal reads as training progressing, not as lines being drawn one by one.

  The animation begins on demand (``begin="indefinite"``) so it can be tied to
  the chart scrolling into view and replayed.
  """
  # End labels sit past the right edge of the viewBox and stay visible only
  # because the SVG does not clip overflow. The reveal rect therefore has to
  # extend well beyond the plot, or it would crop them permanently. The JS also
  # drops the clip once the sweep finishes, so nothing depends on this margin.
  full = width + LABEL_OVERFLOW
  return (
    f'<clipPath id="clip-{key}"><rect x="{x:.1f}" y="0" height="{height:.0f}" width="0">'
    f'<animate id="anim-{key}" attributeName="width" from="0" to="{full:.1f}" '
    f'dur="{dur}s" fill="freeze" begin="indefinite" calcMode="spline" '
    f'keyTimes="0;1" keySplines="0.25 0.1 0.25 1"/></rect></clipPath>'
    f'<g id="marks-{key}" clip-path="url(#clip-{key})">{marks}</g>'
  )


W, H = 720, 380
PAD_L, PAD_R, PAD_T, PAD_B = 56, 96, 16, 40
# Minimum vertical separation between end labels before they overprint.
LABEL_GAP = 15


def _scale(v, lo, hi, out_lo, out_hi):
  if hi == lo:
    return (out_lo + out_hi) / 2
  return out_lo + (v - lo) / (hi - lo) * (out_hi - out_lo)


def _rolling(vals, window):
  out = []
  for i in range(len(vals)):
    chunk = vals[max(0, i - window + 1) : i + 1]
    out.append(sum(chunk) / len(chunk))
  return out


def _runs_in_order(df):
  """(run, config, replicate) tuples, configurations in claim order, replicates alphabetical."""
  seen = df[["run", "config", "replicate"]].drop_duplicates()
  order = config_order(seen["config"])
  rows = []
  for config in order:
    sub = seen[seen["config"] == config].sort_values("replicate", na_position="first")
    for _, r in sub.iterrows():
      rep = None if r["replicate"] is None or _isnan(r["replicate"]) else r["replicate"]
      rows.append((str(r["run"]), config, rep))
  return rows


def config_series(df, smooth: int):
  """Reward per step averaged over replicates, per configuration.

  Pace and gap comparisons use this rather than a single replicate, so a lucky
  seed cannot decide when a configuration is said to have crossed a threshold.
  """
  out = {}
  for config in config_order(df["config"]):
    sub = df[df["config"] == config]
    means = sub.groupby("step")["reward"].mean().sort_index()
    out[config] = (list(means.index), _rolling([float(v) for v in means], smooth))
  return out


def reward_chart(df, smooth: int, idx: int = 0) -> str:
  runs = _runs_in_order(df)
  dashes = dash_for(df["replicate"])
  xmax = float(df["step"].max())
  ymin, ymax = 0.0, max(1.0, float(df["reward"].max()) * 1.05)

  def px(step):
    return _scale(step, 0, xmax, PAD_L, W - PAD_R)

  def py(reward):
    return _scale(reward, ymin, ymax, H - PAD_B, PAD_T)

  parts = []
  marks = []
  end_labels = []
  # Horizontal grid + y labels.
  for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    if t > ymax:
      continue
    y = py(t)
    parts.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>')
    parts.append(f'<text class="tick" x="{PAD_L - 10}" y="{y + 4:.1f}" text-anchor="end">{t:.1f}</text>')
  # X ticks.
  for t in range(0, int(xmax) + 1, 10):
    x = px(t)
    parts.append(f'<text class="tick" x="{x:.1f}" y="{H - PAD_B + 20}" text-anchor="middle">{t}</text>')

  series_json = {}
  for name, config, replicate in runs:
    g = df[df["run"] == name].sort_values("step")
    steps = [float(s) for s in g["step"]]
    raw = [float(v) for v in g["reward"]]
    rolled = _rolling(raw, smooth)
    series_json[name] = {"steps": steps, "raw": raw, "rolled": rolled, "config": config, "label": run_label(config, replicate)}

    dash = _dash_attr(dashes[replicate])
    raw_pts = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, raw, strict=False))
    roll_pts = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, rolled, strict=False))
    marks.append(f'<polyline class="raw" data-config="{config}" points="{raw_pts}"/>')
    marks.append(f'<polyline class="line" data-config="{config}"{dash} points="{roll_pts}"/>')
    end_labels.append({"config": config, "text": run_label(config, replicate), "x": px(steps[-1]) + 8, "y": py(rolled[-1]) + 4})

  # Direct labels carry identity so it never rests on colour alone -- but runs
  # that converge finish at the same height and would overprint each other.
  # Push them apart, keeping their vertical order. With replicates there are
  # twice as many labels, so this matters more than it did.
  end_labels.sort(key=lambda item: item["y"])
  for i in range(1, len(end_labels)):
    gap = end_labels[i]["y"] - end_labels[i - 1]["y"]
    if gap < LABEL_GAP:
      end_labels[i]["y"] = end_labels[i - 1]["y"] + LABEL_GAP
  for item in end_labels:
    marks.append(f'<text class="endlabel" data-config="{item["config"]}" x="{item["x"]:.1f}" y="{item["y"]:.1f}">{item["text"]}</text>')

  # Axes stay put; only the data sweeps in. Crosshair and hit area sit outside
  # the clip so hovering works whether or not the reveal has finished.
  parts.append(reveal("".join(marks), f"r{idx}", PAD_L, W - PAD_L, H))
  parts.append(f'<line id="crosshair-{idx}" class="crosshair" x1="0" y1="{PAD_T}" x2="0" y2="{H - PAD_B}" style="opacity:0"/>')
  parts.append(f'<rect id="hitarea-{idx}" x="{PAD_L}" y="{PAD_T}" width="{W - PAD_L - PAD_R}" height="{H - PAD_T - PAD_B}" fill="transparent"/>')

  svg = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Reward versus training step for {len(runs)} runs">{"".join(parts)}</svg>'
  cfg = json.dumps({"idx": idx, "series": series_json, "padL": PAD_L, "padR": PAD_R, "w": W, "xmax": xmax})
  return svg, cfg


def gap_chart(gaps, floor: float = float("nan"), idx: int = 0) -> str:
  """Gap to FullFT at successive matched-step checkpoints, against the replicate spread."""
  checkpoints = [c for c, _ in gaps]
  w, h = 720, 260
  pad_l, pad_r, pad_t, pad_b = 56, 92, 16, 38
  extent = floor if floor == floor else 0.0
  lo = min(min(v.values()) for _, v in gaps) - 0.03
  hi = max(max(v.values()) for _, v in gaps) + 0.03
  lo, hi = min(lo, -extent - 0.01), max(hi, extent + 0.01)

  def px(i):
    return _scale(i, 0, max(1, len(checkpoints) - 1), pad_l, w - pad_r)

  def py(v):
    return _scale(v, lo, hi, h - pad_b, pad_t)

  parts = [f'<line class="zero" x1="{pad_l}" y1="{py(0):.1f}" x2="{w - pad_r}" y2="{py(0):.1f}"/>']
  parts.append(f'<text class="tick" x="{w - pad_r + 8}" y="{py(0) + 4:.1f}">parity</text>')
  for i, c in enumerate(checkpoints):
    parts.append(f'<text class="tick" x="{px(i):.1f}" y="{h - pad_b + 20}" text-anchor="middle">{c}</text>')
  marks = []
  tracked = [c for c in config_order({k for _, v in gaps for k in v}) if c != "fullft"]
  # The noise band makes the chart readable on its own: a marker inside it is
  # not distinguishable from seed variation, whatever its distance from zero.
  if floor == floor and floor > 0:
    parts.insert(
      0, f'<rect class="band" x="{pad_l}" y="{py(floor):.1f}" width="{w - pad_l - pad_r:.1f}" height="{abs(py(-floor) - py(floor)):.1f}"/>'
    )
    parts.append(f'<text class="tick" x="{w - pad_r + 8}" y="{py(floor) - 4:.1f}">replicate spread</text>')
  for name in tracked:
    pts = " ".join(f"{px(i):.1f},{py(v[name]):.1f}" for i, (_, v) in enumerate(gaps) if name in v)
    marks.append(f'<polyline class="line" data-config="{name}" points="{pts}"/>')
    for i, (_, v) in enumerate(gaps):
      if name in v:
        marks.append(f'<circle class="dot" data-config="{name}" cx="{px(i):.1f}" cy="{py(v[name]):.1f}" r="4.5"/>')
    last = gaps[-1][1]
    lx, ly = px(len(checkpoints) - 1) + 8, py(last[name]) + 4
    marks.append(f'<text class="endlabel" data-config="{name}" x="{lx:.1f}" y="{ly:.1f}">{config_label(name)}</text>')
  parts.append(reveal("".join(marks), f"g{idx}", pad_l, w - pad_l, h))
  return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Gap to full fine-tuning at successive checkpoints">{"".join(parts)}</svg>'


def build_gaps(df, tail: int):
  """Gap of each configuration's mean to full fine-tuning, at matched-step checkpoints.

  Replicates are folded in before the gap is taken, so this tracks the
  configuration rather than whichever seed happened to be ahead.
  """
  common = int(df.groupby("run")["step"].count().min())
  checkpoints = [c for c in (6, 14, 28, common) if c <= common]
  out = []
  for c in checkpoints:
    sub = df[df["step"] < c]
    summary, _ = summarize(sub, min(tail, max(1, c // 3)))
    configs = summarize_configs(summary)
    if "gap_vs_fullft" not in configs:
      continue
    out.append((c, {str(r["config"]): float(r["gap_vs_fullft"]) for _, r in configs.iterrows() if r["config"] != "fullft"}))
  return out


def color_css(configs) -> str:
  """Custom properties and stroke rules for whichever configurations are present.

  Generated rather than fixed, so a sweep that adds a rank does not need the
  stylesheet edited alongside it.
  """
  pal = palette_for(configs)
  light = "".join(f"--c-{c}:{v[0]};" for c, v in pal.items())
  dark = "".join(f"--c-{c}:{v[1]};" for c, v in pal.items())
  strokes = "".join(f"[data-config={c}]{{stroke:var(--c-{c})}}circle[data-config={c}]{{fill:var(--c-{c})}}" for c in pal)
  return f":root{{{light}}}@media (prefers-color-scheme:dark){{:root{{{dark}}}}}{strokes}text.endlabel{{stroke:none}}"


CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e6e5e0}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#33322f}}
*{box-sizing:border-box}
body{background:var(--surface);color:var(--ink);margin:0;
font:17px/1.65 ui-serif,Georgia,"Times New Roman",serif;-webkit-font-smoothing:antialiased}
main{max-width:46rem;margin:0 auto;padding:4rem 1.5rem 6rem}
h1{font-size:2.1rem;line-height:1.2;font-weight:600;letter-spacing:-.01em;margin:0 0 .6rem}
h2{font-size:1.25rem;font-weight:600;margin:3rem 0 .8rem;letter-spacing:-.005em}
h3{font-size:1rem;font-weight:600;margin:2rem 0 .5rem;color:var(--ink2)}
.byline{color:var(--ink2);font-size:.95rem;margin:0 0 2.5rem;padding-bottom:1.5rem;border-bottom:1px solid var(--rule)}
p{margin:0 0 1.1rem}
a{color:inherit;text-decoration:underline;text-underline-offset:2px;text-decoration-color:var(--ink2)}
.lede{font-size:1.1rem;color:var(--ink)}
figure{margin:2.2rem 0;padding:0}
figcaption{color:var(--ink2);font-size:.9rem;line-height:1.5;margin-top:.7rem}
.replay{background:none;border:1px solid var(--rule);border-radius:5px;color:var(--ink2);
cursor:pointer;font:12px ui-sans-serif,system-ui,sans-serif;padding:.25rem .6rem;margin-top:.5rem}
.replay:hover{border-color:var(--ink2);color:var(--ink)}
@media (prefers-reduced-motion:reduce){.replay{display:none}}
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--rule);stroke-width:1}
.zero{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4;opacity:.6}
.tick{fill:var(--ink2);font:11px ui-sans-serif,system-ui,sans-serif}
.raw{fill:none;stroke-width:1;opacity:.22}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.endlabel{font:12px ui-sans-serif,system-ui,sans-serif;fill:var(--ink2)}
.crosshair{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3}
.band{fill:var(--ink2);opacity:.07}
.legend{display:flex;gap:1.4rem;flex-wrap:wrap;font:13px ui-sans-serif,system-ui,sans-serif;color:var(--ink2);margin:.2rem 0 .4rem}
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:.45rem;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font:14px ui-sans-serif,system-ui,sans-serif;margin:1.2rem 0}
th,td{text-align:right;padding:.5rem .6rem;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
tbody tr:last-child td{border-bottom:none}
.num{font-variant-numeric:tabular-nums}
.callout{border-left:2px solid var(--rule);padding:.1rem 0 .1rem 1.1rem;color:var(--ink2);margin:1.6rem 0}
ul{margin:0 0 1.1rem;padding-left:1.2rem;color:var(--ink2)}
li{margin:0 0 .35rem}
code{font:.88em ui-monospace,SFMono-Regular,Menlo,monospace;background:rgba(0,0,0,.04);padding:.1em .3em;border-radius:3px}
details{margin:1.2rem 0;font:14px ui-sans-serif,system-ui,sans-serif;color:var(--ink2)}
summary{cursor:pointer}
#tip{position:fixed;pointer-events:none;opacity:0;background:var(--surface);border:1px solid var(--rule);
border-radius:6px;padding:.5rem .65rem;font:12px ui-sans-serif,system-ui,sans-serif;color:var(--ink);
box-shadow:0 2px 10px rgba(0,0,0,.08);transition:opacity .1s}
footer{margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--rule);color:var(--ink2);font-size:.85rem}
"""

JS = """
// Chart reveal. Each animated chart has a clip rect whose width is driven by a
// SMIL <animate>; playing it sweeps the data in from the left. Charts play once
// when first scrolled into view, and the replay button re-runs them.
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function revealPlay(key) {
  const anim = document.getElementById('anim-' + key);
  if (!anim) return;
  const rect = anim.parentNode;
  if (reduceMotion) { rect.setAttribute('width', anim.getAttribute('to')); return; }
  const marks = document.getElementById('marks-' + key);
  if (marks) marks.setAttribute('clip-path', 'url(#clip-' + key + ')');
  try { anim.beginElement(); } catch (e) { rect.setAttribute('width', anim.getAttribute('to')); }
}
// Once the sweep is done the clip has no job left, and keeping it risks
// cropping anything that renders outside the viewBox.
document.querySelectorAll('[id^="anim-"]').forEach((anim) => {
  anim.addEventListener('endEvent', () => {
    const m = document.getElementById('marks-' + anim.id.slice(5));
    if (m) m.removeAttribute('clip-path');
  });
});
document.querySelectorAll('[id^="anim-"]').forEach((anim) => {
  const key = anim.id.slice(5);
  const svg = anim.closest('svg');
  if (!svg) return;
  if (reduceMotion) { revealPlay(key); return; }
  // Charts already on screen at load still animate; the observer fires immediately
  // for those, so there is no separate first-paint path to keep in sync.
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) { revealPlay(key); io.disconnect(); }
    });
  }, { threshold: 0.35 });
  io.observe(svg);
});
// A chart inside a closed <details> has zero size, so the observer never fires.
// Play it when the disclosure opens instead.
document.querySelectorAll('details[data-reveal]').forEach((d) => {
  d.addEventListener('toggle', () => { if (d.open) revealPlay(d.dataset.reveal); });
});
document.querySelectorAll('.replay').forEach((b) => {
  b.addEventListener('click', () => b.dataset.keys.split(',').forEach(revealPlay));
});

const CHARTS = %CHARTS%;
const NAMES = %NAMES%;
const tip = document.getElementById('tip');
for (const cfg of CHARTS) {
  const hit = document.getElementById('hitarea-' + cfg.idx);
  const cross = document.getElementById('crosshair-' + cfg.idx);
  if (!hit) continue;
  const svg = hit.ownerSVGElement;
  hit.addEventListener('mousemove', (e) => {
    const box = svg.getBoundingClientRect();
    const scale = cfg.w / box.width;
    const xSvg = (e.clientX - box.left) * scale;
    const frac = (xSvg - cfg.padL) / (cfg.w - cfg.padL - cfg.padR);
    const step = Math.round(Math.max(0, Math.min(1, frac)) * cfg.xmax);
    cross.setAttribute('x1', xSvg); cross.setAttribute('x2', xSvg);
    cross.style.opacity = 1;
    let rows = '';
    for (const name of Object.keys(cfg.series)) {
      const s = cfg.series[name];
      const i = s.steps.indexOf(step);
      if (i >= 0) rows += `<div><span class="swatch" style="background:var(--${name})"></span>${NAMES[name]} <b>${s.rolled[i].toFixed(3)}</b></div>`;
    }
    tip.innerHTML = `<div style="color:var(--ink2);margin-bottom:.3rem">step ${step}</div>${rows}`;
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top = (e.clientY - 10) + 'px';
    tip.style.opacity = 1;
  });
  hit.addEventListener('mouseleave', () => { tip.style.opacity = 0; cross.style.opacity = 0; });
}
"""


def steps_to_thresholds(df, smooth: int):
  """First step at which each configuration's replicate-averaged mean crosses each threshold."""
  out = []
  for config, (_steps, rolled) in config_series(df, smooth).items():
    row = {"config": config, "early": sum(rolled[:15]) / min(15, len(rolled))}
    for th in (0.5, 0.8, 0.9):
      row[th] = next((i for i, v in enumerate(rolled) if v >= th), None)
    out.append(row)
  return out


def _pace_table(df, smooth: int) -> str:
  rows = ""
  for r in steps_to_thresholds(df, smooth):
    cells = "".join(f'<td class="num">{"&mdash;" if r[th] is None else r[th]}</td>' for th in (0.5, 0.8, 0.9))
    rows += f'<tr><td>{config_label(r["config"])}</td><td class="num">{r["early"]:.3f}</td>{cells}</tr>'
  return (
    "<table><thead><tr><th>Configuration</th><th>Mean reward<br>steps 0-14</th>"
    "<th>Step to 0.5</th><th>Step to 0.8</th><th>Step to 0.9</th></tr></thead>"
    f"<tbody>{rows}</tbody></table>"
  )


def _summary_table(configs, tail: int) -> str:
  """Per configuration, with the replicate spread beside the gap it has to beat."""
  floor = noise_floor(configs)
  rows = ""
  for _, r in configs.iterrows():
    name = str(r["config"])
    gap = r.get("gap_vs_fullft", 0.0)
    spread = r["spread"]
    spread_txt = "&mdash;" if spread != spread else f"{spread:.4f}"
    if name == "fullft":
      gap_txt, verdict = "&mdash;", "reference"
    else:
      gap_txt = f"{gap:+.4f}"
      verdict = "&mdash;" if floor != floor else ("within noise" if abs(gap) <= floor else "exceeds noise")
    rows += (
      f"<tr><td>{config_label(name)}</td>"
      f'<td class="num">{r["mean"]:.3f}</td>'
      f'<td class="num">{spread_txt}</td>'
      f'<td class="num">{gap_txt}</td>'
      f"<td>{verdict}</td>"
      f'<td class="num">{r["sec_per_step"]:.0f}s</td></tr>'
    )
  return (
    f"<table><thead><tr><th>Configuration</th><th>Mean reward<br>(last {tail})</th>"
    f"<th>Replicate<br>spread</th><th>Gap vs FullFT</th><th>Verdict</th><th>Per step</th></tr></thead>"
    f"<tbody>{rows}</tbody></table>"
  )


def _runs_table(summary, tail: int) -> str:
  """Every individual run, so the folded table above can be checked against it."""
  rows = ""
  for _, r in summary.iterrows():
    rows += (
      f"<tr><td>{run_label(str(r['config']), r['replicate'])}</td>"
      f'<td class="num">{r["tail_mean"]:.3f}</td>'
      f'<td class="num">{r["tail_std"]:.3f}</td>'
      f'<td class="num">{r["final"]:.3f}</td>'
      f'<td class="num">{r["sec_per_step"]:.0f}s</td></tr>'
    )
  return (
    f"<table><thead><tr><th>Run</th><th>Mean reward<br>(last {tail})</th>"
    f"<th>&plusmn;&nbsp;std</th><th>Final</th><th>Per step</th></tr></thead>"
    f"<tbody>{rows}</tbody></table>"
  )


def _legend(df) -> str:
  return "".join(f'<span><span class="swatch" style="background:var(--c-{c})"></span>{config_label(c)}</span>' for c in config_order(df["config"]))


def run_pace_note(df, summary) -> str:
  """Describe how quickly each configuration got there, which the endpoint table hides."""
  pace = {r["config"]: r for r in steps_to_thresholds(df, 5)}
  ff, r1 = pace.get("fullft", {}), pace.get("lora-r1", {})
  if ff.get(0.9) is not None and r1.get(0.9) is not None:
    return (
      f"The endpoints agree, but the paths differ. Full fine-tuning reached 0.9 at step "
      f"{ff[0.9]}; LoRA rank 1 needed {r1[0.9]}. Over the first 15 steps full fine-tuning "
      f"averaged {ff['early']:.2f} against rank 1's {r1['early']:.2f}. The paper's claim covers "
      f"sample efficiency as well as final performance, so this is a difference worth naming. "
      f"Two things plausibly account for it and neither is tested here: LoRA initialises B at "
      f"zero, so its effective learning rate ramps up over the first steps, and we used a fixed "
      f"10&times; learning-rate ratio rather than sweeping each run to its own optimum, which is "
      f"what the paper did."
    )
  return "Neither run reached 0.9 within the run, so the pace comparison below is limited to the lower thresholds. No run leads consistently."


def tail_sensitivity(df, windows=(5, 10, 15, 20)):
  """Gap and noise floor recomputed over several tail windows.

  Reward sits near a ceiling by the end of these runs, so the tail mean moves
  with the window chosen. Showing the dependence is the honest alternative to
  picking one window and reporting its verdict as the result.
  """
  rows = []
  for w in windows:
    summary, common = summarize(df, w)
    if w > common:
      continue
    configs = summarize_configs(summary)
    if "gap_vs_fullft" not in configs:
      continue
    floor = noise_floor(configs)
    gaps = {str(r["config"]): float(r["gap_vs_fullft"]) for _, r in configs.iterrows() if r["config"] != "fullft"}
    rows.append((w, floor, gaps))
  return rows


def _sensitivity_table(df) -> str:
  rows = tail_sensitivity(df)
  if not rows:
    return ""
  tracked = config_order({k for _, _, g in rows for k in g})
  head = "".join(f"<th>{config_label(c)}</th>" for c in tracked)
  body = ""
  for w, floor, gaps in rows:
    cells = ""
    for c in tracked:
      gap = gaps.get(c, 0.0)
      mark = "" if floor != floor or abs(gap) <= floor else " *"
      cells += f'<td class="num">{gap:+.4f}{mark}</td>'
    body += f'<tr><td class="num">{w}</td><td class="num">{floor:.4f}</td>{cells}</tr>'
  return (
    f"<table><thead><tr><th>Tail window</th><th>Replicate spread</th>{head}</tr></thead>"
    f"<tbody>{body}</tbody></table>"
    "<p>An asterisk marks a gap larger than the replicate spread at that window.</p>"
  )


def _lora_count(summary) -> int:
  return int(summary[summary["config"] != "fullft"]["run"].nunique())


def sharing_costs(experiments) -> list[str]:
  """Per-experiment sentence: how many adapters shared a worker, and what it cost.

  Derived rather than written down, because the multiplier depends on how many
  runs the scheduler happened to pack onto one device -- which is the property
  this section is about.
  """
  out = []
  for exp in experiments:
    s = exp["summary"]
    lora, fft = s[s["config"] != "fullft"], s[s["config"] == "fullft"]
    if lora.empty or fft.empty:
      continue
    shared, solo = float(lora["sec_per_step"].mean()), float(fft["sec_per_step"].mean())
    out.append(
      f"{exp['model']}: {_lora_count(s)} LoRA runs on one trainer and one sampler at "
      f"{shared:.0f}s per step, against {solo:.0f}s for full fine-tuning &mdash; "
      f"{shared / solo:.1f}&times; for sharing."
    )
  return out


def limits(total: int, largest: int) -> list[tuple[str, str, str, str, str]]:
  """The three limitations, with the counts this run actually produced."""
  return [
    (
      "Finished jobs keep their workers",
      "Partway through, with the shorter runs complete, every worker Pod "
      "belonging to a finished run was still going, holding its ResourceClaim. Two H100 "
      "nodes and three L4 nodes were pinned by work that had already stopped.",
      "The Kubernetes worker manager implements a per-model <code>shutdown()</code>, but the "
      "only caller is <code>shutdown_all()</code>, which is a no-op in that subclass. Nothing "
      "runs when a training job ends. The claim reconciler collects only claims with no "
      "referencing Pod, so a Pod that outlives its job keeps the claim alive indefinitely.",
      "Call <code>shutdown()</code> when a worker's last active job finishes; the existing "
      "reconciler then reclaims the claim on its next pass. The one subtlety is that LoRA "
      "workers are multi-tenant, so this needs a per-worker count of live adapters rather "
      "than a per-job hook.",
      "Small",
    ),
    (
      "LoRA runs on one base model cannot spread across workers",
      f"The {largest} LoRA runs on the larger model shared a single trainer Pod and a single "
      f"sampler Pod, time-sliced {largest} ways, while other GPUs of the same tier were free.",
      "A worker Pod's name is derived from the base model with a fixed replica suffix, so "
      "every LoRA job on a given base model resolves to the same Pod. There is no way to "
      "ask for a second worker on that model. <code>OPEN_RL_MAX_WORKERS_PER_CLAIM</code> "
      "caps Pods per claim, which has no effect when the packing happens inside one Pod.",
      "Thread a replica index through Pod naming and claim selection, and add a policy for "
      "how many adapters a worker takes before a new one opens. Packing is the right default "
      "under contention &mdash; N adapters on one device hold one copy of the frozen base "
      "weights instead of N &mdash; so the policy needs a capacity signal, not a fixed cap.",
      "Moderate",
    ),
    (
      "Placement is decided once and never revisited",
      f"As the full fine-tuning runs finished, their H100s went idle and stayed idle. The LoRA "
      f"runs continued {largest}-way time-sliced next to free hardware for the rest of the "
      f"experiment.",
      "Claim selection runs once, at first launch. Base-model affinity is tried before any "
      "capacity test, and nothing re-evaluates the decision afterwards. A job that starts "
      "under contention stays packed even when the contention clears.",
      "Two paths. The cheaper one is admission-time sequencing: hold jobs in a queue and "
      "launch as capacity frees, instead of starting every job at once. The complete one is "
      "migrating a live adapter &mdash; checkpoint, relaunch on the freed device, resume. The "
      "snapshot and restore machinery already exists; nothing drives it for rebalancing.",
      "Large",
    ),
  ]


def _early_table(exp, keys, steps=(0, 2, 4, 9, 14, 19, 29, 39)) -> str:
  """Per-configuration trajectory of a few diagnostic series over selected steps."""
  df = exp["df"]
  head = "".join(f'<th class="num">{s}</th>' for s in steps)
  body = ""
  for label, key, fmt in keys:
    if key not in df.columns:
      continue
    for config in config_order(df["config"]):
      sub = df[df["config"] == config]
      means = sub.groupby("step")[key].mean()
      cells = "".join(f'<td class="num">{means[s]:{fmt}}</td>' if s in means.index else '<td class="num">&mdash;</td>' for s in steps)
      body += f"<tr><td>{label}</td><td>{config_label(config)}</td>{cells}</tr>"
  return f"<table><thead><tr><th>Series</th><th>Configuration</th>{head}</tr></thead><tbody>{body}</tbody></table>"


def early_training_section(experiments, max_tokens: int) -> str:
  """Why full fine-tuning led early at one model size and not the other."""
  if len(experiments) < 2:
    return ""
  big = experiments[-1]
  small = experiments[0]
  keys = [
    ("Format score", "format", ".2f"),
    ("Action tokens/turn", "ac_tokens_per_turn", ".0f"),
    ("Groups scoring zero", "frac_all_bad", ".2f"),
  ]
  return f"""
<h2>Why full fine-tuning led early at {big["model"]}</h2>
<p>The endpoint tables say the configurations agree. The pace tables say full fine-tuning
got there first at {big["model"]} and not at {small["model"]}. That asymmetry has a
specific cause, and it is not capacity.</p>

<p>At {big["model"]}, LoRA's reward goes <em>negative</em> around step 2 and stays near
zero until roughly step 14 before recovering. Full fine-tuning never dips. Decomposing
reward into its parts shows what happens.</p>

<h3>{big["model"]}</h3>
{_early_table(big, keys)}

<p>The format score collapses for both LoRA ranks &mdash; from 0.17 to about 0.02 by step
2 &mdash; while full fine-tuning's climbs. Generation length explains it: the sampler is
capped at {max_tokens} tokens, and the LoRA runs stay against that cap for a dozen steps
while full fine-tuning's outputs shorten steadily from the start. Responses that run to
the cap are truncated before they emit the answer in the expected form, so they score
zero on format and on correctness.</p>

<p>The consequence compounds. With most completions failing, whole groups score
identically, and a group whose rewards are constant carries no advantage signal and is
filtered out. Around two thirds of groups are in that state for the LoRA runs through the early
steps, against a quarter to a third for full fine-tuning. The LoRA runs were not learning more slowly
from the same data; they were learning from a fraction of it.</p>

<h3>{small["model"]}</h3>
{_early_table(small, keys)}

<p>Nothing comparable happens here. Every configuration shortens its outputs from the
first steps, format climbs monotonically, and the fraction of dead groups stays low. That
is why the pace gap is absent at this size.</p>

<div class="callout">
<p>This bounds what the sample-efficiency half of the comparison is worth. The paper's
claim covers sample efficiency, and on these settings LoRA does reach a given reward
later at {big["model"]}. But the mechanism is an interaction between the learning rate,
the token cap and early generation length &mdash; not the adapter's capacity to represent
the update. A run with a larger token budget, or with each configuration swept to its own
learning rate as the paper did, would be needed to separate the two. We did neither.</p>
</div>
"""


def scheduling_section(experiments) -> str:
  """Limitations in the placement layer that this run exposed, and what fixing them takes."""
  total = sum(len(exp["summary"]) for exp in experiments)
  largest = max(
    (_lora_count(exp["summary"]) for exp in experiments),
    default=0,
  )
  entries = limits(total, largest)
  costs = "".join(f"<li>{c}</li>" for c in sharing_costs(experiments))
  blocks = ""
  for i, (title, observed, cause, fix, _size) in enumerate(entries, 1):
    blocks += (
      f"<h3>{i}. {title}</h3>"
      f"<p><strong>Observed.</strong> {observed}</p>"
      f"<p><strong>Cause.</strong> {cause}</p>"
      f"<p><strong>Change required.</strong> {fix}</p>"
    )
  rows = "".join(f"<tr><td>{t}</td><td>{s}</td></tr>" for t, _, _, _, s in entries)
  return f"""
<h2>Scheduling limitations this run exposed</h2>
<p>The experiment was designed as a full crossing of model size, fine-tuning mode and
replicate seed, and submitted as {total} concurrent jobs. Not all of them fit the
available hardware. The ones that did not ran roughly twice as slowly as they needed to,
and were still going long after the rest had finished. The reasons are properties of the placement
layer rather than of the experiment, and they are worth separating from the scientific
caveats above, because they would apply to any workload of this shape.</p>
<p>The cost of sharing a device is visible in the per-step times:</p>
<ul>{costs}</ul>
<p>Sharing is not itself a defect &mdash; it is the reason LoRA is cheaper here, and under
real contention it is the correct choice. What follows are three places where the system
shares when it does not have to, listed in increasing order of what changing them
involves.</p>
{blocks}
<h3>Summary</h3>
<table><thead><tr><th>Limitation</th><th>Effort to address</th></tr></thead>
<tbody>{rows}</tbody></table>
<p>One theme connects them: the harness accepted a topology it could only serve badly and
gave no indication that it had done so. Nothing reported that {largest} of the {total} runs
would land on one device, either when the jobs were admitted or while they ran. A
placement summary at submission time would have surfaced this before the run rather than
after it, and is smaller work than any of the three fixes above.</p>
"""


LORA_LR, FFT_LR = "1e-4", "1e-5"
# Sparse delta payload: bf16 values plus int32 indices, verified against the
# on-disk safetensors size.
DELTA_BYTES_PER_ELEMENT = 6
DENSE_BYTES_PER_ELEMENT = 2


def density_chart(density, groups, idx_base: int = 90) -> str:
  """Delta density against step, one line per full fine-tuning run.

  Plotted on the same step axis as the reward curves so the two can be read
  together: the update starts broad and narrows as training settles.
  """
  if density is None or density.empty:
    return ""
  w, h = 720, 300
  pad_l, pad_r, pad_t, pad_b = 56, 96, 16, 40
  xmax = float(density["step"].max())
  ymax = max(1.0, float(density["density_pct"].max()) * 1.08)

  def px(s):
    return _scale(s, 0, xmax, pad_l, w - pad_r)

  def py(v):
    return _scale(v, 0, ymax, h - pad_b, pad_t)

  parts = []
  for t in range(0, int(ymax) + 2, 3):
    if t > ymax:
      break
    y = py(t)
    parts.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}"/>')
    parts.append(f'<text class="tick" x="{pad_l - 10}" y="{y + 4:.1f}" text-anchor="end">{t}%</text>')
  for t in range(0, int(xmax) + 1, 10):
    parts.append(f'<text class="tick" x="{px(t):.1f}" y="{h - pad_b + 20}" text-anchor="middle">{t}</text>')

  # One hue per model group; replicates of a group share it and split by dash.
  hues = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}
  marks = []
  labels = []
  for _run, g in density.groupby("run"):
    g = g.sort_values("step")
    grp = str(g["group"].iloc[0])
    rep = str(g["replicate"].iloc[0])
    pts = " ".join(f"{px(float(s)):.1f},{py(float(v)):.1f}" for s, v in zip(g["step"], g["density_pct"], strict=False))
    dash = _dash_attr("" if rep == "a" else "5 3")
    marks.append(f'<polyline class="line" stroke="{hues[grp][0]}"{dash} points="{pts}"/>')
    labels.append(
      {"x": px(float(g["step"].iloc[-1])) + 8, "y": py(float(g["density_pct"].iloc[-1])) + 4, "t": f"{groups[grp]} ({rep})", "c": hues[grp][0]}
    )
  labels.sort(key=lambda item: item["y"])
  for i in range(1, len(labels)):
    if labels[i]["y"] - labels[i - 1]["y"] < LABEL_GAP:
      labels[i]["y"] = labels[i - 1]["y"] + LABEL_GAP
  for item in labels:
    marks.append(f'<text class="endlabel" fill="{item["c"]}" x="{item["x"]:.1f}" y="{item["y"]:.1f}">{item["t"]}</text>')
  parts.append(reveal("".join(marks), "d0", pad_l, w - pad_l, h))
  return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Delta density against training step">{"".join(parts)}</svg>'


def _weight_sync_table(density, groups) -> str:
  rows = ""
  for grp, g in density.groupby("group"):
    per_run = []
    for _run, r in g.groupby("run"):
      r = r.sort_values("step")
      first = float(r["density_pct"].iloc[0])
      med = float(r["density_pct"].median())
      last = float(r["density_pct"].iloc[-1])
      med_changed = float(r["changed_elements"].median())
      total = float(r["total_elements"].iloc[0])
      per_run.append((first, med, last, med_changed, total))
    n = len(per_run)

    def avg(i, rows=per_run, count=n):
      return sum(p[i] for p in rows) / count

    total = per_run[0][4]
    med_bytes = avg(3) * DELTA_BYTES_PER_ELEMENT
    dense = total * DENSE_BYTES_PER_ELEMENT
    rows += (
      f"<tr><td>{groups.get(str(grp), grp)}</td>"
      f'<td class="num">{total / 1e9:.2f}B</td>'
      f'<td class="num">{avg(0):.1f}%</td>'
      f'<td class="num">{avg(1):.2f}%</td>'
      f'<td class="num">{avg(2):.2f}%</td>'
      f'<td class="num">{med_bytes / 1e9:.2f} GB</td>'
      f'<td class="num">{dense / 1e9:.1f} GB</td>'
      f'<td class="num">{dense / med_bytes:.0f}&times;</td></tr>'
    )
  return (
    "<table><thead><tr><th>Model</th><th>Parameters</th><th>Density<br>step 1</th>"
    "<th>Density<br>median</th><th>Density<br>final</th><th>Median<br>transfer</th>"
    "<th>Dense<br>equivalent</th><th>Saving</th></tr></thead>"
    f"<tbody>{rows}</tbody></table>"
  )


def weight_sync_section(experiments, density, groups) -> str:
  """What full fine-tuning actually ships to the sampler each step."""
  if density is None or density.empty:
    return ""
  timing = ""
  for exp in experiments:
    df = exp["df"]
    if "time/compute_delta_diff" not in df.columns:
      continue
    fft = df[df["config"] == "fullft"]
    diff = fft["time/compute_delta_diff"].dropna()
    total = fft["time/total"].dropna()
    if diff.empty:
      continue
    timing += (
      f"<li>{exp['model']}: {diff.mean():.2f}s to compute the delta, against "
      f"{total.mean():.0f}s for the whole step &mdash; {100 * diff.mean() / total.mean():.1f}% of it.</li>"
    )
  return f"""
<h2>What full fine-tuning ships each step</h2>
<p>Full fine-tuning keeps the trainer and the sampler in step by sending a sparse delta:
the subset of parameters that changed, as values plus indices, rather than the whole
tensor. LoRA has no equivalent, since it ships adapter weights that are small by
construction. This section is therefore about the cost full fine-tuning pays to stay
current, and it is the concrete form of the overhead LoRA avoids.</p>

{_weight_sync_table(density, groups)}

<p>The saving is real but smaller than "sparse" suggests, because the payload is
{DELTA_BYTES_PER_ELEMENT} bytes per changed element &mdash; a bf16 value and an int32
index &mdash; so indices cost twice what the values do. Halving the index width would cut
a third off the transfer. These figures were checked against the safetensors files on
disk rather than inferred from the element counts.</p>

<figure>
{density_chart(density, groups)}
<figcaption><strong>Figure 3.</strong> Fraction of parameters changed per step, both full
fine-tuning replicates at each model size. Replicates share a colour and separate by dash
pattern.</figcaption>
<button class="replay" data-keys="d0">Replay</button>
</figure>

<p>The shape is the same at both sizes: the first step rewrites more than a tenth of the
model, and within five steps the update settles to two or three percent. The two
replicates track each other closely, so this is a property of the training run rather
than of a seed. Density never reaches zero, which is why every step pays a transfer.</p>

<ul>{timing}</ul>

<p>Computing the delta is not the expensive part. The transfer and the sampler-side patch
are, and they scale with density &mdash; so the first few steps, where density is highest,
are also the slowest to synchronise.</p>
"""


def _method(experiments, all_configs) -> tuple[str, str, str]:
  """The Method paragraph, configuration table and seed caveat, sized to the data."""
  n_configs = len(all_configs)
  reps = max((int(e["configs"]["replicates"].max()) for e in experiments), default=1)
  if reps > 1:
    note = (
      f"{n_configs} configurations per experiment, each run {reps} times under a different "
      f"data-shuffling seed. Only the fine-tuning mode, the LoRA rank and the seed differ."
    )
    caveat = (
      f"Each configuration was run {reps} times. The replicate spread bounds how much two runs "
      f"differing only in seed disagree, which is what makes a gap interpretable; with {reps} "
      f"seeds it is a coarse estimate, not a confidence interval."
    )
  else:
    note = f"{n_configs} runs per experiment, differing only in the fine-tuning mode and the LoRA rank."
    caveat = (
      "One seed per configuration. The reported deviation bounds step-to-step noise within a "
      "run, not variation between seeds, so a gap of this size is not evidence in either "
      "direction."
    )
  rows = "".join(
    f"<tr><td>{config_label(c)}</td><td>{'full' if c == 'fullft' else 'lora'}</td>"
    f'<td class="num">{"&mdash;" if c == "fullft" else c.removeprefix("lora-r")}</td>'
    f'<td class="num">{FFT_LR if c == "fullft" else LORA_LR}</td>'
    f'<td class="num">{reps}</td></tr>'
    for c in all_configs
  )
  table = (
    "<table><thead><tr><th>Configuration</th><th>Mode</th><th>Rank</th>"
    f"<th>Learning rate</th><th>Seeds</th></tr></thead><tbody>{rows}</tbody></table>"
  )
  return note, table, caveat


def _episodes(exp) -> int:
  return int(exp["summary"]["steps"].min()) * 64


def _episode_note(experiments) -> str:
  """How many episodes each experiment actually saw, against the paper's scale."""
  parts = [f"the {e['model']} runs saw about {_episodes(e):,}" for e in experiments]
  return " and ".join(parts) + " each."


def _ceiling_note(experiments) -> str:
  """Where each model plateaued: the reason agreement here is cheap to obtain."""
  bits = []
  for e in experiments:
    best = float(e["configs"]["mean"].max())
    bits.append(f"{e['model']} reaches {best:.2f}")
  return "Both models largely solve GSM8K at these settings &mdash; " + ", and ".join(bits) + "."


def _placement_note(experiments) -> str:
  """What the scheduler actually did with these runs, counted from the data."""
  bits = []
  for e in experiments:
    s = e["summary"]
    n_lora = _lora_count(s)
    n_fft = int(s[s["config"] == "fullft"]["run"].nunique())
    bits.append(
      f"the {e['model']} experiment gave each of its {n_fft} full fine-tuning runs a separate "
      f"worker, and placed all {n_lora} LoRA runs on one trainer and one sampler as {n_lora} "
      f"adapters"
    )
  return (
    "In practice, " + "; ".join(bits) + ". Packing adapters is the cost argument for LoRA in "
    "this setting &mdash; several adapters hold one copy of the frozen base weights rather "
    "than one copy each &mdash; and it is also why the LoRA runs took longer per step. The "
    "full fine-tuning workers were not dedicated either: replicates of one configuration were "
    "placed two to a claim, so the per-step figures below are not single-tenant numbers for "
    "any configuration."
  )


def _lede(experiments) -> str:
  """Headline the experiment with the tightest noise floor -- the one that can decide anything."""
  scale = "at two model sizes" if len(experiments) == 2 else f"across {len(experiments)} experiments"
  intro = (
    'Thinking Machines report in <a href="https://thinkingmachines.ai/blog/lora/">LoRA '
    "Without Regret</a> that for reinforcement learning, LoRA matches full fine-tuning even "
    f"at rank&nbsp;1. We ran that comparison {scale}, with every run training at the same "
    "time on one Kubernetes cluster."
  )
  scored = [(noise_floor(e["configs"]), e) for e in experiments]
  scored = [(f, e) for f, e in scored if f == f]
  if not scored:
    return intro
  floor, best = min(scored, key=lambda t: t[0])
  cfg = best["configs"]
  # Rank 1 is the configuration the claim is about, so it leads regardless of
  # which configuration happens to sit furthest from the reference.
  target = "lora-r1" if "lora-r1" in set(cfg["config"].astype(str)) else None
  if target is None:
    return intro
  gap = float(cfg.loc[cfg["config"] == target, "gap_vs_fullft"].iloc[0])
  head = (
    f"{intro} In the {best['model']} experiment, LoRA rank&nbsp;1 finished {abs(gap):.3f} "
    f"{'above' if gap > 0 else 'below'} full fine-tuning, against a spread of {floor:.3f} "
    f"between two runs differing only in seed. That is the paper's claim reproduced: at the "
    f"lowest possible rank, the difference is smaller than the noise the measurement carries."
    if abs(gap) <= floor
    else f"{intro} In the {best['model']} experiment, LoRA rank&nbsp;1 finished {abs(gap):.3f} "
    f"{'above' if gap > 0 else 'below'} full fine-tuning, against a spread of {floor:.3f} "
    f"between two runs differing only in seed &mdash; a difference larger than the noise."
  )
  others = cfg[(cfg["config"] != "fullft") & (cfg["config"].astype(str) != target)]
  outside = [r for _, r in others.iterrows() if abs(float(r["gap_vs_fullft"])) > floor]
  if outside:
    r = max(outside, key=lambda r: abs(float(r["gap_vs_fullft"])))
    g = float(r["gap_vs_fullft"])
    head += (
      f" {config_label(str(r['config']))} sat {abs(g):.3f} "
      f"{'above' if g > 0 else 'below'} full fine-tuning, which does exceed that spread; "
      f"the direction matters, since LoRA ahead of full fine-tuning is not a failure of the "
      f"claim under test."
    )
  return head + " Repeat seeds are what make these statements possible: they measure the noise floor rather than assuming it."


def render(experiments, out: Path, smooth: int, tail: int, density=None, max_tokens: int = 512) -> None:
  """experiments: list of dicts with label, model, df, summary, configs, gaps, notes."""
  all_configs = config_order([c for exp in experiments for c in exp["df"]["config"]])
  groups = {str(e["df"]["group"].iloc[0]): e["model"] for e in experiments if e["df"]["group"].notna().any()}
  names = json.dumps({c: config_label(c) for c in all_configs})
  cfgs, sections = [], []

  for i, exp in enumerate(experiments):
    chart, cfg = reward_chart(exp["df"], smooth, idx=i)
    cfgs.append(cfg)
    tracked = [c for c in config_order(exp["df"]["config"]) if c != "fullft"]
    gap_head = "".join(f"<th>{config_label(c)}</th>" for c in tracked)
    gap_rows = "".join(
      f'<tr><td class="num">{c}</td>' + "".join(f'<td class="num">{v.get(t, 0):+.3f}</td>' for t in tracked) + "</tr>" for c, v in exp["gaps"]
    )
    gap_svg = gap_chart(exp["gaps"], noise_floor(exp["configs"]), idx=i) if exp["gaps"] else ""
    n_runs = exp["df"]["run"].nunique()
    reps = exp["summary"]["replicate"].notna().any()
    caption_extra = " Replicates of one configuration share a colour and separate by dash pattern." if reps else ""
    sections.append(f"""
<h2>{exp["label"]}</h2>
<p>{exp["intro"]}</p>

<figure>
<div class="legend">{_legend(exp["df"])}</div>
{chart}
<figcaption><strong>Figure {i + 1}.</strong> Reward against training step for {exp["model"]},
{n_runs} runs, {smooth}-step rolling mean over the raw per-step values.{caption_extra}</figcaption>
<button class="replay" data-keys="r{i}">Replay</button>
</figure>

{_summary_table(exp["configs"], tail)}

<p>{exp["note"]}</p>

<h3>Pace</h3>
<p>{exp["pace_note"]}</p>
{_pace_table(exp["df"], smooth)}

<details><summary>Individual runs</summary>{_runs_table(exp["summary"], tail)}</details>

<details><summary>Sensitivity to the tail window</summary>
<p>Reward is near its ceiling by the end of these runs, so the tail mean depends on how
many steps the window covers. Recomputing the comparison over several windows shows how
much of the verdict rests on that choice.</p>
{_sensitivity_table(exp["df"])}</details>

<details open data-reveal="g{i}"><summary>Gap to full fine-tuning at successive checkpoints</summary>
{gap_svg}
<table><thead><tr><th>Matched steps</th>{gap_head}</tr></thead>
<tbody>{gap_rows}</tbody></table></details>
""")

  data_rows = ""
  for e in experiments:
    for _, r in e["df"].iterrows():
      data_rows += (
        f"<tr><td>{e['model']}</td><td>{r['run']}</td>"
        f'<td class="num">{int(r["step"])}</td>'
        f'<td class="num">{r["reward"]:.4f}</td>'
        f'<td class="num">{r["step_seconds"]:.1f}</td></tr>'
      )
  total_rows = sum(len(e["df"]) for e in experiments)
  lede = _lede(experiments)
  method_note, config_table, seed_caveat = _method(experiments, all_configs)
  episode_note = _episode_note(experiments)
  ceiling_note = _ceiling_note(experiments)
  placement_note = _placement_note(experiments)

  doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reproducing LoRA Without Regret on GSM8K RL</title>
<style>{CSS}{color_css(all_configs)}</style></head>
<body><main>

<h1>Reproducing LoRA Without Regret</h1>
<p class="byline">Rank 1 against full fine-tuning on GSM8K RL, run concurrently on Open-RL
&middot; {date.today().isoformat()}</p>

<p class="lede">{lede}</p>

<h2>The claim</h2>
<p>A policy gradient extracts on the order of one bit per episode. Supervised learning
extracts O(tokens) per example, roughly a thousand times more per token. If that holds,
an adapter with millions of parameters has enough capacity for an RL run carrying a few
hundred thousand bits, and rank should not affect the outcome.</p>

<h2>Method</h2>
<p>{method_note} Everything else is held fixed: dataset, group size, batch, token limit,
temperature and step count. Learning rates keep the paper's ratio of about 10&times;
between LoRA and full fine-tuning &mdash; equalising them would place each configuration
at a different distance from its own optimum.</p>

{config_table}

<p>The runs execute at the same time rather than in sequence. They then share cluster state,
dataset and GPU contention, which removes the drift between separate runs. The scheduler
places LoRA runs sharing a base model on a single worker as separate adapters; full
fine-tuning takes its own workers. Comparisons are made per step, not per second, since
the shared runs progress at a fraction of the rate.</p>
{"".join(sections)}
<h2>What these runs do not show</h2>
<div class="callout">
<p>Each step covers 64 episodes, so {episode_note} The paper's MATH experiment used roughly
320,000. By its own information argument, capacity binds when episode-bits approach
adapter parameters, and these runs are two to three orders of magnitude short of that.
They show that rank&nbsp;1 trains and keeps pace; they do not test the capacity limit.</p>
<p>{ceiling_note} Once every run is at its ceiling, agreement between runs is easy to
obtain. A harder task is needed for a result that could fail.</p>
<p>{seed_caveat} The authors used Llama for GSM8K because Qwen's pretraining includes a
large amount of mathematics; these runs used Qwen.</p>
</div>

<h2>Infrastructure</h2>
<p>Each run's trainer and sampler request a GPU through a Kubernetes ResourceClaim, and
the claim's device selector decides placement. The tier comes from the model's parameter
count: a 0.6B full fine-tune fits a 24&nbsp;GB L4, while an 8B one needs an 80&nbsp;GB
H100, and an 8B LoRA worker needs one too, because the sampler must hold 16.4&nbsp;GB of
frozen weights inside the fraction of the device vLLM is given.</p>
<p>{placement_note}</p>
{early_training_section(experiments, max_tokens)}
{weight_sync_section(experiments, density, groups)}
{scheduling_section(experiments)}
<details><summary>Full data ({total_rows} rows)</summary>
<table><thead><tr><th>Model</th><th>Run</th><th>Step</th><th>Reward</th><th>Seconds</th></tr></thead>
<tbody>{data_rows}</tbody></table></details>

<footer>Generated by <code>dev/tools/rank_sweep_html.py</code> from each run's
<code>metrics.jsonl</code>. Scenario: <code>gsm8k-rl-rank-sweep</code>.</footer>
</main>
<div id="tip"></div>
<script>{JS.replace("%CHARTS%", "[" + ",".join(cfgs) + "]").replace("%NAMES%", names)}</script>
</body></html>
"""
  out.write_text(doc)
  print(f"wrote {out}  ({out.stat().st_size // 1024} KB, self-contained)")


def _intro(df, summary, configs, common: int) -> str:
  by_config = configs.set_index(configs["config"].astype(str))
  n_runs = df["run"].nunique()
  parts = [f"{n_runs} runs, {common} steps each, {common * 64:,} episodes per run."]
  if "fullft" in by_config.index:
    lora = configs[configs["config"] != "fullft"]
    solo = float(by_config.loc["fullft", "sec_per_step"])
    n_lora = int(summary[summary["config"] != "fullft"]["run"].nunique())
    if not lora.empty:
      shared = float(lora["sec_per_step"].mean())
      parts.append(f"Full fine-tuning ran at {solo:.0f}s per step; the {n_lora} LoRA runs shared one trainer and one sampler at {shared:.0f}s each.")
  return " ".join(parts)


def _note(configs) -> str:
  floor = noise_floor(configs)
  lora = configs[configs["config"] != "fullft"]
  if lora.empty or "gap_vs_fullft" not in configs:
    return ""
  worst = lora.loc[lora["gap_vs_fullft"].abs().idxmax()]
  gap = float(worst["gap_vs_fullft"])
  if floor != floor:
    return (
      f"The largest gap to full fine-tuning is {gap:+.3f}. With one seed per configuration "
      f"there is nothing to compare that against, so it is not evidence in either direction."
    )
  verdict = (
    "smaller than the distance between two runs that differ only in seed, so it is not distinguishable from noise"
    if abs(gap) <= floor
    else "larger than the distance between two runs that differ only in seed, so it is not explained by seed variation alone"
  )
  return (
    f"The largest gap to full fine-tuning is {abs(gap):.3f}, with "
    f"{config_label(str(worst['config']))} {'above' if gap > 0 else 'below'} it, against a "
    f"replicate spread of {floor:.3f}. That gap is {verdict}."
  )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
    "--run",
    action="append",
    required=True,
    metavar="LABEL=MODEL=DIR[=GROUP]",
    help="repeatable; GROUP keeps only runs from one model size when a directory holds several",
  )
  ap.add_argument("--out", type=Path, default=Path("report.html"))
  ap.add_argument("--tail", type=int, default=10)
  ap.add_argument("--smooth", type=int, default=5)
  ap.add_argument("--density", type=Path, help="CSV of per-step sparse-delta density for the full fine-tuning runs")
  ap.add_argument("--max-tokens", type=int, default=512, help="sampler token cap the runs used")
  args = ap.parse_args()
  density = pd.read_csv(args.density) if args.density else None

  experiments = []
  for spec in args.run:
    label, model, path, *rest = spec.split("=", 3)
    df = load_runs(Path(path), group=rest[0] if rest else None)
    summary, common = summarize(df, args.tail)
    configs = summarize_configs(summary)
    experiments.append(
      {
        "label": html.escape(label),
        "model": html.escape(model),
        "df": df,
        "summary": summary,
        "configs": configs,
        "gaps": build_gaps(df, args.tail),
        "intro": _intro(df, summary, configs, common),
        "pace_note": run_pace_note(df, summary),
        "note": _note(configs),
      }
    )
  render(experiments, args.out, args.smooth, args.tail, density=density, max_tokens=args.max_tokens)


if __name__ == "__main__":
  main()
