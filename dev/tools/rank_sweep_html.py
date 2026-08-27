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
from datetime import date
from pathlib import Path

from rank_sweep_report import RUN_ORDER, load_runs, summarize

# Validated categorical slots 1-3 (light / dark), from the design-system palette.
SERIES = {
  "fullft": ("#2a78d6", "#3987e5", "Full fine-tuning"),
  "lora-r32": ("#eb6834", "#d95926", "LoRA rank 32"),
  "lora-r1": ("#1baf7a", "#199e70", "LoRA rank 1"),
}

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


def reward_chart(df, smooth: int, idx: int = 0) -> str:
  run_names = [a for a in RUN_ORDER if a in set(df["run"])]
  xmax = float(df["step"].max())
  ymin, ymax = 0.0, max(1.0, float(df["reward"].max()) * 1.05)

  def px(step):
    return _scale(step, 0, xmax, PAD_L, W - PAD_R)

  def py(reward):
    return _scale(reward, ymin, ymax, H - PAD_B, PAD_T)

  parts = []
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
  for name in run_names:
    g = df[df["run"] == name].sort_values("step")
    steps = [float(s) for s in g["step"]]
    raw = [float(v) for v in g["reward"]]
    rolled = _rolling(raw, smooth)
    series_json[name] = {"steps": steps, "raw": raw, "rolled": rolled}

    raw_pts = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, raw, strict=False))
    roll_pts = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, rolled, strict=False))
    parts.append(f'<polyline class="raw" data-run="{name}" points="{raw_pts}"/>')
    parts.append(f'<polyline class="line" data-run="{name}" points="{roll_pts}"/>')
    end_labels.append({"run": name, "x": px(steps[-1]) + 8, "y": py(rolled[-1]) + 4})

  # Direct labels carry identity so it never rests on colour alone -- but runs
  # that converge finish at the same height and would overprint each other.
  # Push them apart, keeping their vertical order.
  end_labels.sort(key=lambda item: item["y"])
  for i in range(1, len(end_labels)):
    gap = end_labels[i]["y"] - end_labels[i - 1]["y"]
    if gap < LABEL_GAP:
      end_labels[i]["y"] = end_labels[i - 1]["y"] + LABEL_GAP
  for item in end_labels:
    parts.append(f'<text class="endlabel" data-run="{item["run"]}" x="{item["x"]:.1f}" y="{item["y"]:.1f}">{SERIES[item["run"]][2]}</text>')

  parts.append(f'<line id="crosshair-{idx}" class="crosshair" x1="0" y1="{PAD_T}" x2="0" y2="{H - PAD_B}" style="opacity:0"/>')
  parts.append(f'<rect id="hitarea-{idx}" x="{PAD_L}" y="{PAD_T}" width="{W - PAD_L - PAD_R}" height="{H - PAD_T - PAD_B}" fill="transparent"/>')

  svg = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Reward versus training step for three runs">{"".join(parts)}</svg>'
  cfg = json.dumps({"idx": idx, "series": series_json, "padL": PAD_L, "padR": PAD_R, "w": W, "xmax": xmax})
  return svg, cfg


def gap_chart(gaps) -> str:
  """Gap to FullFT at successive matched-step checkpoints, with a zero line."""
  checkpoints = [c for c, _ in gaps]
  w, h = 720, 260
  pad_l, pad_r, pad_t, pad_b = 56, 92, 16, 38
  lo = min(min(v.values()) for _, v in gaps) - 0.03
  hi = max(max(v.values()) for _, v in gaps) + 0.03

  def px(i):
    return _scale(i, 0, max(1, len(checkpoints) - 1), pad_l, w - pad_r)

  def py(v):
    return _scale(v, lo, hi, h - pad_b, pad_t)

  parts = [f'<line class="zero" x1="{pad_l}" y1="{py(0):.1f}" x2="{w - pad_r}" y2="{py(0):.1f}"/>']
  parts.append(f'<text class="tick" x="{w - pad_r + 8}" y="{py(0) + 4:.1f}">parity</text>')
  for i, c in enumerate(checkpoints):
    parts.append(f'<text class="tick" x="{px(i):.1f}" y="{h - pad_b + 20}" text-anchor="middle">{c}</text>')
  for name in ("lora-r32", "lora-r1"):
    pts = " ".join(f"{px(i):.1f},{py(v[name]):.1f}" for i, (_, v) in enumerate(gaps) if name in v)
    parts.append(f'<polyline class="line" data-run="{name}" points="{pts}"/>')
    for i, (_, v) in enumerate(gaps):
      if name in v:
        parts.append(f'<circle class="dot" data-run="{name}" cx="{px(i):.1f}" cy="{py(v[name]):.1f}" r="4.5"/>')
    last = gaps[-1][1]
    lx, ly = px(len(checkpoints) - 1) + 8, py(last[name]) + 4
    parts.append(f'<text class="endlabel" data-run="{name}" x="{lx:.1f}" y="{ly:.1f}">{SERIES[name][2]}</text>')
  return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Gap to full fine-tuning at successive checkpoints">{"".join(parts)}</svg>'


def build_gaps(df, tail: int):
  """Recompute the summary at several matched-step checkpoints."""
  common = int(df.groupby("run")["step"].count().min())
  checkpoints = [c for c in (6, 14, 28, common) if c <= common]
  out = []
  for c in checkpoints:
    sub = df[df["step"] < c]
    summary, _ = summarize(sub, min(tail, max(1, c // 3)))
    ref = float(summary.loc[summary["run"] == "fullft", "tail_mean"].iloc[0])
    out.append((c, {str(r["run"]): float(r["tail_mean"]) - ref for _, r in summary.iterrows() if r["run"] != "fullft"}))
  return out


CSS = """
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e6e5e0;
--fullft:#2a78d6;--lora-r32:#eb6834;--lora-r1:#1baf7a}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--rule:#33322f;
--fullft:#3987e5;--lora-r32:#d95926;--lora-r1:#199e70}}
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
svg{width:100%;height:auto;display:block;overflow:visible}
.grid{stroke:var(--rule);stroke-width:1}
.zero{stroke:var(--ink2);stroke-width:1;stroke-dasharray:4 4;opacity:.6}
.tick{fill:var(--ink2);font:11px ui-sans-serif,system-ui,sans-serif}
.raw{fill:none;stroke-width:1;opacity:.22}
.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.endlabel{font:12px ui-sans-serif,system-ui,sans-serif;fill:var(--ink2)}
.crosshair{stroke:var(--ink2);stroke-width:1;stroke-dasharray:3 3}
[data-run=fullft]{stroke:var(--fullft)}[data-run=lora-r32]{stroke:var(--lora-r32)}[data-run=lora-r1]{stroke:var(--lora-r1)}
circle[data-run=fullft]{fill:var(--fullft)}circle[data-run=lora-r32]{fill:var(--lora-r32)}circle[data-run=lora-r1]{fill:var(--lora-r1)}
text.endlabel{stroke:none}
.legend{display:flex;gap:1.4rem;flex-wrap:wrap;font:13px ui-sans-serif,system-ui,sans-serif;color:var(--ink2);margin:.2rem 0 .4rem}
.swatch{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:.45rem;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font:14px ui-sans-serif,system-ui,sans-serif;margin:1.2rem 0}
th,td{text-align:right;padding:.5rem .6rem;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
tbody tr:last-child td{border-bottom:none}
.num{font-variant-numeric:tabular-nums}
.callout{border-left:2px solid var(--rule);padding:.1rem 0 .1rem 1.1rem;color:var(--ink2);margin:1.6rem 0}
details{margin:1.2rem 0;font:14px ui-sans-serif,system-ui,sans-serif;color:var(--ink2)}
summary{cursor:pointer}
#tip{position:fixed;pointer-events:none;opacity:0;background:var(--surface);border:1px solid var(--rule);
border-radius:6px;padding:.5rem .65rem;font:12px ui-sans-serif,system-ui,sans-serif;color:var(--ink);
box-shadow:0 2px 10px rgba(0,0,0,.08);transition:opacity .1s}
footer{margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--rule);color:var(--ink2);font-size:.85rem}
"""

JS = """
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
  """First step at which each run's rolling mean crosses each threshold."""
  out = []
  for name in [a for a in RUN_ORDER if a in set(df["run"])]:
    g = df[df["run"] == name].sort_values("step")
    rolled = _rolling([float(v) for v in g["reward"]], smooth)
    row = {"run": name, "early": sum(rolled[:15]) / min(15, len(rolled))}
    for th in (0.5, 0.8, 0.9):
      hit = next((i for i, v in enumerate(rolled) if v >= th), None)
      row[th] = hit
    out.append(row)
  return out


def _pace_table(df, smooth: int) -> str:
  rows = ""
  for r in steps_to_thresholds(df, smooth):
    cells = "".join(f'<td class="num">{"&mdash;" if r[th] is None else r[th]}</td>' for th in (0.5, 0.8, 0.9))
    rows += f'<tr><td>{SERIES[r["run"]][2]}</td><td class="num">{r["early"]:.3f}</td>{cells}</tr>'
  return (
    "<table><thead><tr><th>Run</th><th>Mean reward<br>steps 0-14</th>"
    "<th>Step to 0.5</th><th>Step to 0.8</th><th>Step to 0.9</th></tr></thead>"
    f"<tbody>{rows}</tbody></table>"
  )


def _summary_table(summary, tail: int) -> str:
  rows = ""
  for _, r in summary.iterrows():
    name = str(r["run"])
    gap = r.get("gap_vs_fullft", 0.0)
    gap_txt = "&mdash;" if name == "fullft" else f"{gap:+.4f}"
    rows += (
      f"<tr><td>{SERIES[name][2]}</td>"
      f'<td class="num">{r["tail_mean"]:.3f}</td>'
      f'<td class="num">{r["tail_std"]:.3f}</td>'
      f'<td class="num">{gap_txt}</td>'
      f'<td class="num">{r["final"]:.3f}</td>'
      f'<td class="num">{r["sec_per_step"]:.0f}s</td></tr>'
    )
  return (
    f"<table><thead><tr><th>Run</th><th>Mean reward<br>(last {tail})</th><th>&plusmn;&nbsp;std</th>"
    f"<th>Gap vs FullFT</th><th>Final</th><th>Per step</th></tr></thead>"
    f"<tbody>{rows}</tbody></table>"
  )


def _legend(df) -> str:
  return "".join(f'<span><span class="swatch" style="background:var(--{a})"></span>{SERIES[a][2]}</span>' for a in RUN_ORDER if a in set(df["run"]))


def run_pace_note(df, summary) -> str:
  """Describe how quickly each run got there, which the endpoint table hides."""
  pace = {r["run"]: r for r in steps_to_thresholds(df, 5)}
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


def render(experiments, out: Path, smooth: int, tail: int) -> None:
  """experiments: list of dicts with label, model, df, summary, gaps, notes."""
  names = json.dumps({a: SERIES[a][2] for a in SERIES})
  cfgs, sections = [], []

  for i, exp in enumerate(experiments):
    chart, cfg = reward_chart(exp["df"], smooth, idx=i)
    cfgs.append(cfg)
    gap_rows = "".join(
      f'<tr><td class="num">{c}</td><td class="num">{v.get("lora-r32", 0):+.3f}</td><td class="num">{v.get("lora-r1", 0):+.3f}</td></tr>'
      for c, v in exp["gaps"]
    )
    sections.append(f"""
<h2>{exp["label"]}</h2>
<p>{exp["intro"]}</p>

<figure>
<div class="legend">{_legend(exp["df"])}</div>
{chart}
<figcaption><strong>Figure {i + 1}.</strong> Reward against training step for {exp["model"]},
{smooth}-step rolling mean over the raw per-step values.</figcaption>
</figure>

{_summary_table(exp["summary"], tail)}

<p>{exp["note"]}</p>

<h3>Pace</h3>
<p>{exp["pace_note"]}</p>
{_pace_table(exp["df"], smooth)}

<details><summary>Gap to full fine-tuning at successive checkpoints</summary>
<table><thead><tr><th>Matched steps</th><th>LoRA rank 32</th><th>LoRA rank 1</th></tr></thead>
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

  doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reproducing LoRA Without Regret on GSM8K RL</title>
<style>{CSS}</style></head>
<body><main>

<h1>Reproducing LoRA Without Regret</h1>
<p class="byline">Rank 1 against full fine-tuning on GSM8K RL, run concurrently on Open-RL
&middot; {date.today().isoformat()}</p>

<p class="lede">Thinking Machines report in
<a href="https://thinkingmachines.ai/blog/lora/">LoRA Without Regret</a> that for
reinforcement learning, LoRA matches full fine-tuning even at rank&nbsp;1. We ran that
comparison twice, at 0.6B and 8B parameters, with all runs training at the same time on
one Kubernetes cluster. At 8B, rank&nbsp;1 finished within 0.015 of full fine-tuning,
against a per-run step-to-step deviation of 0.06. It took longer to get there: full
fine-tuning passed 0.9 at step 18 and rank&nbsp;1 at step 28. Final performance matched;
sample efficiency, on these settings, did not.</p>

<h2>The claim</h2>
<p>A policy gradient extracts on the order of one bit per episode. Supervised learning
extracts O(tokens) per example, roughly a thousand times more per token. If that holds,
an adapter with millions of parameters has enough capacity for an RL run carrying a few
hundred thousand bits, and rank should not affect the outcome.</p>

<h2>Method</h2>
<p>Three runs per experiment, identical in dataset, group size, batch, token limit,
temperature, step count and seed. Only the fine-tuning mode and the LoRA rank differ.
Learning rates keep the paper's ratio of about 10&times; between LoRA and full
fine-tuning: equalising them would place each run at a different distance from its own
optimum.</p>

<table><thead><tr><th>Run</th><th>Mode</th><th>Rank</th><th>Learning rate</th></tr></thead>
<tbody>
<tr><td>Full fine-tuning</td><td>full</td><td class="num">&mdash;</td><td class="num">1e-5</td></tr>
<tr><td>LoRA rank 32</td><td>lora</td><td class="num">32</td><td class="num">1e-4</td></tr>
<tr><td>LoRA rank 1</td><td>lora</td><td class="num">1</td><td class="num">1e-4</td></tr>
</tbody></table>

<p>The runs execute at the same time rather than in sequence. They then share cluster state,
dataset and GPU contention, which removes the drift between separate runs. The scheduler
places the two LoRA runs on a single worker as two adapters, because they share a base
model; full fine-tuning takes its own workers. Comparisons are made per step, not per
second, since the shared runs progress at about half the rate.</p>
{"".join(sections)}
<h2>What these runs do not show</h2>
<div class="callout">
<p>Each step covers 64 episodes, so the 0.6B run saw about 3,200 and the 8B run about
2,560. The paper's MATH experiment used roughly 320,000. By its own information argument,
capacity binds when episode-bits approach adapter parameters, and these runs are two to
three orders of magnitude short of that. They show that rank&nbsp;1 trains and keeps pace;
they do not test the capacity limit.</p>
<p>Both models largely solve GSM8K at these settings &mdash; 0.6B plateaus near 0.75 by
step 20, and 8B reaches 0.95. Once every run is at its ceiling, agreement between runs is
easy to obtain. A harder task is needed for a result that could fail.</p>
<p>One seed per run. The reported deviation bounds step-to-step noise within a run, not
variation between seeds, so a gap of this size is not evidence in either direction.
The authors used Llama for GSM8K because Qwen's pretraining includes a large amount of
mathematics; these runs used Qwen.</p>
</div>

<h2>Infrastructure</h2>
<p>Each run's trainer and sampler request a GPU through a Kubernetes ResourceClaim, and
the claim's device selector decides placement. The tier comes from the model's parameter
count: a 0.6B full fine-tune fits a 24&nbsp;GB L4, while an 8B one needs an 80&nbsp;GB
H100, and an 8B LoRA worker needs one too, because the sampler must hold 16.4&nbsp;GB of
frozen weights inside the fraction of the device vLLM is given.</p>
<p>The 0.6B experiment therefore ran three runs on four L4s, and the 8B experiment ran
them on four H100s. In both cases the two LoRA runs shared one trainer and one sampler,
holding two adapters of different rank on the same device. That is the cost argument for
LoRA in this setting: the same result on half the hardware.</p>

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


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--run", action="append", required=True, metavar="LABEL=MODEL=DIR", help="repeatable")
  ap.add_argument("--out", type=Path, default=Path("report.html"))
  ap.add_argument("--tail", type=int, default=10)
  ap.add_argument("--smooth", type=int, default=5)
  args = ap.parse_args()

  experiments = []
  for spec in args.run:
    label, model, path = spec.split("=", 2)
    df = load_runs(Path(path))
    summary, common = summarize(df, args.tail)
    by_run = summary.set_index(summary["run"].astype(str))
    gaps = build_gaps(df, args.tail)
    spread = max(abs(float(by_run.loc[a, "gap_vs_fullft"])) for a in ("lora-r32", "lora-r1"))
    worst_std = float(summary["tail_std"].max())
    experiments.append(
      {
        "label": html.escape(label),
        "model": html.escape(model),
        "df": df,
        "summary": summary,
        "gaps": gaps,
        "intro": (
          f"{common} steps per run, {common * 64:,} episodes. Full fine-tuning ran on its own "
          f"GPU at {float(by_run.loc['fullft', 'sec_per_step']):.0f}s per step; the two LoRA runs "
          f"shared one at {float(by_run.loc['lora-r1', 'sec_per_step']):.0f}s each."
        ),
        "pace_note": run_pace_note(df, summary),
        "note": (
          f"The largest gap between a run and full fine-tuning is {spread:.3f}, against a "
          f"within-run step-to-step deviation of up to {worst_std:.3f}. The runs differ by less "
          f"than any one of them varies between steps."
        ),
      }
    )
  render(experiments, args.out, args.smooth, args.tail)


if __name__ == "__main__":
  main()
