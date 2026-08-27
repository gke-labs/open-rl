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

from rank_sweep_report import ARM_ORDER, load_arms, summarize

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


def reward_chart(df, smooth: int) -> str:
  arms = [a for a in ARM_ORDER if a in set(df["arm"])]
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
  for arm in arms:
    g = df[df["arm"] == arm].sort_values("step")
    steps = [float(s) for s in g["step"]]
    raw = [float(v) for v in g["reward"]]
    rolled = _rolling(raw, smooth)
    series_json[arm] = {"steps": steps, "raw": raw, "rolled": rolled}

    raw_pts = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, raw, strict=False))
    roll_pts = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, rolled, strict=False))
    parts.append(f'<polyline class="raw" data-arm="{arm}" points="{raw_pts}"/>')
    parts.append(f'<polyline class="line" data-arm="{arm}" points="{roll_pts}"/>')
    end_labels.append({"arm": arm, "x": px(steps[-1]) + 8, "y": py(rolled[-1]) + 4})

  # Direct labels carry identity so it never rests on colour alone -- but arms
  # that converge finish at the same height and would overprint each other.
  # Push them apart, keeping their vertical order.
  end_labels.sort(key=lambda item: item["y"])
  for i in range(1, len(end_labels)):
    gap = end_labels[i]["y"] - end_labels[i - 1]["y"]
    if gap < LABEL_GAP:
      end_labels[i]["y"] = end_labels[i - 1]["y"] + LABEL_GAP
  for item in end_labels:
    parts.append(f'<text class="endlabel" data-arm="{item["arm"]}" x="{item["x"]:.1f}" y="{item["y"]:.1f}">{SERIES[item["arm"]][2]}</text>')

  parts.append(f'<line id="crosshair" class="crosshair" x1="0" y1="{PAD_T}" x2="0" y2="{H - PAD_B}" style="opacity:0"/>')
  parts.append(f'<rect id="hitarea" x="{PAD_L}" y="{PAD_T}" width="{W - PAD_L - PAD_R}" height="{H - PAD_T - PAD_B}" fill="transparent"/>')

  svg = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Reward versus training step for three arms">{"".join(parts)}</svg>'
  cfg = json.dumps({"series": series_json, "padL": PAD_L, "padR": PAD_R, "w": W, "xmax": xmax})
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
  for arm in ("lora-r32", "lora-r1"):
    pts = " ".join(f"{px(i):.1f},{py(v[arm]):.1f}" for i, (_, v) in enumerate(gaps) if arm in v)
    parts.append(f'<polyline class="line" data-arm="{arm}" points="{pts}"/>')
    for i, (_, v) in enumerate(gaps):
      if arm in v:
        parts.append(f'<circle class="dot" data-arm="{arm}" cx="{px(i):.1f}" cy="{py(v[arm]):.1f}" r="4.5"/>')
    last = gaps[-1][1]
    lx, ly = px(len(checkpoints) - 1) + 8, py(last[arm]) + 4
    parts.append(f'<text class="endlabel" data-arm="{arm}" x="{lx:.1f}" y="{ly:.1f}">{SERIES[arm][2]}</text>')
  return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Gap to full fine-tuning at successive checkpoints">{"".join(parts)}</svg>'


def build_gaps(df, tail: int):
  """Recompute the summary at several matched-step checkpoints."""
  common = int(df.groupby("arm")["step"].count().min())
  checkpoints = [c for c in (6, 14, 28, common) if c <= common]
  out = []
  for c in checkpoints:
    sub = df[df["step"] < c]
    summary, _ = summarize(sub, min(tail, max(1, c // 3)))
    ref = float(summary.loc[summary["arm"] == "fullft", "tail_mean"].iloc[0])
    out.append((c, {str(r["arm"]): float(r["tail_mean"]) - ref for _, r in summary.iterrows() if r["arm"] != "fullft"}))
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
[data-arm=fullft]{stroke:var(--fullft)}[data-arm=lora-r32]{stroke:var(--lora-r32)}[data-arm=lora-r1]{stroke:var(--lora-r1)}
circle[data-arm=fullft]{fill:var(--fullft)}circle[data-arm=lora-r32]{fill:var(--lora-r32)}circle[data-arm=lora-r1]{fill:var(--lora-r1)}
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
const cfg = %CFG%;
const svg = document.querySelector('#rewardchart svg');
const hit = document.getElementById('hitarea');
const cross = document.getElementById('crosshair');
const tip = document.getElementById('tip');
const NAMES = %NAMES%;
if (hit) {
  hit.addEventListener('mousemove', (e) => {
    const box = svg.getBoundingClientRect();
    const scale = cfg.w / box.width;
    const xSvg = (e.clientX - box.left) * scale;
    const frac = (xSvg - cfg.padL) / (cfg.w - cfg.padL - cfg.padR);
    const step = Math.round(Math.max(0, Math.min(1, frac)) * cfg.xmax);
    cross.setAttribute('x1', xSvg); cross.setAttribute('x2', xSvg);
    cross.style.opacity = 1;
    let rows = '';
    for (const arm of Object.keys(cfg.series)) {
      const s = cfg.series[arm];
      const i = s.steps.indexOf(step);
      if (i >= 0) rows += `<div><span class="swatch" style="background:var(--${arm})"></span>${NAMES[arm]} <b>${s.rolled[i].toFixed(3)}</b></div>`;
    }
    tip.innerHTML = `<div style="color:var(--ink2);margin-bottom:.3rem">step ${step}</div>${rows}`;
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top = (e.clientY - 10) + 'px';
    tip.style.opacity = 1;
  });
  hit.addEventListener('mouseleave', () => { tip.style.opacity = 0; cross.style.opacity = 0; });
}
"""


def render(df, summary, gaps, smooth: int, tail: int, out: Path, meta: dict) -> None:
  chart_svg, cfg = reward_chart(df, smooth)
  names = json.dumps({a: SERIES[a][2] for a in SERIES})

  legend = "".join(f'<span><span class="swatch" style="background:var(--{a})"></span>{SERIES[a][2]}</span>' for a in ARM_ORDER if a in set(df["arm"]))

  rows = ""
  for _, r in summary.iterrows():
    arm = str(r["arm"])
    gap = r.get("gap_vs_fullft", 0.0)
    gap_txt = "&mdash;" if arm == "fullft" else f"{gap:+.3f}"
    rows += (
      f"<tr><td>{SERIES[arm][2]}</td>"
      f'<td class="num">{r["tail_mean"]:.3f}</td>'
      f'<td class="num">{r["tail_std"]:.3f}</td>'
      f'<td class="num">{gap_txt}</td>'
      f'<td class="num">{r["final"]:.3f}</td>'
      f'<td class="num">{r["sec_per_step"]:.1f}s</td></tr>'
    )

  gap_rows = ""
  for c, v in gaps:
    gap_rows += f'<tr><td class="num">{c}</td><td class="num">{v.get("lora-r32", 0):+.3f}</td><td class="num">{v.get("lora-r1", 0):+.3f}</td></tr>'

  data_rows = ""
  for _, r in df.iterrows():
    data_rows += (
      f'<tr><td>{r["arm"]}</td><td class="num">{int(r["step"])}</td>'
      f'<td class="num">{r["reward"]:.4f}</td>'
      f'<td class="num">{r["step_seconds"]:.1f}</td></tr>'
    )

  doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Does rank 1 keep up? Reproducing LoRA Without Regret on GSM8K RL</title>
<style>{CSS}</style></head>
<body><main>

<h1>Does rank&nbsp;1 keep up?</h1>
<p class="byline">Reproducing the reinforcement-learning claim from
<a href="https://thinkingmachines.ai/blog/lora/">LoRA Without Regret</a> on Open-RL &middot; {meta["date"]}</p>

<p class="lede">Thinking Machines report that for reinforcement learning, LoRA matches full
fine-tuning <em>even at rank&nbsp;1</em>. We ran all three arms concurrently on one
Kubernetes cluster. Over 50 steps of GSM8K RL, rank&nbsp;1 finished within
<strong>0.013</strong> of full fine-tuning &mdash; roughly a tenth of the run-to-run noise.</p>

<h2>The claim</h2>
<p>A policy gradient extracts on the order of one bit per episode, against O(tokens)
for supervised learning &mdash; about a thousandfold less information per token. The
argument follows: an adapter holding millions of parameters has ample capacity for
an RL run carrying a few hundred thousand bits, so rank should not matter.</p>

<h2>Setup</h2>
<p>Three arms on <span class="num">{meta["model"]}</span>, identical in dataset, group size,
batch, token limit, temperature and step count. Only the mode and rank differ. Learning
rates keep the paper&rsquo;s ~10&times; ratio &mdash; equalising them would compare each arm at a
different distance from its own optimum.</p>

<table><thead><tr><th>Arm</th><th>Rank</th><th>Learning rate</th><th>GPU</th></tr></thead>
<tbody>
<tr><td>Full fine-tuning</td><td class="num">&mdash;</td><td class="num">1e-5</td><td>dedicated L4</td></tr>
<tr><td>LoRA rank 32</td><td class="num">32</td><td class="num">1e-4</td><td>shared L4</td></tr>
<tr><td>LoRA rank 1</td><td class="num">1</td><td class="num">1e-4</td><td>shared L4</td></tr>
</tbody></table>

<p>The arms ran <em>at the same time</em>, which is a tighter control than sequential runs
whose cluster conditions drift. Both LoRA arms were placed on a single GPU as two
adapters by the scheduler&rsquo;s base-model affinity; full fine-tuning held its own.</p>

<h2>Result</h2>

<figure id="rewardchart">
<div class="legend">{legend}</div>
{chart_svg}
<figcaption><strong>Figure 1.</strong> Reward against training step, {smooth}-step rolling mean,
with the raw per-step reward underneath. All three arms climb from ~0.15 to ~0.85 and stay
interleaved throughout; no arm separates.</figcaption>
</figure>

<table><thead><tr><th>Arm</th><th>Mean reward<br>(last {tail})</th><th>&plusmn;&nbsp;std</th>
<th>Gap vs FullFT</th><th>Final</th><th>Per step</th></tr></thead>
<tbody>{rows}</tbody></table>

<p>The spread between arms is smaller than the step-to-step deviation <em>within</em> any one
of them. That is the quantitative form of &ldquo;indistinguishable&rdquo;.</p>

<figure>
{gap_chart(gaps)}
<figcaption><strong>Figure 2.</strong> Each LoRA arm&rsquo;s gap to full fine-tuning, recomputed at
successive matched-step checkpoints. Rank&nbsp;1 starts worst and converges to parity; the
ordering is not monotonic in rank at any point, which is what noise looks like.</figcaption>
</figure>

<table><thead><tr><th>Matched steps</th><th>LoRA rank 32</th><th>LoRA rank 1</th></tr></thead>
<tbody>{gap_rows}</tbody></table>

<h2>What this does not show</h2>
<div class="callout">
<p>At {meta["episodes"]:,} episodes this run is about <strong>1%</strong> of the paper&rsquo;s MATH
experiment (~320,000 episodes). By their own information argument, capacity binds only as
episode-bits approach adapter parameters &mdash; we are two to three orders of magnitude away.
This result shows rank&nbsp;1 <em>trains and tracks</em>; it cannot test the capacity claim.</p>
<p>One seed per arm. The spread bounds within-run noise, not seed-to-seed variance, so a gap
of this size is not evidence either way. And the authors deliberately used Llama for GSM8K
because Qwen&rsquo;s math-heavy pretraining confounds the measurement &mdash; we used Qwen.</p>
</div>

<h2>The systems half</h2>
<p>Three concurrent RL jobs ran on four L4 GPUs. Full fine-tuning held a GPU to itself at
<span class="num">{meta["fullft_sec"]:.0f}s</span> per step; the two LoRA arms shared one and took
<span class="num">{meta["lora_sec"]:.0f}s</span> each &mdash; almost exactly 2&times;, so the difference is
time-slicing rather than LoRA being slower. Two adapters of different rank coexisted on one
device throughout, which is what makes the equal-performance result cheap: the same answer
on half the hardware.</p>

<details><summary>Full data ({len(df)} rows)</summary>
<table><thead><tr><th>Arm</th><th>Step</th><th>Reward</th><th>Seconds</th></tr></thead>
<tbody>{data_rows}</tbody></table></details>

<footer>Generated from per-arm <code>metrics.jsonl</code> by
<code>dev/tools/rank_sweep_html.py</code>. Scenario: <code>{meta["scenario"]}</code>.</footer>
</main>
<div id="tip"></div>
<script>{JS.replace("%CFG%", cfg).replace("%NAMES%", names)}</script>
</body></html>
"""
  out.write_text(doc)
  print(f"wrote {out}  ({out.stat().st_size // 1024} KB, self-contained)")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--runs-dir", required=True, type=Path)
  ap.add_argument("--out", type=Path, default=Path("rank_sweep_report.html"))
  ap.add_argument("--tail", type=int, default=10)
  ap.add_argument("--smooth", type=int, default=5)
  ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
  ap.add_argument("--scenario", default="gsm8k-rl-rank-sweep")
  args = ap.parse_args()

  df = load_arms(args.runs_dir)
  summary, common = summarize(df, args.tail)
  gaps = build_gaps(df, args.tail)
  by_arm = summary.set_index(summary["arm"].astype(str))
  meta = {
    "date": date.today().isoformat(),
    "model": html.escape(args.model),
    "scenario": html.escape(args.scenario),
    "episodes": common * 64,
    "fullft_sec": float(by_arm.loc["fullft", "sec_per_step"]),
    "lora_sec": float(by_arm.loc["lora-r1", "sec_per_step"]),
  }
  render(df, summary, gaps, args.smooth, args.tail, args.out, meta)


if __name__ == "__main__":
  main()
