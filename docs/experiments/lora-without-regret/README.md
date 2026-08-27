# LoRA Without Regret — GSM8K RL rank sweep

Reproduction of the reinforcement-learning claim from
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): LoRA matches full
fine-tuning even at rank 1. Design and measurement plan in
[../../designs/012-lora-without-regret-reproduction.md](../../designs/012-lora-without-regret-reproduction.md).

**[report.html](report.html)** is the write-up — self-contained, open it in a
browser.

## Results

Two experiments, three runs each (full fine-tuning, LoRA rank 32, LoRA rank 1),
all training concurrently on one cluster. Mean reward over the last 10
steps:

| Model | FullFT | LoRA r32 | LoRA r1 | Largest gap | Per-run std |
|---|---|---|---|---|---|
| Qwen3-0.6B, 50 steps, L4 | 0.703 | 0.787 | 0.716 | 0.084 | 0.12–0.16 |
| Qwen3-8B, 40 steps, H100 | 0.953 | 0.948 | 0.938 | 0.015 | 0.06–0.07 |

In both experiments the runs differ by less than any one of them varies between
steps. The 8B experiment is the tighter result: a smaller spread against a lower noise floor.

**Final performance matched; pace did not.** At 8B, full fine-tuning crossed 0.9
at step 18, LoRA rank 32 at step 24 and rank 1 at step 28. Over the first 15
steps full fine-tuning averaged 0.58 against rank 1's 0.09. The paper's claim
covers sample efficiency as well as final performance, so this is a difference,
though two untested explanations would produce it: LoRA initialises B at zero so
its effective learning rate ramps up early, and we used a fixed 10× learning-rate
ratio rather than sweeping each run to its own optimum as the paper did.

## What these runs do not show

Each step covers 64 episodes, so the runs saw ~3,200 and ~2,560 episodes against
the paper's ~320,000 for MATH. Capacity binds when episode-bits approach adapter
parameters, and these are two to three orders of magnitude short, so they show
rank 1 trains and keeps pace rather than testing the capacity limit.

Both models largely solve GSM8K at these settings — 0.6B plateaus near 0.75 by
step 20, 8B reaches 0.95 — so agreement between runs is easy to obtain. A harder
task is needed for a result that could fail.

One seed per run; the reported deviation bounds within-run noise, not seed
variance. The authors used Llama for GSM8K because Qwen's pretraining is
mathematics-heavy; these runs used Qwen.

## Files

```
report.html            the write-up, self-contained
metrics-0.6b/*.jsonl   raw per-run metrics, Qwen3-0.6B run
metrics-8b/*.jsonl     raw per-run metrics, Qwen3-8B run
rank_sweep-0.6b.csv    tidy extract (run, step, reward, entropy, kl, seconds)
rank_sweep-8b.csv      same, 8B run
```

Raw metrics are kept so the analysis can be rebuilt without re-running. The
scenario deletes its log directory on start, so a later run of the same scenario
overwrites the previous one's metrics on the shared volume; these copies are the
record.

## Reproducing

```bash
# Cluster with the open-rl gateway deployed:
make cluster-e2e E2E_SCENARIO=gsm8k-rl-rank-sweep \
  E2E_ARGS="base_model=Qwen/Qwen3-8B steps=40"

# Analysis. Each --run is LABEL=MODEL=DIR, and DIR holds one subdirectory per
# run containing metrics.jsonl (the files here are flattened for readability).
cd dev/tools
uv run python rank_sweep_report.py --runs-dir <dir> --out-dir out
uv run python rank_sweep_html.py \
  --run "Qwen3-0.6B on L4=Qwen3-0.6B=<dir>" \
  --run "Qwen3-8B on H100=Qwen3-8B=<dir>" --out report.html
```

## Run conditions

- GKE, 3× L4 nodes (2 GPUs each) and 4× H100 nodes. Tier selection follows the
  model: a 0.6B full fine-tune fits a 24GB L4; an 8B one needs an 80GB H100, as
  does an 8B LoRA worker, because the sampler must hold 16.4GB of frozen weights
  inside the fraction of the device vLLM is given.
- Placement: full fine-tuning held its own trainer and sampler claims; the two
  LoRA runs shared one trainer and one sampler as separate adapters, by
  base-model affinity.
- Throughput: 0.6B 42s/step dedicated against 88s shared; 8B 89s against 160s.
  The roughly 2× is time-slicing between the two LoRA runs, not LoRA being
  slower.
