# LoRA Without Regret — GSM8K RL rank sweep

Reproduction of the reinforcement-learning claim from
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): LoRA matches full
fine-tuning even at rank 1. Design and measurement plan in
[../../designs/012-lora-without-regret-reproduction.md](../../designs/012-lora-without-regret-reproduction.md).

**[report.html](report.html)** is the published write-up — self-contained, open it
in a browser.

## Result

Three arms, same base model, run concurrently on one cluster. Over 50 steps,
mean reward across the last 10:

| Arm | Mean reward | ± std | Gap vs FullFT |
|---|---|---|---|
| Full fine-tuning | 0.703 | 0.160 | — |
| LoRA rank 32 | 0.788 | 0.123 | +0.084 |
| LoRA rank 1 | 0.716 | 0.135 | **+0.013** |

The spread between arms is smaller than the step-to-step deviation within any one
of them, and rank 1's gap converges toward zero as steps accumulate
(−0.112 → −0.060 → +0.071 → +0.013 at 6/14/28/50 matched steps). Consistent with
the paper.

**What it does not show.** ~3,200 episodes is about 1% of the paper's MATH run, so
this cannot test their capacity argument — it shows rank 1 trains and tracks. One
seed per arm; the spread bounds within-run noise, not seed variance. The authors
used Llama for GSM8K because Qwen's math pretraining confounds it, and this used
Qwen.

## Reproducing

```bash
# On a cluster with the open-rl gateway deployed:
make cluster-e2e E2E_SCENARIO=gsm8k-rl-rank-sweep \
  E2E_ARGS="base_model=Qwen/Qwen3-0.6B steps=50"

# Regenerate the analysis from this directory's captured metrics:
cd dev/tools
uv run python rank_sweep_report.py --runs-dir <dir-of-per-arm-metrics> --out-dir out
uv run python rank_sweep_html.py   --runs-dir <dir-of-per-arm-metrics> --out report.html
```

`metrics/` holds each arm's raw `metrics.jsonl` as captured from the shared volume,
so the report can be rebuilt without re-running the experiment. `rank_sweep.csv` is
the tidy extract (arm, step, reward, entropy, kl, seconds).

The analysis tools expect one subdirectory per arm, each containing a
`metrics.jsonl`; the files here are flattened to `<arm>.jsonl` for readability.

## Run conditions

- Cluster: GKE, 3× L4 nodes (2 GPUs each) + 2× H100 nodes; all three arms landed on
  L4s, since size-aware tiering puts a 0.6B full fine-tune on the 24gb tier.
- Placement: full fine-tuning held its own trainer and sampler claims; both LoRA
  arms were co-located on one shared worker pair as separate adapters, by
  base-model affinity.
- Throughput: 42s/step for the dedicated FullFT arm, 87.9s/step for each LoRA arm
  sharing a GPU — the ~2× is time-slicing, not LoRA being slower.
