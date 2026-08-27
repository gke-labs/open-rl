# LoRA Without Regret — GSM8K RL rank sweep

Reproduction of the reinforcement-learning claim from
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): LoRA matches full
fine-tuning even at rank 1. Design and measurement plan in
[../../designs/012-lora-without-regret-reproduction.md](../../designs/012-lora-without-regret-reproduction.md).

**[report.html](report.html)** is the write-up — self-contained, open it in a
browser.

## Results

Twelve runs: two model sizes × three configurations (full fine-tuning, LoRA
rank 32, LoRA rank 1) × two seeds, all training concurrently on one cluster.
Running each configuration twice is what makes the comparison readable — the
spread between two runs that differ only in their data-shuffling seed is the
reference any gap has to beat.

Mean reward over the last 10 of 40 steps:

| Model | FullFT | LoRA r32 | LoRA r1 | Replicate spread |
|---|---|---|---|---|
| Qwen3-0.6B, L4 | 0.697 | 0.694 | 0.718 | 0.033–0.042 |
| Qwen3-8B, H100 | 0.934 | 0.957 | 0.936 | 0.013–0.016 |

**At 8B, rank 1 matched full fine-tuning.** The gap is +0.002 against a
replicate spread of 0.016 — far inside the noise. That is the paper's claim,
reproduced at the lowest possible rank.

Two things complicate a clean headline, and the report states both:

- **Rank 32 finished 0.023 above full fine-tuning**, which exceeds the spread.
  LoRA ahead of full fine-tuning is not a failure of the claim under test, but
  it does mean the two are not indistinguishable at this rank.
- **The verdict moves with the tail window.** At 8B the rank-32 excess is
  significant at windows of 5, 10 and 15 steps and not at 20. Reward is at a
  ceiling by then, so the tail mean is unstable. The report shows the full
  sensitivity table rather than one window's answer.

**The 0.6B half decides nothing.** Its replicate spread (0.033–0.042) is wider
than any gap being measured, so "within noise" there describes the noise, not
the ranks. It is included because it shows what an underpowered version of this
experiment looks like.

**Pace differed from final performance.** At 8B, full fine-tuning crossed 0.9 at
step 19, rank 32 at 25 and rank 1 at 27. Over the first 15 steps full
fine-tuning averaged 0.48 against rank 1's 0.15. The paper's claim covers sample
efficiency as well as final performance, so this is a real difference; two
untested explanations would produce it, LoRA initialising B at zero so its
effective learning rate ramps up, and our use of a fixed 10× learning-rate ratio
rather than sweeping each configuration to its own optimum.

## What these runs do not show

Each step covers 64 episodes, so each run saw about 2,560 episodes against the
paper's ~320,000 for MATH. Capacity binds when episode-bits approach adapter
parameters, and these are two to three orders of magnitude short, so they show
rank 1 trains and keeps pace rather than testing the capacity limit.

Both models largely solve GSM8K at these settings. Once every run is at its
ceiling, agreement is easy to obtain and the tail mean gets noisy — which is
what the sensitivity table is about. A harder task is needed for a result that
could fail cleanly.

Two seeds bound the noise floor coarsely; that is an estimate, not a confidence
interval. The authors used Llama for GSM8K because Qwen's pretraining is
mathematics-heavy; these runs used Qwen.

## Scheduling limitations

The report has a section on three placement-layer limitations this run exposed:
finished jobs never release their workers, LoRA runs sharing a base model cannot
spread across workers, and placement is never revisited as capacity frees. Those
are why the four 8B LoRA runs took 288s per step against 134s for full
fine-tuning, and finished hours after the rest.

## Files

```
report.html            the write-up, self-contained
metrics-0.6b/*.jsonl   raw per-run metrics, Qwen3-0.6B (fft/r1/r32 × a/b)
metrics-8b/*.jsonl     raw per-run metrics, Qwen3-8B
rank_sweep-0.6b.csv    tidy extract (run, group, config, replicate, step, reward, ...)
rank_sweep-8b.csv      same, 8B
```

Raw metrics are kept so the analysis can be rebuilt without re-running. The
scenario deletes its log directory on start, so a later run of the same scenario
overwrites the previous one's metrics on the shared volume; these copies are the
record.

## Reproducing

```bash
# Cluster with the open-rl gateway deployed:
make cluster-e2e E2E_SCENARIO=gsm8k-rl-rank-sweep-mega E2E_ARGS="steps=40"

# Analysis. DIR holds one subdirectory per run; GROUP selects a model size when
# one directory holds several (the files here are flattened for readability).
cd dev/tools
uv run python rank_sweep_report.py --runs-dir <dir> --group large --out-dir out
uv run python rank_sweep_html.py \
  --run "Qwen3-0.6B on L4=Qwen3-0.6B=<dir>=small" \
  --run "Qwen3-8B on H100=Qwen3-8B=<dir>=large" --out report.html
```

## Run conditions

- GKE, 3× L4 nodes (2 GPUs each) and 4× H100 nodes. Tier selection follows the
  model: a 0.6B full fine-tune fits a 24GB L4; an 8B one needs an 80GB H100, as
  does an 8B LoRA worker, because the sampler must hold 16.4GB of frozen weights
  inside the fraction of the device vLLM is given.
- Placement: the four LoRA runs of one model size shared a single trainer and a
  single sampler as four adapters, by base-model affinity. The two full
  fine-tuning replicates took separate workers, but were packed two to a claim,
  so no configuration here had a dedicated GPU.
- Throughput: 0.6B 58s/step for full fine-tuning against 128s for the shared
  LoRA runs; 8B 134s against 288s. The roughly 2× is time-slicing, not LoRA
  being slower.
