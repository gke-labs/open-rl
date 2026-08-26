# Design Doc 012: Reproducing "LoRA Without Regret" — Concurrent Rank Sweep on GSM8K RL

## 1. Executive Summary

Thinking Machines' [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
argues that LoRA matches full fine-tuning (FullFT) in both sample efficiency and
final performance across most post-training regimes, and makes a much stronger
claim for reinforcement learning specifically: **LoRA matches FullFT even at
rank 1**. Their reasoning is information-theoretic — a policy gradient extracts
O(1) bits per episode against O(tokens) for supervised learning, roughly a
1000× difference per token — so an adapter holding a few million parameters has
ample capacity for an RL run that carries a few hundred thousand bits in total.

This document proposes reproducing that claim on Open-RL, running every arm of
the comparison **concurrently on one cluster**. The parallel design is not
presentational: arms launched together see identical cluster state, dataset, and
time-slicer contention, which makes it a tighter control than sequential runs
whose conditions drift. It also exercises exactly what Open-RL exists to do —
multiplexing several RL jobs over a shared accelerator fleet — so a single run
produces both a scientific result and an end-to-end demonstration.

---

## 2. What We Are Testing

The claim under test, stated so it can fail:

> On GSM8K RL, reward-versus-step curves for FullFT, LoRA rank 32, and LoRA
> rank 1 are indistinguishable within run-to-run noise.

Falsification is meaningful in either direction. If rank 1 tracks FullFT, then
for RL workloads the FFT machinery in this repo — the `80gb` memory tier, the
pinned-DRAM optimizer shadow, delta weight synchronization of hundreds of
millions of parameters — is avoidable, and LoRA should become the default for
new RL recipes. If rank 1 falls behind while rank 32 does not, we have located a
capacity threshold the paper's information-theoretic argument does not predict
at our scale, which is a finding worth publishing back.

### 2.1 Secondary observations

Cheap to collect from the same run, and worth recording:

- **Learning-rate ratio.** The paper fits an optimal LoRA LR of ~10× FullFT
  (9.8 across 14 models). This repo already hardcodes `1e-4` for LoRA against
  `1e-5` for FFT in the GSM8K RL path. The run does not *measure* the optimum,
  but a healthy result at the 10× ratio is corroborating evidence.
- **Throughput under sharing.** Wall-clock per step for arms multiplexed on one
  GPU versus an arm holding its own.

---

## 3. Experiment Design

### 3.1 Arms

Three arms on one base model, differing only in the variable under test:

| Arm | Mode | Rank | Learning rate |
|---|---|---|---|
| A | FullFT | — | 1e-5 |
| B | LoRA | 32 | 1e-4 |
| C | LoRA | 1 | 1e-4 |

Everything else is held fixed: base model, dataset, `group_size`,
`groups_per_batch`, `max_tokens`, temperature, step count, and seed.

The learning-rate asymmetry is deliberate and must not be "corrected". Equalizing
LR across modes would compare each arm at a different distance from its own
optimum and is the more misleading experiment. The 10× ratio is the paper's
finding; we adopt it.

### 3.2 Concurrency and expected placement

All three arms launch together against one gateway. Given the tiering and claim
rules already in the scheduler, the expected topology on a small base model is:

- Arms **B and C share one trainer worker and one sampler worker**. Both are
  LoRA on the same base model, so base-model affinity binds them to the same
  claim, and the LoRA trainer builds a **separate PEFT adapter per `model_id`**
  with its own rank. Two adapters of different rank therefore coexist on one
  GPU — the multi-tenant path this project is built around.
- Arm **A takes its own trainer and sampler claims**, since FFT workers do not
  share adapters.

That is four GPUs at the `24gb` tier for a sub-billion-parameter model. The
run should not require the `80gb` tier at all, which is itself a consequence of
size-aware tiering: before it, any FFT arm would have demanded an 80 GB device.

### 3.3 Measurement

Primary axis is **reward versus step**, not wall-clock. Arms B and C share a GPU
while A does not, so throughput differs by construction; per-step learning is
the only fair comparison. Each arm writes metrics to its own log path, and the
comparison is a join across arms on step index.

---

## 4. Graphs We Need

The experiment is only as good as the plots that decide it.

1. **Reward vs. step, all arms overlaid** — the primary result. Confirmation
   looks like three curves inside one noise band; refutation looks like arm C
   separating downward and staying there.
2. **Reward vs. step with a noise band** — the same data with a shaded
   inter-seed range, so "indistinguishable" is a claim about variance rather
   than an eyeball judgement. This requires repeat seeds (§6).
3. **Final/mean reward vs. rank**, with FullFT as a horizontal reference line —
   the compact summary. The paper's claim is a flat line up to rank 1.
4. **Wall-clock per step, per arm** — the systems half. Shows the cost of
   multiplexing two adapters on one GPU against a dedicated one, and is the plot
   that makes the concurrency story legible.
5. **GPU occupancy over the run** — which claims existed, how many workers each
   carried, and which tier they landed on. Evidence that three jobs genuinely
   shared a fleet rather than running in sequence.

Plots 1 and 3 decide the claim; 4 and 5 are the Open-RL story.

---

## 5. Fidelity Caveats

Stated up front, because they bound what a null result means.

1. **Model family.** The paper deliberately used Llama for MATH and GSM8K,
   because Qwen's math-heavy pretraining confounds the measurement. Open-RL's
   renderer selection currently routes to Qwen variants. A null result on Qwen
   ("rank 1 matches") is therefore weaker evidence than the same result on
   Llama: it may mean the task was already solved by pretraining rather than
   that rank 1 sufficed. Mitigation is to confirm reward climbs from a low base;
   the honest fix is a Llama renderer.
2. **Scale.** The paper's RL runs cover far more episodes than a CI-sized run.
   A short run can only show curves tracking, not final-performance parity.
3. **Schedule.** They swept with a constant learning rate and no warmup or
   cooldown. Any schedule in our trainer shifts the effective LR and weakens the
   comparison to their fitted ratio.
4. **Single seed.** One run per arm cannot separate a real gap from noise.

---

## 6. Phasing

**Phase 1 — one seed, small model.** Three arms, short run, confirm the
placement topology and that all three arms train. Produces plots 1, 4 and 5.
Primarily a validation that the harness measures what we think.

**Phase 2 — repeat seeds.** Three arms × N seeds for the noise band, enabling
plot 2 and making plot 3 defensible. This is the phase that can actually
support or refute the claim.

**Phase 3 — optional extensions.** A Llama renderer to remove the pretraining
confound; a larger base model where FullFT lands on the `80gb` tier and LoRA
does not, which puts a cost figure on the claim.

---

## 7. Non-Goals

- Measuring the optimal learning rate for either mode. That is a sweep, not a
  comparison, and belongs in a separate experiment.
- Reproducing the supervised-learning half of the paper (capacity thresholds,
  batch-size penalty, layer placement). Those need dataset scales this harness
  is not currently pointed at.
- Producing benchmark-quality absolute scores. The claim is about arms relative
  to each other under identical conditions.
