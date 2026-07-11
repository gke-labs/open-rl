# GKE Benchmark Report: 30-Step Qwen3-8B Reinforcement Learning (`192 Batch Size`)

**Date:** July 10, 2026  
**Model:** `Qwen/Qwen3-8B` (8,190,735,360 total parameters)  
**Configuration:** `192 trajectories per step` (`groups_per_batch=24`, `group_size=8`, `max_tokens=512`)  
**Cluster:** Native GKE (`open-rl-dra`, dedicated 2× NVIDIA H100 80GB HBM3 nodes: Trainer + Sampler)  
**Synchronization Engine:** **CPU Weights Snapshot Delta Sync** (`llmd-snapshot-agent` + `tinker_cookbook` slice mapping)

---

## 1. Executive Summary

This benchmark evaluates a full 30-step Reinforcement Learning training run (`fft-gsm8k-rl`) on the 8.19 Billion parameter `Qwen/Qwen3-8B` base model configured with an expanded rollout batch size of **192 trajectories per step** (`24 prompt questions` × `8 rollouts/group`).

### Key Achievements (`30 / 30 Steps Completed & Verified`):
1. **Multiple 100.00% Reasoning Accuracy Peaks (`Steps 20, 23, 24`)**:
   * At **Step 20**, **Step 23**, and **Step 24**, the model achieved **100.00% reasoning accuracy (`192 / 192 trajectories correct`)**.
   * Across Steps 16–29, reasoning accuracy consistently maintained **90.62% – 100.00%**, with **0% complete failures** (`frac_all_bad = 0.00%`).
2. **Peak Format Compliance (`98.96% at Step 23`)**:
   * Format adherence (`\boxed{...}`) climbed from **8.85%** at initialization up to **98.96%** (`190 / 192 trajectories`) at Step 23, finishing at **98.44%** on Step 29.
3. **46.5% Chain-of-Thought Token Compression**:
   * Generation length (`ac_tokens_per_turn`) dropped from **506.1 tokens** down to **270.8 tokens/turn** at Step 29, demonstrating that the RL policy eliminates verbose rambling and discovers direct, concise multi-step proof paths.
4. **Empirical Step Execution Speed (`~82.7s – 85.0s`)**:
   * By Step 20+, generating 192 trajectories (`512 max tokens`) in vLLM takes **26.6s – 29.1s**, and the full 192-trajectory RL backprop + Delta Sync step completes in **82.7s – 85.7 seconds (~1.4 minutes per step)**.

---

## 2. Complete Complete Progression Table (`Step 0` – `Step 29`)

| Step | Average Reward (`reward/total`) | Correct Rate (`correct`) | Format Compliance (`format`) | Generation Tokens / Turn (`ac_tokens_per_turn`) | Perfect-Solve Questions (`frac_all_good`) | Sampling Time (`time/sampling`) | Total Step Time (`time/total`) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 (Init)** | **0.0339** | 12.50% | 8.85% | 506.1 | 0.00% | 147.4s | 250.8s |
| **1** | **0.2729** | 34.90% | 23.96% | 484.6 | 12.50% | 44.5s | 122.1s |
| **2** | **0.4755** | 54.69% | 28.65% | 475.1 | 8.33% | 41.4s | 112.4s |
| **3** | **0.5375** | 60.94% | 28.12% | 494.8 | 16.67% | 40.7s | 107.4s |
| **4** | **0.6198** | 68.75% | 32.29% | 454.8 | 29.17% | 38.3s | 110.2s |
| **5** | **0.8812** | 91.15% | 69.79% | 410.3 | 54.17% | 37.2s | 104.0s |
| **6** | **0.8604** | 90.10% | 59.38% | 410.1 | 66.67% | 36.7s | 101.3s |
| **7** | **0.8396** | 87.50% | 64.58% | 415.3 | 62.50% | 36.7s | 98.9s |
| **8** | **0.9266** | 94.79% | 78.65% | 371.9 | 66.67% | 34.9s | 97.8s |
| **9** | **0.9552** | 98.96% | 65.62% | 397.3 | 62.50% | 32.9s | 92.6s |
| **10** | **0.9547** | 96.88% | 85.94% | 375.7 | 75.00% | 32.8s | 95.5s |
| **11** | **0.9495** | 96.35% | 85.94% | 361.5 | 79.17% | 31.6s | 89.9s |
| **12** | **0.8760** | 89.58% | 80.21% | 368.0 | 70.83% | 32.4s | 90.5s |
| **13** | **0.9083** | 92.71% | 81.25% | 373.2 | 70.83% | 32.8s | 90.4s |
| **14** | **0.8927** | 90.62% | 86.46% | 338.3 | 66.67% | 32.9s | 90.4s |
| **15** | **0.9323** | 94.27% | 89.58% | 317.0 | 79.17% | 31.3s | 88.2s |
| **16** | **0.9870** | 99.48% | 92.19% | 306.4 | 87.50% | 31.0s | 91.6s |
| **17** | **0.9849** | 99.48% | 90.10% | 333.5 | 83.33% | 29.8s | 89.1s |
| **18** | **0.9844** | 98.96% | 94.79% | 335.7 | 79.17% | 30.1s | 87.4s |
| **19** | **0.9542** | 95.83% | 95.83% | 274.0 | 95.83% | 28.4s | 87.1s |
| **20** *(100% Hit)* | **0.9958** | **100.00%** *(192/192)* | 95.83% | 300.4 | 91.67% | 28.6s | 87.3s |
| **21** | **0.9198** | 92.71% | 92.71% | 280.4 | 79.17% | **26.6s** *(Fastest)* | **82.7s** *(Fastest)* |
| **22** | **0.9911** | 99.48% | 96.35% | 279.6 | 91.67% | 27.8s | 84.2s |
| **23** *(Peak Reward)* | **0.9990** *(Peak)* | **100.00%** *(192/192)* | **98.96%** *(Peak)* | 282.0 | 91.67% | 29.3s | 85.2s |
| **24** *(100% Hit)* | **0.9984** | **100.00%** *(192/192)* | 98.44% | 281.9 | 91.67% | 30.1s | 85.7s |
| **25** | **0.9453** | 94.79% | 97.40% | 279.2 | 87.50% | 29.1s | 84.6s |
| **26** | **0.9109** | 91.67% | 94.27% | 292.7 | 83.33% | 29.0s | 83.9s |
| **27** | **0.9495** | 95.31% | 96.35% | 277.6 | 83.33% | 26.9s | 83.7s |
| **28** | **0.8964** | 90.62% | 90.10% | 290.2 | 79.17% | 28.2s | 84.0s |
| **29 (Final)** | **0.9568** | 95.83% | 98.44% | **270.8** *(Shortest)* | **95.83%** *(23/24 Qs)* | 27.6s | **83.5s** |

---

## 3. Empirical Delta Weight Sparsity Decay (`Qwen3-8B` Parameters)

We tracked the exact percentage of parameters modified (`num_changed / 8,190,735,360`) above threshold at every step:

| Step | Changed Parameters (`num_changed`) | Total Parameters | Fraction Changed (`%`) | Sparse Delta File Size |
|:---:|:---:|:---:|:---:|:---:|
| **1** | `1,056,300,801` | `8,190,735,360` | **12.90%** *(Initial exploration)* | ~4.22 GB |
| **2** | `737,022,810` | `8,190,735,360` | **9.00%** | ~2.95 GB |
| **5** | `481,104,355` | `8,190,735,360` | **5.87%** | ~1.92 GB |
| **10** | `329,379,530` | `8,190,735,360` | **4.02%** | ~1.32 GB |
| **15** | `273,122,061` | `8,190,735,360` | **3.34%** | ~1.09 GB |
| **20** *(100% Hit)* | `232,783,311` | `8,190,735,360` | **2.84%** | ~0.93 GB |
| **21** *(Sparsity Peak)* | `218,717,742` | `8,190,735,360` | **2.67%** *(Lowest sparsity)* | ~0.87 GB |
| **24** | `219,364,934` | `8,190,735,360` | **2.68%** | ~0.88 GB |

### Synchronization Timing Performance (`Steps 20–29`):
* **CPU Sparse Delta Application (`llmd-snapshot-agent` patch)**: **`4.03s – 4.66s`** (`4032.87 ms – 4661.80 ms`)
* **Host CPU Snapshot ➔ GPU VRAM Transfer (`collective_rpc reload_weights`)**: **`0.66 seconds`** (`662.68 ms – 665.23 ms`)
* **Total Weight Synchronization Overhead**: **`~4.7 seconds total`** to update the full 8.19 Billion parameter model in-place.

---

## 4. Comparison Against Historical Baselines

| Baseline / Run | Batch Size (`Trajectories`) | Peak Accuracy (`correct`) | Peak Format (`format`) | Average Step Time | Relative Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|
| **July 3 Historical (`2jobs_concurrent_gpb24`)** | 192 (`24×8`) | 97.92% *(Step 11)* | 83.85% *(Step 11)* | ~435s – 485s | 1.0× *(Old Baseline)* |
| **July 9 64-Batch Single Job (`gpb8`)** | 64 (`8×8`) | 100.00% *(Step 18)* | 90.62% *(Step 18)* | ~172.0s | 2.6× *(Smaller Batch)* |
| **Current Live Run (`1job_fft_rl_192batch`)** | **192 (`24×8`)** | **100.00%** *(Steps 20, 23, 24)* | **98.96%** *(Step 23)* | **~82.7s – 87.3s** | **5.3× Faster than July 3 (`192 batch`)** |
