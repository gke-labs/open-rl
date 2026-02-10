
## Phase 2: RL Objectives
+ [x] Understand Tinker REINFORCE input spec (`importance_sampling`)
+ [x] Implement `importance_sampling` loss in `TrainerEngine`
+ [x] Add `logprobs` support to `/api/v1/asample` generated output
+ [x] Write `test_rl_workflow.py` script 
  - Simulate rollouts 
  - Apply length-based mock reward
  - Compute advantages
  - Run RL training loop

## Phase 3: Countdown RL Exeriment
+ [x] Understand Countdown rules and reward functions (from Sami's blog)
+ [x] Implement `client/test_countdown_rl.py`
  - Logic to generate source numbers and target
  - Logic to parse XML `<reasoning>` and `<solution>`
  - Reward calculation (exact match, closeness, formatting)
+ [x] Run test loop and plot performance curve

## Phase 4: Basic Math RL convergence test
+ [ ] Plan `test_simple_rl.py`
+ [ ] Build logic for generating single-digit addition/subtraction prompts
+ [ ] Calculate exact-match rewards
+ [ ] Run RL test for 100 epochs to prove `Qwen-0.5B` convergence.
