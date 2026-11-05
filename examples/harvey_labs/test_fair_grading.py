import unittest
from types import SimpleNamespace

import tinker
from tinker_cookbook.rl.types import Env, InitialObservationOverflow, StepResult

from harvey_labs.episode import LabEpisodeEnv
from harvey_labs.results import eval_result
from harvey_labs.reward import reward_from_scores


class FakeEnv(Env):
  def __init__(self, initial_len=100, step_len=100, step_done=False):
    self.initial_len, self.step_len, self.step_done = initial_len, step_len, step_done

    async def reward_fn(_history):
      return 1.0, {"lab/criteria_total": 5.0, "lab/graded": 1.0}

    self.message_env = SimpleNamespace(reward_fn=reward_fn, history=[])

  async def initial_observation(self):
    return tinker.ModelInput.from_ints([0] * self.initial_len), []

  async def step(self, action, *, extra=None):
    return StepResult(
      reward=0.0,
      episode_done=self.step_done,
      next_observation=tinker.ModelInput.from_ints([0] * self.step_len),
      next_stop_condition=None,
      metrics={"max_tokens_reached": 1.0} if self.step_done else {},
    )


class FairGradingTest(unittest.IsolatedAsyncioTestCase):
  async def test_turn_budget_floor_and_grading(self):
    overflow = await LabEpisodeEnv(FakeEnv(initial_len=3000), 5, max_trajectory_tokens=5000).initial_observation()
    self.assertIsInstance(overflow, InitialObservationOverflow)
    self.assertEqual(overflow.metrics["context_overflow"], 1.0)

    env = LabEpisodeEnv(FakeEnv(initial_len=100, step_len=3000), 5, max_trajectory_tokens=5000)
    self.assertIsInstance(await env.initial_observation(), tuple)
    step = await env.step("a")
    self.assertEqual((step.episode_done, step.reward, step.metrics["context_overflow"], step.metrics["lab/graded"]), (True, 1.0, 1.0, 1.0))

    capped = await LabEpisodeEnv(FakeEnv(step_done=True), 5, max_trajectory_tokens=5000).step("a")
    self.assertEqual((capped.reward, capped.metrics["lab/graded"]), (1.0, 1.0))

  def test_reward_errors(self):
    self.assertEqual(reward_from_scores({"n_criteria": 0}), (0.0, {"lab/all_pass": 0.0, "lab/reward_error": 1.0, "lab/graded": 0.0}))
    reward, valid = reward_from_scores({"n_criteria": 2, "n_passed": 1})
    self.assertEqual((reward, valid["lab/reward_error"], valid["lab/graded"]), (0.5, 0.0, 1.0))
    row = {
      "test/env/all/lab/criteria_passed": 5.0,
      "test/env/all/lab/criteria_total": 10.0,
      "test/env/all/total_episodes": 4.0,
      "test/env/all/lab/reward_error": 0.5,
    }
    self.assertEqual(eval_result(row)["reward_errors"], 2.0)


if __name__ == "__main__":
  unittest.main()
