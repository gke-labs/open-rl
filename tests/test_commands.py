import unittest

from training import commands
from training.types import Datum


class CommandWireFormatTest(unittest.TestCase):
  def test_commands_round_trip_through_the_wire(self) -> None:
    for command in (
      commands.CreateModel(request_id="r", model_id="m", base_model="base", fine_tuning_type="full", full_config={"cpu_offload": False}),
      commands.CreateModelFromState(request_id="r", model_id="m", state_path="/ckpt", restore_optimizer=True),
      commands.OptimStep(request_id="r", model_id="m", adam_params={"learning_rate": 1e-4}),
      commands.Sample(request_id="r", model_id="m", prompt_tokens=[1, 2, 3], max_tokens=4),
      commands.SaveState(request_id="r", model_id="m", state_path="/ckpt", include_optimizer=True, kind="weights"),
      commands.LoadWeights(request_id="r", model_id="m", state_path="/ckpt"),
      commands.SaveWeightsForSampler(request_id="r", model_id="m", alias="final", sampling_session_id="tinker://m/sampler_weights/sampler-0"),
      commands.SaveWeights(request_id="r", model_id="m", alias="final"),
      commands.Shutdown(model_id="m"),
    ):
      with self.subTest(op=command.op):
        raw = commands.wire(command)
        self.assertEqual(raw["op"], command.op)
        self.assertEqual(commands.parse_command(raw), command)

  def test_forward_backward_flattens_the_tinker_datum(self) -> None:
    raw = {
      "request_id": "r",
      "model_id": "m",
      "op": "forward_backward",
      "loss_fn": "importance_sampling",
      "data": [
        {
          "model_input": {"chunks": [{"tokens": [1, 2]}, {"tokens": [3]}]},
          "loss_fn_inputs": {"target_tokens": [2, 3, 4], "weights": {"data": [1.0, 0.5, 0.25]}},
        }
      ],
    }
    command = commands.parse_command(raw)
    self.assertIsInstance(command, commands.ForwardBackward)
    datum = command.data[0]
    self.assertEqual(datum.model_input, [1, 2, 3])
    self.assertEqual(datum.loss_fn_inputs["target_tokens"].data, [2, 3, 4])
    self.assertEqual(datum.loss_fn_inputs["weights"].data, [1.0, 0.5, 0.25])
    # Once flat, the datum survives another trip unchanged.
    self.assertEqual(commands.parse_command(commands.wire(command)), command)
    self.assertEqual(Datum.model_validate(datum.model_dump()), datum)

  def test_legacy_shutdown_sentinel_parses(self) -> None:
    command = commands.parse_command({"request_id": "SHUTDOWN_SENTINEL", "model_id": "m", "op": "shutdown_workers"})
    self.assertIsInstance(command, commands.Shutdown)
    self.assertEqual(command.model_id, "m")

  def test_unknown_op_is_rejected(self) -> None:
    with self.assertRaises(ValueError):
      commands.parse_command({"request_id": "r", "model_id": "m", "op": "frobnicate"})

  def test_gpu_commands_cover_every_model_touching_op(self) -> None:
    gpu_ops = {command.model_fields["op"].default for command in commands.GPU_COMMANDS}
    self.assertEqual(gpu_ops, {"create_model", "create_model_from_state", "forward_backward", "optim_step", "sample", "load_weights"})


if __name__ == "__main__":
  unittest.main()
