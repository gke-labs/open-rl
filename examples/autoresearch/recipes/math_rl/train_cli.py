"""CLI wrapper around tinker_cookbook.recipes.math_rl.train that ensures Open-RL custom strategy headers are patched on startup."""

import asyncio

import chz
from common.precision import ensure_precision_tags
from tinker_cookbook.recipes.math_rl.train import CLIConfig, cli_main

if __name__ == "__main__":
  # The cookbook CLI has no dtype field; set TINKER_TAGS to train in another one.
  ensure_precision_tags("bfloat16")
  cli_config = chz.entrypoint(CLIConfig)
  asyncio.run(cli_main(cli_config))
