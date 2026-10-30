"""CLI wrapper around tinker_cookbook.recipes.math_rl.train that ensures Open-RL custom strategy headers are patched on startup."""

import asyncio

import chz
from tinker_cookbook.recipes.math_rl.train import CLIConfig, cli_main

if __name__ == "__main__":
  cli_config = chz.entrypoint(CLIConfig)
  asyncio.run(cli_main(cli_config))
