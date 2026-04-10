import asyncio

from .trainer import clock_cycle_loop, engine


def main() -> None:
  asyncio.run(clock_cycle_loop())


if __name__ == "__main__":
  main()
