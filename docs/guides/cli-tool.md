# CLI Tool Usage

The project includes a CLI tool for inspecting and interacting with trained adapters.

### 1. List Available Adapters
View all fine-tuned sessions, including their aliases and creation timestamps.

```bash
make cli list
```

**Output Example:**
```
ID                                       | ALIAS                          | CREATED
----------------------------------------------------------------------------------------------------
882faa32-3cc3-4dc6-9269-5e7c8aa7c01f     | rlvr_concise_capital           | 2026-02-15 03:27:24
1b705680-aee0-49f5-bd70-914f4260ab50     | rlvr_concise_answer            | 2026-02-15 03:27:11
```

### 2. Chat with an Adapter
Interactively test a specific adapter model.

```bash
make cli chat --model <model_id>
```

### 3. Other Commands
For other arguments:

```bash
make cli chat --model <id> --temperature 0.9 --system-prompt "You are a pirate."
```

Or run directly (after `uv sync`):

```bash
open-rl list
open-rl chat --model <id>
```
