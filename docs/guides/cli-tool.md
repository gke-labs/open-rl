# CLI Tool Usage

The project includes a CLI tool for inspecting and interacting with trained adapters.

### 1. List Available Adapters
View all fine-tuned sessions, including their aliases and creation timestamps.

```bash
make run-cli-list
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
make run-cli-chat MODEL=<model_id>
```

**Optional: System Prompt Override**
```bash
make run-cli-chat MODEL=<model_id> PROMPT="You are a pirate."
```

### 3. Generic Usage
For other commands or arguments not covered by shortcuts:

```bash
make run-cli list
make run-cli chat --model <id> --temperature 0.9
```
