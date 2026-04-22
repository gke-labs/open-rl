# Text-to-SQL Supervised Fine-Tuning

This directory contains examples for Supervised Fine-Tuning (SFT) on Text-to-SQL tasks using Open-RL.

## Prerequisites

1. **Install dependencies**:
   Set up the server and client environments:
   ```bash
   cd src && uv sync --extra cpu
   cd ../examples && uv sync
   ```

## Running the Training Server

Start the local single-process Open-RL server:
```bash
make server BASE_MODEL=google/gemma-3-1b-pt
```

## Running the SFT Script

Execute the training script:
```bash
cd examples/sft/text-to-sql
uv run python texttosql_sft.py gemma
```

## Contents

* `texttosql_sft.py`: Script to run the SFT training.
* `texttosql_sft_notebook.ipynb`: Jupyter notebook demonstrating the SFT process.
* `texttosql_gemma4_plain_notebook.ipynb`: Jupyter notebook demonstrating SFT with Gemma 4.
* `README.md`: This documentation file.
