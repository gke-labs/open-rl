# Reinforcement Learning with Gemma 4 Models in Open-RL

When running Reinforcement Learning recipes (such as `rl_loop.py` or GSM8K / Math RL) with **Gemma 4** models, you may encounter formatting incompatibility issues if relying on default built-in chat renderers (e.g., `qwen3_instruct`), which emit tags unsupported by Gemma 4 (`<|im_start|>`, `<|im_end|>`).

Because the installed `tinker_cookbook` SDK package cannot be edited directly on disk, this guide outlines the recommended architectural approaches for supporting Gemma 4 without modifying SDK source code.

---

## Approach 1: Official SDK Extension via `register_renderer` (Recommended)

Inspection of `tinker_cookbook.renderers` reveals an official global hook designed specifically for extensibility: **`register_renderer(name, factory)`**.

### Implementation Steps
If you write a custom Gemma 4 renderer subclass implementing the SDK's `Renderer` interface (or import an external renderer library such as `https://github.com/PrimeIntellect-ai/renderers.git`), register it dynamically at the top of your training script before launching the run:

```python
from typing import Any
import tinker_cookbook.renderers as renderers
from transformers import PreTrainedTokenizerBase


# 1. Define or import your Gemma 4 Renderer subclass
class Gemma4Renderer(renderers.Renderer):
  def __init__(self, tokenizer: PreTrainedTokenizerBase):
    self.tokenizer = tokenizer

  def build_generation_prompt(self, messages: list[dict[str, str]]) -> str:
    # Utilize HuggingFace's official Gemma 4 chat template
    return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

  # Implement remaining Renderer methods (build_supervised_example, parse_response, etc.)
  ...


# 2. Define the factory function expected by the SDK
def gemma4_renderer_factory(tokenizer: PreTrainedTokenizerBase, image_processor: Any = None) -> renderers.Renderer:
  return Gemma4Renderer(tokenizer)


# 3. Register into the SDK global lookup dictionary at script startup
renderers.register_renderer("gemma4_instruct", gemma4_renderer_factory)
```

### Configuration
Once registered, configure your training script or TOML config to request the new renderer name:
```toml
model = "google/gemma-4-..."
renderer = "gemma4_instruct"
```
When `get_renderer("gemma4_instruct", tokenizer)` is invoked inside the SDK, it resolves your custom factory cleanly without throwing registry errors.

---

## Approach 2: The Text-to-SQL Pattern (Raw String & Token Bypassing)

Alternatively, you can follow the architectural pattern established in our `examples/text-to-sql/texttosql_sft_grpo.py` recipe, which bypasses the SDK chat renderer subsystem entirely:

1. **Format Prompts as Raw Strings:** Construct your prompt text directly using Gemma 4's native chat delimiters:
   ```python
   GEMMA4_PROMPT_TEMPLATE = "<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n"
   prompt_text = GEMMA4_PROMPT_TEMPLATE.format(question=row["question"])
   ```
2. **Pre-tokenize Inputs:** Pass the formatted text directly into the HuggingFace tokenizer:
   ```python
   prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
   ```
3. **Feed Raw Tokens to the Training Config:** By passing pre-tokenized `prompt_tokens` directly into your dataset items and training client requests, the SDK chat rendering layer is bypassed, ensuring model-agnostic execution across any tokenizer architecture.
