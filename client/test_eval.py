import traceback
from test_countdown_rl import evaluate_puzzle

print(evaluate_puzzle("Some <reasoning>test</reasoning><solution>5 + 10 * 2</solution>", [5, 10, 2], 25))
