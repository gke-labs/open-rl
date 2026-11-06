output "port_forward" {
  description = "Expose the OpenRL API server on 127.0.0.1:9003, where the recipe expects it."
  value       = "kubectl -n openrl-system port-forward svc/open-rl-api-server-service 9003:8000"
}

output "run_recipe" {
  description = "Run the text-to-SQL recipe as published, on Gemma 4 E2B (from the repo root, with the port-forward running)."
  value       = "(cd examples/text-to-sql && uv run python texttosql_sft_grpo.py gemma4_e2b_rl_recipe)"
}
