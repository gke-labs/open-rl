.PHONY: run-server run-client-sft run-client-simple

# Run the Uvicorn server locally, forcing the public PyPI index for uv
run-server:
	cd server && UV_INDEX_URL="https://pypi.org/simple" uv run uvicorn src.main:app --host 127.0.0.1 --port 8000

# Client test targets
run-client-basic:
	cd client && uv run --no-sync -i https://pypi.org/simple python test_basic_workflow.py

run-client-simple:
	cd client && uv run --no-sync -i https://pypi.org/simple python test_simple_rl.py

run-client-sft:
	cd client && uv run --no-sync -i https://pypi.org/simple python test_sft.py $(ARGS)

run-client-rlvr:
	cd client && uv run --no-sync -i https://pypi.org/simple python test_rlvr.py

run-client-showcase:
	cd client && uv run --no-sync -i https://pypi.org/simple python showcase_rlvr.py

