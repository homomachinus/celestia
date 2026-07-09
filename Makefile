.PHONY: install install-dev cpt sft merge quantize hub generate clean lint

# ── Installation ───────────────────────────────

install:
	pip install -e .
install-dev:
	pip install -e ".[gguf]"

# ── Training ───────────────────────────────────

cpt:
	celestia cpt --config config/cpt.yaml

sft:
	celestia sft --config config/sft.yaml

# ── Post-training ──────────────────────────────

merge:
	celestia merge --adapter output/sft_adapter --base Qwen/Qwen2.5-3B --output merged

quantize:
	celestia quantize --model merged --outdir gguf

hub:
	celestia hub --model merged --repo nuxt/celestia-plato-3b

# ── Inference ──────────────────────────────────

generate:
	celestia generate --model merged --prompt "What is justice?" --system "You are Plato"

# ── Housekeeping ───────────────────────────────

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

test:
	python -m pytest tests/ -v
