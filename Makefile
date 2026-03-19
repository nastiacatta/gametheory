# Minority game project — common commands for reproducibility
# Use from project root with venv activated.

.PHONY: install test run-static run-repeated run-repeated-full clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run-static:
	python -m src.main static

run-repeated:
	python -m src.main repeated --output_dir outputs

run-repeated-full:
	python -m src.main repeated --n_rounds 200 --output_dir outputs

run-baselines:
	python -m src.experiments.run_repeated_baselines --n_rounds 200 --output_dir outputs/baselines

run-inductive:
	python -m src.experiments.run_inductive --mode recency --n_rounds 200 --output_dir outputs/inductive

run-heterogeneous:
	python -m src.experiments.run_heterogeneous --mode mix --p_best 0.5 --p_softmax 0.5 --p_random 0.0 --n_rounds 200 --output_dir outputs/heterogeneous

clean:
	rm -rf __pycache__ src/__pycache__ src/agents/__pycache__ src/game/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
	rm -f outputs/repeated_*.csv outputs/*.png
