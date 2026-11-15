.PHONY: install install-dev test test-cov lint format clean build docs

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=memorable_ai --cov-report=html --cov-report=term-missing

lint:
	flake8 memorable_ai tests examples
	mypy memorable_ai

format:
	black memorable_ai tests examples benchmarks
	isort memorable_ai tests examples benchmarks

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.db" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +

build:
	python setup.py sdist bdist_wheel

docs:
	@echo "Documentation is in docs/ directory"
	@echo "See README.md for main documentation"

benchmark:
	python -m pytest benchmarks/ -v

benchmark-locomo:
	pytest tests/unit/test_locomo_loader.py tests/unit/test_locomo_smoke.py -v --no-cov

locomo-smoke:
	python scripts/run_locomo_benchmark.py --mode smoke
	pytest tests/unit/test_locomo_loader.py tests/unit/test_locomo_smoke.py -q --no-cov

# Requires .env with NVIDIA_API_KEY or GEMINI_API_KEY; ~15–30 min for 10 questions
locomo-subset:
	python scripts/run_locomo_benchmark.py --mode full --conversations 1 --questions 10 \
		--qa-timeout 300 --ingest-mode raw --context-format dated \
		--save-rows --output benchmarks/locomo/data/measured_subset.json

locomo-experiments:
	LOCOMO_QA_LLM_TIMEOUT=300 LOCOMO_JUDGE_LLM_TIMEOUT=180 \
	python scripts/run_locomo_experiments.py \
		--experiments retrieval_only_dated,dated_context,extracted,oracle \
		--questions 10 --conversations 1

benchmark-multihop:
	python -m pytest benchmarks/multihop/ -v

benchmark-temporal:
	python -m pytest benchmarks/temporal/ -v

benchmark-comparison:
	python -m pytest benchmarks/comparison/ -v

