PYTHON := uv run python
SRC := src
OUTPUT := data/output/function_calling_results.json

.PHONY: install run debug clean lint lint-strict test benchmark help

install:
	uv sync

run:
	$(PYTHON) -m $(SRC)

debug:
	$(PYTHON) -m pdb -m $(SRC)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf data/output/accuracy_results
	rm -rf data/output/generated_accuracy
	rm -f $(OUTPUT)

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest -q

benchmark:
	./benchmark_accuracy_with_comparison.sh

help:
	@echo "Verfügbare Targets:"
	@echo "  make install      - Abhängigkeiten mit uv installieren"
	@echo "  make run          - Projekt starten"
	@echo "  make debug        - Projekt mit pdb starten"
	@echo "  make clean        - Caches und generierte Dateien löschen"
	@echo "  make lint         - flake8 und mypy mit Pflicht-Flags ausführen"
	@echo "  make lint-strict  - flake8 und mypy --strict ausführen"
	@echo "  make test         - pytest ausführen"
	@echo "  make benchmark    - Accuracy-Benchmark starten"
