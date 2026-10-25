.PHONY: install ingest serve test lint up
install:
	pip install -r requirements-dev.txt
ingest:
	python -m rag.ingest data/sample
serve:
	uvicorn app.main:app --reload --port 8000
test:
	pytest -q
lint:
	ruff check .
up:
	docker compose up --build -d
