.PHONY: install data test run docker deploy

install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt

data:
	python -m scripts.bootstrap

test:
	python -m pytest -q

run:
	uvicorn app.main:app --reload --port 8000

docker:
	docker build -t baytak-ai . && docker run --rm -p 8000:8000 --env-file .env baytak-ai

deploy:
	bash infra/deploy.sh
