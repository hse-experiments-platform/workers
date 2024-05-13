CONFIGS_PATH=./configs

.PHONY: run
run:
	DOTENV_FILE=$(CONFIGS_PATH)/dev/.env LOG_LEVEL=DEBUG uvicorn main:app --host 0.0.0.0 --port 8100 --reload

.PHONY: docker-build
docker-build:
	docker build -t workers:latest -f ./build/Dockerfile .

.PHONY: docker-run
docker-run:
	docker compose -f deployments/dev/docker-compose.yaml up --build

.PHONY: docker-run
docker-run-background:
	docker compose -f deployments/dev/docker-compose.yaml up --build -d
