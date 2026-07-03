# Makefile for Data Pipeline Project
# Usage: make <target>

# Workflow order:
# 0. build              -> Build Docker images
# 1. infra-up           -> Start infrastructure and initialize services (docker/docker-compose.infrastructure.yml)
# 2. consumer.minio-up  -> Start MinIO consumer (docker/docker-compose.streaming.consumer.minio.yml)
# 3. producer-up        -> Start streaming producer and wait for reddit first-run status
# 4. batch_minio        -> Trigger batch_minio DAG (dags/batch_minio.py)
# 5. consumer.timescaledb-up -> Start TimescaleDB dashboard + featurestore consumers (docker/docker-compose.streaming.consumer.timescaledb.yml)
# 6. batch_timescaledb  -> Trigger batch_timescaledb DAG (dags/batch_timescaledb.py)
# 7. predict-up         -> Start streaming predict container (docker/docker-compose.streaming.predict.yml)
# 8. batch_predict      -> Trigger batch_predict DAG (dags/batch_predict.py)

.PHONY: infra-up producer-up consumer.minio-up batch_minio consumer.timescaledb-up batch_timescaledb predict-up batch_predict up down start-all build

CLOUDFLARED_LOCAL_URL := http://localhost:3000
CLOUDFLARED_WORKER_URL := https://thesis-redirect.honghongduc0102.workers.dev

# Build target
build:
	@echo "========================================================="
	@echo "STEP 0: Building Docker images..."
	@echo "========================================================="
	@echo "Building shared application image: thesis-pipeline:latest"
	@echo "Note: Docker will cache layers for faster rebuilds"
	docker build -t thesis-pipeline:latest -f docker/Dockerfile .
	@echo "[OK] Shared application image built successfully!"

# Infrastructure targets
infra-up:
	@echo "========================================================="
	@echo "STEP 1: Starting infrastructure..."
	@echo "========================================================="
	docker-compose -f docker/docker-compose.infrastructure.yml up -d --force-recreate
	@powershell -NoProfile -Command "& { \$$logDir = Join-Path (Get-Location) '.cloudflared'; \$$logFile = Join-Path \$$logDir 'cloudflared-infra.log'; New-Item -ItemType Directory -Force \$$logDir | Out-Null; if (Test-Path \$$logFile) { Remove-Item \$$logFile -Force }; Start-Process cloudflared -ArgumentList @('tunnel', '--url', '$(CLOUDFLARED_LOCAL_URL)', '--logfile', \$$logFile) -WindowStyle Hidden | Out-Null; \$$publicUrl = \$$null; \$$deadline = (Get-Date).AddSeconds(30); do { Start-Sleep -Seconds 1; if (Test-Path \$$logFile) { \$$match = Select-String -Path \$$logFile -Pattern 'https://[-a-zA-Z0-9]+\.trycloudflare\.com' | Select-Object -Last 1; if (\$$match) { \$$publicUrl = \$$match.Matches[0].Value } } } while ((-not \$$publicUrl) -and ((Get-Date) -lt \$$deadline)); if (-not \$$publicUrl) { Write-Host '[ERROR] Quick Tunnel URL not found.' -ForegroundColor Red; exit 1 }; \$$dashboardUrl = \$$publicUrl.TrimEnd('/') + '/dashboards'; Invoke-RestMethod -Method Post -Uri '$(CLOUDFLARED_WORKER_URL)' -ContentType 'application/json' -Body (@{ url = \$$dashboardUrl } | ConvertTo-Json -Compress) | Out-Null; Write-Host ('[OK] Quick Tunnel URL: ' + \$$publicUrl) -ForegroundColor Green; Write-Host ('[OK] Dashboard URL: ' + \$$dashboardUrl) -ForegroundColor Green; Write-Host ('[OK] Worker updated: $(CLOUDFLARED_WORKER_URL)') -ForegroundColor Green }"
	@echo "[OK] Infrastructure started!"



# MinIO consumer targets
consumer.minio-up:
	@echo "========================================================="
	@echo "STEP 2: Starting MinIO consumer..."
	@echo "========================================================="
	docker-compose -f docker/docker-compose.infrastructure.yml -f docker/docker-compose.streaming.consumer.minio.yml up -d --force-recreate streaming-consumer-minio-binance streaming-consumer-minio-reddit
	@echo "[OK] MinIO consumer started!"

# Producer targets
producer-up:
	@echo "========================================================="
	@echo "STEP 3: Starting producers..."
	@echo "========================================================="
	docker-compose -f docker/docker-compose.infrastructure.yml -f docker/docker-compose.streaming.producer.yml up -d --force-recreate streaming-producer-binance streaming-producer-reddit
	docker-compose -f docker/docker-compose.infrastructure.yml exec -T airflow-webserver python -m src.wait.reddit_status
	@echo "[OK] Producers started!"
	

# Airflow MinIO DAG targets
batch_minio:
	@echo "========================================================="
	@echo "STEP 4: Running DAG batch_minio..."
	@echo "========================================================="
	@powershell -Command "& { \$$timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss'); docker exec -e PYTHONWARNINGS=ignore airflow-webserver airflow dags trigger -e \$$timestamp batch_minio 2>&1 | Select-String -Pattern 'Created|triggered' -CaseSensitive; do { Start-Sleep -Seconds 5; \$$status = (docker exec -e PYTHONWARNINGS=ignore airflow-webserver airflow dags state batch_minio \$$timestamp 2>&1 | Select-String -Pattern 'queued|running|success|failed' -CaseSensitive).ToString().Trim(); } while (\$$status -match 'running|queued'); if (\$$status -ne 'success') { Write-Host ('[ERROR] DAG Failed: ' + \$$status) -ForegroundColor Red; exit 1 } }"
	@echo "[OK] DAG batch_minio finished!"

# TimescaleDB consumer targets
consumer.timescaledb-up:
	@echo "========================================================="
	@echo "STEP 5: Starting TimescaleDB consumers..."
	@echo "========================================================="
	docker-compose -f docker/docker-compose.infrastructure.yml -f docker/docker-compose.streaming.consumer.timescaledb.yml up -d --force-recreate streaming-consumer-timescaledb-dashboard streaming-consumer-timescaledb-featurestore
	@powershell -NoProfile -Command "Start-Sleep -Seconds 240"
	@echo "[OK] TimescaleDB consumers started!"

# Airflow TimescaleDB DAG targets
batch_timescaledb:
	@echo "========================================================="
	@echo "STEP 6: Running DAG batch_timescaledb..."
	@echo "========================================================="
	@powershell -Command "& { \$$timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss'); docker exec -e PYTHONWARNINGS=ignore airflow-webserver airflow dags trigger -e \$$timestamp batch_timescaledb 2>&1 | Select-String -Pattern 'Created|triggered' -CaseSensitive; do { Start-Sleep -Seconds 5; \$$status = (docker exec -e PYTHONWARNINGS=ignore airflow-webserver airflow dags state batch_timescaledb \$$timestamp 2>&1 | Select-String -Pattern 'queued|running|success|failed' -CaseSensitive).ToString().Trim(); } while (\$$status -match 'running|queued'); if (\$$status -ne 'success') { Write-Host ('[ERROR] DAG Failed: ' + \$$status) -ForegroundColor Red; exit 1 } }"
	@echo "[OK] DAG batch_timescaledb finished!"

# Predict targets
predict-up:
	@echo "========================================================="
	@echo "STEP 7: Starting predict container..."
	@echo "========================================================="
	docker-compose -f docker/docker-compose.infrastructure.yml -f docker/docker-compose.streaming.predict.yml up -d --force-recreate streaming-predict
	@echo "[OK] Predict container started!"

batch_predict:
	@echo "========================================================="
	@echo "STEP 8: Running DAG batch_predict..."
	@echo "========================================================="
	@powershell -Command "& { \$$timestamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss'); docker exec -e PYTHONWARNINGS=ignore airflow-webserver airflow dags trigger -e \$$timestamp batch_predict 2>&1 | Select-String -Pattern 'Created|triggered' -CaseSensitive; do { Start-Sleep -Seconds 5; \$$status = (docker exec -e PYTHONWARNINGS=ignore airflow-webserver airflow dags state batch_predict \$$timestamp 2>&1 | Select-String -Pattern 'queued|running|success|failed' -CaseSensitive).ToString().Trim(); } while (\$$status -match 'running|queued'); if (\$$status -ne 'success') { Write-Host ('[ERROR] DAG Failed: ' + \$$status) -ForegroundColor Red; exit 1 } }"
	@echo "[OK] DAG batch_predict finished!"

# Combined operations
up: infra-up consumer.minio-up producer-up batch_minio consumer.timescaledb-up batch_timescaledb predict-up batch_predict

down:
	@powershell -NoProfile -Command "& { \$$timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss'); \$$logDir = Join-Path (Get-Location) ('.log/' + \$$timestamp); New-Item -ItemType Directory -Force \$$logDir | Out-Null; \$$containers = @('streaming.producer.binance', 'streaming.producer.reddit', 'streaming.consumer.minio.binance', 'streaming.consumer.minio.reddit', 'streaming.consumer.timescaledb.dashboard', 'streaming.consumer.timescaledb.featurestore', 'streaming.predict'); foreach (\$$container in \$$containers) { \$$exists = docker ps -a --format '{{.Names}}' | Where-Object { \$$_ -eq \$$container }; if (\$$exists) { docker logs \$$container *> (Join-Path \$$logDir (\$$container + '.log')) } }; Write-Host ('[OK] Container logs exported to ' + \$$logDir) -ForegroundColor Green }"
	@docker-compose -f docker/docker-compose.infrastructure.yml -f docker/docker-compose.streaming.producer.yml -f docker/docker-compose.streaming.consumer.minio.yml -f docker/docker-compose.streaming.consumer.timescaledb.yml -f docker/docker-compose.streaming.predict.yml down
	@powershell -NoProfile -Command "& { Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue; Write-Host '[OK] cloudflared stopped if running.' -ForegroundColor Green }"

start-all: build up
