#!/bin/bash
# =============================================================
# Swiss aFRR Pipeline — Infrastructure Setup Script
# Run this from: swiss-afrr-spike-prediction/pipeline/
# Usage: bash setup.sh
# =============================================================
 
PIPELINE_DIR="/Users/yuan/Work/profolio/swiss-afrr-spike-prediction/pipeline"
PROJECT_DIR="/Users/yuan/Work/profolio/swiss-afrr-spike-prediction"
VENV="$PIPELINE_DIR/.venv-airflow/bin/activate"
ENV_FILE="$PROJECT_DIR/.env"
 
echo "=== Step 1: Activating venv ==="
source $VENV
echo "Python: $(which python)"
 
echo ""
echo "=== Step 2: AWS credentials (override SSO for LocalStack) ==="
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
 
echo ""
echo "=== Step 3: Fixing known dependency issues ==="
pip install "flask-session==0.5.0" "email-validator>=2.0" fastapi openpyxl -q
echo "Dependencies OK"
 
echo ""
echo "=== Step 4: Starting LocalStack (Docker) ==="
if docker ps | grep -q localstack; then
  echo "LocalStack already running — skipping"
else
  docker run --rm -d \
    --name localstack \
    -p 4566:4566 \
    localstack/localstack
  echo "Waiting 10s for LocalStack to be ready..."
  sleep 10
fi
 
echo ""
echo "=== Step 5: Creating S3 bucket (if not exists) ==="
awslocal s3 mb s3://energy-pipeline 2>/dev/null || echo "Bucket already exists — skipping"
awslocal s3 ls
 
echo ""
echo "=== Step 6: Starting MLflow (background) ==="
cd $PIPELINE_DIR
lsof -ti:5001 | xargs kill -9 2>/dev/null || true
sleep 2
nohup mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5001 > mlflow.log 2>&1 &
echo "MLflow started → http://localhost:5001 (logs: pipeline/mlflow.log)"
 
echo ""
echo "=== Step 7: Airflow setup ==="
export AIRFLOW_HOME=~/airflow
 
# Init DB only if not already done
if [ ! -f ~/airflow/airflow.db ]; then
  echo "Initialising Airflow DB..."
  airflow db init
  airflow config set core load_examples False
fi
 
# Always recreate admin user — safe to run every time
# If user already exists, the error is suppressed silently
echo "Ensuring admin user exists..."
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@admin.com \
  --password admin 2>/dev/null || echo "Admin user already exists — skipping"
 
echo ""
echo "=== Step 8: Starting Airflow scheduler + webserver (background) ==="
pkill -f "airflow scheduler" 2>/dev/null || true
pkill -f "airflow webserver" 2>/dev/null || true
rm -f ~/airflow/airflow-webserver.pid 2>/dev/null || true
sleep 2
 
nohup airflow scheduler > ~/airflow/scheduler.log 2>&1 &
echo "Scheduler started (logs: ~/airflow/scheduler.log)"
sleep 3
 
nohup airflow webserver --port 8080 > ~/airflow/webserver.log 2>&1 &
echo "Webserver started → http://localhost:8080 (logs: ~/airflow/webserver.log)"
sleep 5
 
echo ""
echo "=== Step 9: Setting Airflow Variables ==="
airflow variables set S3_BUCKET "energy-pipeline"
airflow variables set MLFLOW_TRACKING_URI "http://localhost:5001"
airflow variables set PIPELINE_TEST_DATE "2025-12-15"
echo "PIPELINE_TEST_DATE set to 2025-12-15"
 
# Read ENTSOE_API_KEY from .env file automatically
if [ -f "$ENV_FILE" ]; then
  ENTSOE_KEY=$(grep ENTSOE_TOKEN "$ENV_FILE" | cut -d '=' -f2 | tr -d ' "')
  if [ -n "$ENTSOE_KEY" ]; then
    airflow variables set ENTSOE_API_KEY "$ENTSOE_KEY"
    echo "ENTSOE_API_KEY set from .env"
  else
    echo "⚠️  ENTSOE_TOKEN not found in .env — set manually:"
    echo "    airflow variables set ENTSOE_API_KEY 'your_key'"
  fi
else
  echo "⚠️  .env file not found at $ENV_FILE"
fi
 
echo ""
echo "=== Step 10: Setting Airflow S3 connection ==="
airflow connections delete 'aws_localstack' 2>/dev/null || true
airflow connections add 'aws_localstack' \
  --conn-type 'aws' \
  --conn-extra '{
    "endpoint_url": "http://localhost:4566",
    "aws_access_key_id": "test",
    "aws_secret_access_key": "test",
    "region_name": "us-east-1"
  }'
echo "Connection aws_localstack set"
 
echo ""
echo "============================================"
echo "✅ All services started:"
echo "   LocalStack S3 → http://localhost:4566"
echo "   MLflow UI     → http://localhost:5001"
echo "   Airflow UI    → http://localhost:8080  (admin/admin)"
echo "============================================"