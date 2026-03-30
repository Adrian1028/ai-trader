#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# AI Trading Bot — Google Cloud Run Deployment
# ══════════════════════════════════════════════════════════════════
# Deploys the trading bot to Cloud Run with Cloud Scheduler triggers.
#
# Free tier usage (per month):
#   Requests:     ~620 / 2,000,000 free  (0.03%)
#   CPU:          ~37,000 / 180,000 free vCPU-seconds (20%)
#   Memory:       ~19,000 / 360,000 free GB-seconds (5%)
#   Scheduler:    3 jobs / 3 free jobs    (100%)
#
# Prerequisites:
#   1. Install gcloud CLI: https://cloud.google.com/sdk/install
#   2. gcloud auth login
#   3. gcloud config set project YOUR_PROJECT_ID
#   4. Enable APIs: Cloud Run, Cloud Scheduler, Cloud Build
#
# Usage:
#   ./deploy/deploy-cloudrun.sh              # Full deploy
#   ./deploy/deploy-cloudrun.sh --env-only   # Update env vars only
# ══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
REGION="${CLOUD_RUN_REGION:-us-central1}"
SERVICE_NAME="ai-trader"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Cloud Run resource settings (free-tier friendly)
MEMORY="512Mi"
CPU="1"
TIMEOUT="300"          # 5 min max per request
CONCURRENCY="1"        # One request at a time (sequential trading)
MIN_INSTANCES="0"      # Scale to zero when idle (free!)
MAX_INSTANCES="1"      # Only one instance ever

echo "=========================================="
echo " AI Trading Bot — Cloud Run Deploy"
echo "=========================================="
echo " Project:  $PROJECT_ID"
echo " Region:   $REGION"
echo " Service:  $SERVICE_NAME"
echo " Image:    $IMAGE"
echo "=========================================="

# ── Check prerequisites ──────────────────────────────────────────
if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# Enable required APIs
echo ""
echo "[1/5] Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    cloudscheduler.googleapis.com \
    secretmanager.googleapis.com \
    2>/dev/null || true
echo "  APIs enabled."

# ── Load environment variables ───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$REPO_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE"
    echo "Copy .env.example to .env and fill in your API keys."
    exit 1
fi

# Read .env and build --set-env-vars string
ENV_VARS=""
while IFS='=' read -r key value; do
    # Skip comments and empty lines
    [[ "$key" =~ ^#.*$ ]] && continue
    [[ -z "$key" ]] && continue
    # Trim whitespace
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    [ -z "$key" ] && continue

    if [ -n "$ENV_VARS" ]; then
        ENV_VARS="${ENV_VARS},${key}=${value}"
    else
        ENV_VARS="${key}=${value}"
    fi
done < "$ENV_FILE"

# Add Cloud Run specific env vars
ENV_VARS="${ENV_VARS},DATA_DIR=/tmp/data,LOG_LEVEL=INFO"

if [ "${1:-}" = "--env-only" ]; then
    echo ""
    echo "Updating environment variables only..."
    gcloud run services update "$SERVICE_NAME" \
        --region="$REGION" \
        --set-env-vars="$ENV_VARS"
    echo "Done. Environment variables updated."
    exit 0
fi

# ── Build and deploy ─────────────────────────────────────────────
echo ""
echo "[2/5] Building container image..."
cd "$REPO_DIR"
gcloud builds submit \
    --tag "$IMAGE" \
    --dockerfile Dockerfile.cloudrun \
    --timeout=600

echo ""
echo "[3/5] Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image="$IMAGE" \
    --region="$REGION" \
    --memory="$MEMORY" \
    --cpu="$CPU" \
    --timeout="$TIMEOUT" \
    --concurrency="$CONCURRENCY" \
    --min-instances="$MIN_INSTANCES" \
    --max-instances="$MAX_INSTANCES" \
    --no-allow-unauthenticated \
    --set-env-vars="$ENV_VARS" \
    --platform=managed

# Get the service URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" --format='value(status.url)')
echo ""
echo "  Service URL: $SERVICE_URL"

# ── Create Cloud Scheduler jobs ──────────────────────────────────
echo ""
echo "[4/5] Setting up Cloud Scheduler..."

# Create service account for scheduler
SA_NAME="scheduler-invoker"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Create SA if not exists
gcloud iam service-accounts create "$SA_NAME" \
    --display-name="Cloud Scheduler Invoker" 2>/dev/null || true

# Grant Cloud Run invoker role
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
    --region="$REGION" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.invoker" 2>/dev/null || true

# Job 1: Trading cycle — every 15 min during US market hours (Mon-Fri 13:00-19:45 UTC)
echo "  Creating trading scheduler..."
gcloud scheduler jobs delete "ai-trader-cycle" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs create http "ai-trader-cycle" \
    --location="$REGION" \
    --schedule="*/15 13-19 * * 1-5" \
    --time-zone="UTC" \
    --uri="${SERVICE_URL}/trade" \
    --http-method=POST \
    --oidc-service-account-email="$SA_EMAIL" \
    --attempt-deadline="300s" \
    --description="AI Trading Bot — 15-min market scan"

# Job 2: Post-market reflection — Mon-Fri 20:30 UTC
echo "  Creating reflection scheduler..."
gcloud scheduler jobs delete "ai-trader-reflect" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs create http "ai-trader-reflect" \
    --location="$REGION" \
    --schedule="30 20 * * 1-5" \
    --time-zone="UTC" \
    --uri="${SERVICE_URL}/reflect" \
    --http-method=POST \
    --oidc-service-account-email="$SA_EMAIL" \
    --attempt-deadline="300s" \
    --description="AI Trading Bot — Post-market reflection"

# Job 3: Daily report — Mon-Fri 21:00 UTC
echo "  Creating report scheduler..."
gcloud scheduler jobs delete "ai-trader-report" --location="$REGION" --quiet 2>/dev/null || true
gcloud scheduler jobs create http "ai-trader-report" \
    --location="$REGION" \
    --schedule="0 21 * * 1-5" \
    --time-zone="UTC" \
    --uri="${SERVICE_URL}/report" \
    --http-method=POST \
    --oidc-service-account-email="$SA_EMAIL" \
    --attempt-deadline="300s" \
    --description="AI Trading Bot — Daily learning report"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying deployment..."
gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --format='table(status.url, spec.template.spec.containers[0].resources.limits.memory, spec.template.metadata.annotations["autoscaling.knative.dev/minScale"], spec.template.metadata.annotations["autoscaling.knative.dev/maxScale"])'

echo ""
echo "=========================================="
echo " Deployment Complete!"
echo "=========================================="
echo ""
echo " Service URL: $SERVICE_URL"
echo ""
echo " Scheduler jobs:"
gcloud scheduler jobs list --location="$REGION" --format='table(name, schedule, state)' 2>/dev/null || true
echo ""
echo " Management:"
echo "   View logs:        gcloud run services logs read $SERVICE_NAME --region=$REGION"
echo "   Test trade:       curl -X POST ${SERVICE_URL}/trade -H 'Authorization: Bearer \$(gcloud auth print-identity-token)'"
echo "   Test health:      curl ${SERVICE_URL}/health"
echo "   Update env:       ./deploy/deploy-cloudrun.sh --env-only"
echo "   Redeploy:         ./deploy/deploy-cloudrun.sh"
echo "   Pause scheduler:  gcloud scheduler jobs pause ai-trader-cycle --location=$REGION"
echo "   Resume scheduler: gcloud scheduler jobs resume ai-trader-cycle --location=$REGION"
echo ""
echo " Cost estimate: \$0 / month (within free tier)"
echo ""
