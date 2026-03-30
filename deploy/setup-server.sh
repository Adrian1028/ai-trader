#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# AI Trading Bot — Cloud Server Setup Script
# ══════════════════════════════════════════════════════════════════
# Supports: Oracle Cloud (ARM/AMD), AWS, GCP, any Ubuntu/Debian VPS
#
# Usage:
#   1. SSH into your server
#   2. Upload this script:  scp deploy/setup-server.sh user@server:~
#   3. Run:  chmod +x setup-server.sh && ./setup-server.sh
#
# What it does:
#   - Installs Docker (if not installed)
#   - Clones your repo (or copies files)
#   - Sets up auto-restart via systemd
#   - Configures log rotation
#   - Opens no ports (bot is outbound-only, no attack surface)
# ══════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=========================================="
echo " AI Trading Bot — Server Setup"
echo "=========================================="

# ── 1. Install Docker ────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[1/5] Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    sudo systemctl enable docker
    sudo systemctl start docker
    echo "  Docker installed. You may need to log out and back in."
else
    echo "[1/5] Docker already installed: $(docker --version)"
fi

# Install docker compose plugin if not present
if ! docker compose version &>/dev/null 2>&1; then
    echo "  Installing docker compose plugin..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-compose-plugin
fi

# ── 2. Create app directory ──────────────────────────────────────
APP_DIR="$HOME/ai-trader"
echo "[2/5] Setting up app directory: $APP_DIR"

mkdir -p "$APP_DIR"/{data,logs}

# If running from the repo, copy files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$REPO_DIR/src/main.py" ]; then
    echo "  Copying project files from $REPO_DIR..."
    cp -r "$REPO_DIR/src" "$APP_DIR/"
    cp -r "$REPO_DIR/config" "$APP_DIR/"
    cp "$REPO_DIR/Dockerfile" "$APP_DIR/"
    cp "$REPO_DIR/Dockerfile.slim" "$APP_DIR/" 2>/dev/null || true
    cp "$REPO_DIR/docker-compose.yml" "$APP_DIR/"
    cp "$REPO_DIR/requirements.txt" "$APP_DIR/"
    cp "$REPO_DIR/.env.example" "$APP_DIR/"

    if [ -f "$REPO_DIR/.env" ]; then
        cp "$REPO_DIR/.env" "$APP_DIR/.env"
        echo "  .env copied (contains API keys — keep secure!)"
    else
        echo "  WARNING: No .env file found. Copy .env.example to .env and fill in keys."
    fi
else
    echo "  NOTE: Run this script from the project directory, or"
    echo "  manually copy project files to $APP_DIR"
fi

# ── 3. Setup systemd service for auto-restart ────────────────────
echo "[3/5] Creating systemd service for auto-restart..."

sudo tee /etc/systemd/system/ai-trader.service > /dev/null << 'SYSTEMD_EOF'
[Unit]
Description=AI Trading Bot (Docker)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=%h/ai-trader
ExecStart=/usr/bin/docker compose up -d --build
ExecStop=/usr/bin/docker compose down
ExecReload=/usr/bin/docker compose restart
TimeoutStartSec=120
TimeoutStopSec=60

# Auto-restart on failure
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

sudo systemctl daemon-reload
sudo systemctl enable ai-trader.service
echo "  Systemd service created: ai-trader.service"

# ── 4. Setup log rotation ────────────────────────────────────────
echo "[4/5] Configuring log rotation..."

sudo tee /etc/logrotate.d/ai-trader > /dev/null << 'LOGROTATE_EOF'
/home/*/ai-trader/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
LOGROTATE_EOF

# ── 5. Setup health check cron ───────────────────────────────────
echo "[5/5] Setting up health check cron job..."

# Create health check script
mkdir -p "$APP_DIR/deploy"
cat > "$APP_DIR/deploy/healthcheck.sh" << 'HEALTH_EOF'
#!/usr/bin/env bash
# Check if the trading bot container is running
# Called by cron every 5 minutes

CONTAINER="ai-trader"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "$(date) — Container $CONTAINER is NOT running! Restarting..."
    cd "$HOME/ai-trader"
    docker compose up -d
    echo "$(date) — Restart command issued."
fi
HEALTH_EOF

chmod +x "$APP_DIR/deploy/healthcheck.sh"

# Add cron job (every 5 minutes)
CRON_CMD="*/5 * * * * $APP_DIR/deploy/healthcheck.sh >> $APP_DIR/logs/healthcheck.log 2>&1"
(crontab -l 2>/dev/null | grep -v "healthcheck.sh"; echo "$CRON_CMD") | crontab -

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo " Next steps:"
echo "   1. Edit API keys:     nano $APP_DIR/.env"
echo "   2. Start the bot:     cd $APP_DIR && docker compose up -d --build"
echo "   3. Check logs:        docker compose logs -f"
echo "   4. Check status:      docker ps"
echo ""
echo " Management commands:"
echo "   Stop bot:             docker compose down"
echo "   Restart bot:          docker compose restart"
echo "   View recent logs:     docker compose logs --tail=50"
echo "   Update & rebuild:     docker compose up -d --build"
echo ""
echo " Auto-restart: enabled (systemd + cron healthcheck)"
echo " Log rotation: enabled (7 days, compressed)"
echo ""
