#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# AI Trading Bot — Direct Python Setup (No Docker)
# ══════════════════════════════════════════════════════════════════
# For VMs where Docker is too heavy or not installed.
# Uses Python venv + systemd for auto-restart.
#
# Usage:
#   1. SSH into your server
#   2. Clone/copy the project to ~/ai-trader
#   3. Run: chmod +x deploy/setup-python.sh && deploy/setup-python.sh
# ══════════════════════════════════════════════════════════════════

set -euo pipefail

echo "=========================================="
echo " AI Trading Bot — Python Setup (No Docker)"
echo "=========================================="

APP_DIR="$HOME/ai-trader"

# ── 1. Install Python 3.11+ ──────────────────────────────────────
echo "[1/5] Checking Python..."
if command -v python3.11 &>/dev/null; then
    PYTHON=python3.11
elif command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if python3 -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
        PYTHON=python3
    else
        echo "  Python 3.11+ required. Current: $PY_VER"
        echo "  Installing Python 3.11..."
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip
        PYTHON=python3.11
    fi
else
    echo "  Python not found. Installing..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip
    PYTHON=python3.11
fi
echo "  Using: $($PYTHON --version)"

# ── 2. Create virtual environment ────────────────────────────────
echo "[2/5] Creating virtual environment..."
cd "$APP_DIR"

if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
fi
source venv/bin/activate

# Install only production dependencies (skip test/dashboard to save RAM)
pip install --no-cache-dir -q \
    aiohttp>=3.9 \
    numpy>=1.26 \
    pandas>=2.1 \
    pydantic>=2.0 \
    apscheduler>=3.10 \
    python-dotenv>=1.0 \
    google-genai>=1.0

echo "  Dependencies installed."

# ── 3. Create directories ────────────────────────────────────────
echo "[3/5] Creating data directories..."
mkdir -p data logs

# ── 4. Create systemd service ────────────────────────────────────
echo "[4/5] Creating systemd service..."

sudo tee /etc/systemd/system/ai-trader.service > /dev/null << SYSTEMD_EOF
[Unit]
Description=AI Trading Bot (Python)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment=PYTHONPATH=$APP_DIR
ExecStart=$APP_DIR/venv/bin/python src/main.py
Restart=always
RestartSec=30

# Memory limit (prevents runaway memory on free-tier VMs)
MemoryMax=400M
MemoryHigh=300M

# Graceful shutdown
TimeoutStopSec=30
KillSignal=SIGINT

# Logging
StandardOutput=append:$APP_DIR/logs/bot-stdout.log
StandardError=append:$APP_DIR/logs/bot-stderr.log

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF

sudo systemctl daemon-reload
sudo systemctl enable ai-trader.service
echo "  Service created: ai-trader.service"

# ── 5. Log rotation ──────────────────────────────────────────────
echo "[5/5] Setting up log rotation..."

sudo tee /etc/logrotate.d/ai-trader > /dev/null << LOGROTATE_EOF
$APP_DIR/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
LOGROTATE_EOF

echo ""
echo "=========================================="
echo " Setup Complete!"
echo "=========================================="
echo ""
echo " Next steps:"
echo "   1. Edit API keys:     nano $APP_DIR/.env"
echo "   2. Test run:          cd $APP_DIR && source venv/bin/activate && python src/main.py"
echo "   3. Start as service:  sudo systemctl start ai-trader"
echo "   4. Check status:      sudo systemctl status ai-trader"
echo "   5. View logs:         journalctl -u ai-trader -f"
echo ""
echo " Management:"
echo "   Stop:       sudo systemctl stop ai-trader"
echo "   Restart:    sudo systemctl restart ai-trader"
echo "   Logs:       tail -f $APP_DIR/logs/bot-stdout.log"
echo "   Update:     git pull && sudo systemctl restart ai-trader"
echo ""
echo " Memory limit: 400MB (fits free-tier VMs)"
echo " Auto-restart: enabled (systemd Restart=always)"
echo ""
