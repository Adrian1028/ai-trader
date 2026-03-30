#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# AI Trading Bot — Management Script
# ══════════════════════════════════════════════════════════════════
# Usage:
#   ./deploy/manage.sh status    — Check bot status
#   ./deploy/manage.sh start     — Start the bot
#   ./deploy/manage.sh stop      — Stop the bot
#   ./deploy/manage.sh restart   — Restart the bot
#   ./deploy/manage.sh logs      — Follow live logs
#   ./deploy/manage.sh update    — Pull latest code and restart
#   ./deploy/manage.sh health    — Run health checks
#   ./deploy/manage.sh backup    — Backup data and state
# ══════════════════════════════════════════════════════════════════

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

# Detect deployment mode
if docker compose version &>/dev/null 2>&1 && [ -f docker-compose.yml ]; then
    MODE="docker"
elif systemctl is-enabled ai-trader &>/dev/null 2>&1; then
    MODE="systemd"
else
    MODE="manual"
fi

case "${1:-help}" in

    status)
        echo "=== AI Trading Bot Status ==="
        echo "Deploy mode: $MODE"
        echo ""
        if [ "$MODE" = "docker" ]; then
            docker compose ps
            echo ""
            echo "--- Container Stats ---"
            docker stats --no-stream ai-trader 2>/dev/null || echo "(not running)"
        elif [ "$MODE" = "systemd" ]; then
            sudo systemctl status ai-trader --no-pager
        else
            echo "Bot is not configured as a service."
            echo "Run manually: python src/main.py"
        fi
        ;;

    start)
        echo "Starting AI Trading Bot..."
        if [ "$MODE" = "docker" ]; then
            docker compose up -d --build
        elif [ "$MODE" = "systemd" ]; then
            sudo systemctl start ai-trader
        else
            echo "Starting in foreground (Ctrl+C to stop)..."
            source venv/bin/activate 2>/dev/null || true
            python src/main.py
        fi
        echo "Done. Use './deploy/manage.sh logs' to follow output."
        ;;

    stop)
        echo "Stopping AI Trading Bot (graceful shutdown)..."
        if [ "$MODE" = "docker" ]; then
            docker compose down
        elif [ "$MODE" = "systemd" ]; then
            sudo systemctl stop ai-trader
        else
            echo "Find and kill the process manually:"
            pgrep -af "python.*main.py" || echo "(not running)"
        fi
        echo "Done."
        ;;

    restart)
        echo "Restarting AI Trading Bot..."
        if [ "$MODE" = "docker" ]; then
            docker compose restart
        elif [ "$MODE" = "systemd" ]; then
            sudo systemctl restart ai-trader
        else
            echo "Kill and restart manually."
        fi
        echo "Done."
        ;;

    logs)
        echo "=== Live Logs (Ctrl+C to exit) ==="
        if [ "$MODE" = "docker" ]; then
            docker compose logs -f --tail=100
        elif [ "$MODE" = "systemd" ]; then
            journalctl -u ai-trader -f -n 100
        else
            tail -f "$APP_DIR/data/trading_bot.log" 2>/dev/null || \
            tail -f "$APP_DIR/logs/bot-stdout.log" 2>/dev/null || \
            echo "No log file found."
        fi
        ;;

    update)
        echo "=== Updating AI Trading Bot ==="

        # Pull latest code
        if [ -d .git ]; then
            echo "Pulling latest code..."
            git pull
        else
            echo "Not a git repo. Copy files manually."
        fi

        # Rebuild and restart
        if [ "$MODE" = "docker" ]; then
            echo "Rebuilding Docker image..."
            docker compose up -d --build
        elif [ "$MODE" = "systemd" ]; then
            echo "Reinstalling dependencies..."
            source venv/bin/activate
            pip install -q -r requirements.txt
            echo "Restarting service..."
            sudo systemctl restart ai-trader
        fi
        echo "Update complete."
        ;;

    health)
        echo "=== Health Check ==="

        # Check container/process
        if [ "$MODE" = "docker" ]; then
            RUNNING=$(docker ps --filter name=ai-trader --format '{{.Status}}' 2>/dev/null)
            if [ -n "$RUNNING" ]; then
                echo "[OK] Container: $RUNNING"
            else
                echo "[FAIL] Container not running!"
            fi
            # Memory usage
            MEM=$(docker stats --no-stream --format '{{.MemUsage}}' ai-trader 2>/dev/null)
            echo "[INFO] Memory: ${MEM:-N/A}"
        elif [ "$MODE" = "systemd" ]; then
            if systemctl is-active ai-trader &>/dev/null; then
                echo "[OK] Service is active"
            else
                echo "[FAIL] Service is not running!"
            fi
        fi

        # Check data freshness
        if [ -f "$APP_DIR/data/trading_bot.log" ]; then
            LAST_LOG=$(tail -1 "$APP_DIR/data/trading_bot.log" | cut -d'|' -f1 | xargs)
            echo "[INFO] Last log: $LAST_LOG"
        fi

        # Check learning report
        if [ -f "$APP_DIR/data/learning_report.json" ]; then
            echo "[INFO] Learning report exists"
            python3 -c "
import json
with open('$APP_DIR/data/learning_report.json') as f:
    d = json.load(f)
    print(f'  Episodes: {d.get(\"episodes_stored\", 0)}')
    print(f'  OPRO gen: {d.get(\"opro_generation\", 0)}')
    stats = d.get('audit_stats', {})
    print(f'  Trades:   {stats.get(\"total_trades\", 0)}')
    print(f'  Win rate: {stats.get(\"win_rate\", 0):.1%}')
" 2>/dev/null || echo "  (could not parse)"
        fi

        # Check disk usage
        echo "[INFO] Disk usage:"
        du -sh "$APP_DIR/data" 2>/dev/null || echo "  data: N/A"
        du -sh "$APP_DIR/logs" 2>/dev/null || echo "  logs: N/A"
        ;;

    backup)
        echo "=== Backing up data ==="
        BACKUP_DIR="$APP_DIR/backups"
        mkdir -p "$BACKUP_DIR"
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_FILE="$BACKUP_DIR/ai-trader-backup-$TIMESTAMP.tar.gz"

        tar -czf "$BACKUP_FILE" \
            -C "$APP_DIR" \
            data/ \
            .env \
            2>/dev/null

        echo "Backup saved: $BACKUP_FILE"
        echo "Size: $(du -sh "$BACKUP_FILE" | cut -f1)"

        # Keep only last 5 backups
        ls -t "$BACKUP_DIR"/ai-trader-backup-*.tar.gz 2>/dev/null | tail -n +6 | xargs rm -f
        echo "Kept last 5 backups."
        ;;

    *)
        echo "AI Trading Bot Management"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  status    Show bot status and resource usage"
        echo "  start     Start the trading bot"
        echo "  stop      Stop the trading bot (graceful)"
        echo "  restart   Restart the trading bot"
        echo "  logs      Follow live logs"
        echo "  update    Pull latest code and restart"
        echo "  health    Run health checks"
        echo "  backup    Backup data and state"
        echo ""
        echo "Deploy mode: $MODE"
        ;;
esac
