#!/usr/bin/env bash
set -euo pipefail

# Repo location on EC2
BASE_DIR="/home/ec2-user/tradingbot/crypto-prediction"

# Ensure cron has a usable PATH (adjust if your poetry/python live elsewhere)
export PATH="/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# Poetry binary (absolute path is safest under cron)
POETRY_BIN="/home/ec2-user/.local/bin/poetry"

# Log file (absolute path; keep logs inside the repo by default)
LOG_FILE="$BASE_DIR/trading_bot_crypto_cron.log"

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

echo "$(date) - Starting run_crypto.sh (cwd=$(pwd))" >> "$LOG_FILE" 2>&1

# ---------- INSTALL DEPENDENCIES ----------
echo "$(date) - Installing poetry dependencies (if missing)..." >> "$LOG_FILE" 2>&1
"$POETRY_BIN" install --no-interaction >> "$LOG_FILE" 2>&1

# ---------- GET VENV PYTHON ----------
VENV_PYTHON=$("$POETRY_BIN" env info -p 2>/dev/null)/bin/python
if [ ! -f "$VENV_PYTHON" ]; then
    echo "$(date) - ERROR: Poetry virtualenv Python not found!" >> "$LOG_FILE" 2>&1
    exit 1
fi
echo "$(date) - Using virtualenv python: $VENV_PYTHON" >> "$LOG_FILE" 2>&1

# ---------- RUN THE BOT (CRYPTO ONLY) ----------
echo "$(date) - Running crypto bot..." >> "$LOG_FILE" 2>&1
"$VENV_PYTHON" -u app/core/main.py --asset=crypto >> "$LOG_FILE" 2>&1
BOT_EXIT_CODE=$?

if [ $BOT_EXIT_CODE -eq 0 ]; then
    echo "$(date) - Crypto bot finished successfully" >> "$LOG_FILE" 2>&1
else
    echo "$(date) - Crypto bot exited with code $BOT_EXIT_CODE" >> "$LOG_FILE" 2>&1
fi

echo "$(date) - Run completed" >> "$LOG_FILE" 2>&1
echo "===================================================" >> "$LOG_FILE" 2>&1


