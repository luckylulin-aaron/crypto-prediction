 ---------- INSTALL DEPENDENCIES ----------
echo "$(date) - Installing poetry dependencies (if missing)..." >> "$LOG_FILE" 2>&1
/home/ec2-user/.local/bin/poetry install >> "$LOG_FILE" 2>&1

# ---------- GET VENV PYTHON ----------
VENV_PYTHON=$(/home/ec2-user/.local/bin/poetry env info -p 2>/dev/null)/bin/python
if [ ! -f "$VENV_PYTHON" ]; then
    echo "$(date) - ERROR: Poetry virtualenv Python not found!" >> "$LOG_FILE" 2>&1
    exit 1
fi
echo "$(date) - Using virtualenv python: $VENV_PYTHON" >> "$LOG_FILE" 2>&1

# ---------- RUN THE BOT ----------
echo "$(date) - Running bot..." >> "$LOG_FILE" 2>&1
"$VENV_PYTHON" app/core/main.py >> "$LOG_FILE" 2>&1
BOT_EXIT_CODE=$?

if [ $BOT_EXIT_CODE -eq 0 ]; then
    echo "$(date) - Bot finished successfully" >> "$LOG_FILE" 2>&1
else
    echo "$(date) - Bot exited with code $BOT_EXIT_CODE" >> "$LOG_FILE" 2>&1
fi

echo "$(date) - Run completed" >> "$LOG_FILE" 2>&1
echo "===================================================" >> "$LOG_FILE" 2>&1