#!/usr/bin/env bash
# Idempotent local Ollama + Qwen 2.5 7B setup for topic labeling.
#
# - Installs Ollama if not present.
# - Ensures the daemon is reachable on 127.0.0.1:11434.
# - Pulls qwen2.5:7b (default ~Q4_K_M quant, ~4.7 GB on disk).
# - Verifies a hello-world completion.
#
# Safe to re-run.

set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODEL="${RAG_DPR_BLOG_OLLAMA_MODEL:-qwen2.5:14b}"

log() { printf '[ollama_setup] %s\n' "$*"; }

# 1. Install Ollama (Linux).
if ! command -v ollama >/dev/null 2>&1; then
  log "Ollama not found — installing via official script."
  curl -fsSL https://ollama.com/install.sh | sh
else
  log "Ollama already installed: $(ollama --version 2>&1 | head -n1)"
fi

# 2. Make sure the daemon is up. The installer typically registers a
#    systemd unit; if not, fall back to starting it in the background.
if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files 2>/dev/null | grep -q '^ollama\.service'; then
    log "Starting ollama.service via systemd."
    sudo systemctl start ollama || true
  else
    log "Starting ollama serve in the background (logs → /tmp/ollama.log)."
    nohup ollama serve >/tmp/ollama.log 2>&1 &
  fi
  # Give the daemon a few seconds to bind.
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl -fsS "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  log "ERROR: Ollama daemon did not come up at ${OLLAMA_HOST}."
  exit 1
fi
log "Ollama daemon is up at ${OLLAMA_HOST}."

# 3. Pull the model. Ollama's `pull` is idempotent — it short-circuits
#    when layers are already cached.
log "Pulling ${MODEL} ..."
ollama pull "${MODEL}"

# 4. Smoke test.
log "Smoke-testing ${MODEL} ..."
RESP="$(curl -fsS "${OLLAMA_HOST}/api/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"prompt\":\"Reply with exactly: OK\",\"stream\":false,\"options\":{\"temperature\":0}}" \
  | python3 -c 'import sys, json; print(json.load(sys.stdin).get("response","").strip())')"

log "Model said: ${RESP}"
log "Done."
