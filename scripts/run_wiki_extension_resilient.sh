#!/usr/bin/env bash
# Two-phase resilient runner for the wiki extension.
#   Phase A: top up OpenAI + Mistral generators, then validate + compare
#            with all 4 providers (incl. Anthropic).
#            A watcher SIGTERMs the pipeline when Anthropic spend hits the cap.
#   Phase B: continue validate + compare with the remaining 3 providers
#            (no Anthropic). No budget watcher in this phase.
# Both phases auto-restart on transient crashes (e.g. WiFi outages).

set -u
cd "$(dirname "$0")/.."

PY="$HOME/.pyenv/versions/3.11.7/bin/python"
RUN_ID="20260625T062938Z_wiki_extension_bcb6af9"
ITEMS="configs/wiki_extension_all.txt"

CFG_A="configs/wiki_extension_phaseA.yaml"
CFG_B="configs/wiki_extension_phaseB.yaml"
STAGES_A="generate,validate,compare,analyze"
# Validate is skipped on resume - we have 100k+ validations already and the
# remaining ~7% would take hours due to OpenAI daily-limit / Mistral throttle.
STAGES_B="compare,analyze"

# Hard absolute cap on Anthropic total cost (USD) across the whole costs.csv.
# Original baseline before the 5 EUR topup was $3.0629; the user wants
# to leave ~1 EUR remaining of the 5 EUR. Cap = $3.06 + $4.10 = $7.16
# (~3.9 EUR new spend at 1.05 EUR/USD).
HARD_MAX_USD="7.1629"

COSTS="results/${RUN_ID}/costs.csv"

mkdir -p logs
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/wiki_resume_${TS}.log"
FLAG="/tmp/wiki_budget_hit_${RUN_ID}"
rm -f "$FLAG"

echo "[wrapper] start  hard_max_anthropic=\$${HARD_MAX_USD}  log=${LOG}" | tee -a "$LOG"

anthropic_total() {
  python3 -c "
import csv
t=0.0
try:
    for r in csv.DictReader(open('${COSTS}')):
        if r.get('provider')=='anthropic':
            t += float(r.get('cost_usd') or 0)
except FileNotFoundError:
    pass
print(f'{t:.4f}')
"
}

over_budget() {
  cur=$(anthropic_total)
  awk -v a="$cur" -v m="$HARD_MAX_USD" 'BEGIN{exit !(a >= m)}'
}

watcher() {
  while sleep 20; do
    if over_budget; then
      cur=$(anthropic_total)
      echo "[watcher] BUDGET HIT: anthropic=\$${cur} >= \$${HARD_MAX_USD} - terminating pipeline" | tee -a "$LOG"
      touch "$FLAG"
      pkill -TERM -f "run_level1.py" 2>/dev/null
      sleep 5
      pkill -KILL -f "run_level1.py" 2>/dev/null
      return 0
    fi
  done
}

cleanup() {
  jobs -p | xargs -r kill 2>/dev/null
  exit
}
trap cleanup INT TERM

run_phase() {
  local cfg=$1
  local stages=$2
  local phase_name=$3
  local watch=$4
  local max_restarts=20
  local attempt=0
  local WATCHER_PID=""

  if [ "$watch" = "yes" ]; then
    rm -f "$FLAG"
    watcher &
    WATCHER_PID=$!
    echo "[wrapper] watcher_pid=$WATCHER_PID  (phase=$phase_name)" | tee -a "$LOG"
  fi

  while [ $attempt -lt $max_restarts ]; do
    attempt=$((attempt+1))
    if [ "$watch" = "yes" ] && { [ -f "$FLAG" ] || over_budget; }; then
      cur=$(anthropic_total)
      echo "[wrapper] $phase_name: budget exhausted (\$${cur}) - ending phase" | tee -a "$LOG"
      break
    fi
    cur=$(anthropic_total)
    echo "[wrapper] $phase_name attempt $attempt/$max_restarts  anthropic=\$${cur}  $(date -u +%H:%M:%SZ)" | tee -a "$LOG"

    "$PY" scripts/run_level1.py \
      --config "$cfg" --items-file "$ITEMS" --run-id "$RUN_ID" --stages "$stages" >> "$LOG" 2>&1
    rc=$?
    cur=$(anthropic_total)
    echo "[wrapper] $phase_name exit=$rc  anthropic=\$${cur}  $(date -u +%H:%M:%SZ)" | tee -a "$LOG"

    if [ "$watch" = "yes" ] && [ -f "$FLAG" ]; then
      echo "[wrapper] $phase_name: watcher fired - ending phase" | tee -a "$LOG"
      break
    fi
    if [ $rc -eq 0 ]; then
      echo "[wrapper] $phase_name: success" | tee -a "$LOG"
      break
    fi
    if tail -80 "$LOG" | grep -qiE "FatalLLMError|insufficient_quota|invalid_api_key|HTTP/[0-9.]+ 40[13]"; then
      echo "[wrapper] $phase_name: fatal API error - ending phase" | tee -a "$LOG"
      break
    fi
    echo "[wrapper] $phase_name: transient crash, sleeping 60s..." | tee -a "$LOG"
    sleep 60
  done

  if [ -n "$WATCHER_PID" ]; then
    kill "$WATCHER_PID" 2>/dev/null
    wait "$WATCHER_PID" 2>/dev/null || true
  fi
}

echo "[wrapper] === PHASE A: 4 providers, watching Anthropic budget ===" | tee -a "$LOG"
run_phase "$CFG_A" "$STAGES_A" "phaseA" "yes"

echo "[wrapper] === PHASE B: 3 providers (no Anthropic), no budget watcher ===" | tee -a "$LOG"
run_phase "$CFG_B" "$STAGES_B" "phaseB" "no"

final=$(anthropic_total)
echo "[wrapper] done  final_anthropic=\$${final}" | tee -a "$LOG"
