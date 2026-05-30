#!/bin/bash

set -euo pipefail

# Usage:
#   bash monitor_training.sh <job_id> [--watch] [--interval 30]

if [[ $# -lt 1 ]]; then
    echo "Usage: bash monitor_training.sh <job_id> [--watch] [--interval 30]"
    exit 1
fi

JOB_ID="$1"
shift || true

WATCH=false
INTERVAL=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch)
            WATCH=true
            shift
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

LOG_DIR_A="$REPO_DIR/layout-model-training/logs"
LOG_DIR_B="$REPO_DIR/logs"

OUT_LOG_A="$LOG_DIR_A/yb_text_clean_${JOB_ID}.out"
ERR_LOG_A="$LOG_DIR_A/yb_text_clean_${JOB_ID}.err"
OUT_LOG_B="$LOG_DIR_B/yb_text_clean_${JOB_ID}.out"
ERR_LOG_B="$LOG_DIR_B/yb_text_clean_${JOB_ID}.err"

pick_log() {
    local a="$1"
    local b="$2"
    if [[ -f "$a" ]]; then
        echo "$a"
    elif [[ -f "$b" ]]; then
        echo "$b"
    else
        echo "$a"
    fi
}

show_once() {
    local now queue_line state_line state reason out_log err_log

    now="$(date '+%Y-%m-%d %H:%M:%S')"
    out_log="$(pick_log "$OUT_LOG_A" "$OUT_LOG_B")"
    err_log="$(pick_log "$ERR_LOG_A" "$ERR_LOG_B")"

    echo "=========================================="
    echo "Training Monitor | ${now}"
    echo "Job ID: ${JOB_ID}"
    echo "=========================================="

    queue_line="$(squeue -j "$JOB_ID" -h -o '%T|%M|%R|%P' 2>/dev/null || true)"
    state_line="$(sacct -j "$JOB_ID" --format=State,ExitCode,Elapsed,Start,End -n -P 2>/dev/null | head -n 1 || true)"

    if [[ -n "$queue_line" ]]; then
        IFS='|' read -r state runtime reason partition <<< "$queue_line"
        echo "State: ${state}"
        echo "Partition: ${partition}"
        echo "Runtime: ${runtime}"
        echo "Reason: ${reason}"
    elif [[ -n "$state_line" ]]; then
        echo "Final: ${state_line}"
        state="${state_line%%|*}"
    else
        echo "State: UNKNOWN (job not visible yet in squeue/sacct)"
        state="UNKNOWN"
    fi

    echo ""
    echo "Logs:"
    echo "  OUT: ${out_log}"
    echo "  ERR: ${err_log}"
    echo ""

    if [[ -f "$out_log" ]]; then
        echo "Recent OUT:"
        tail -n 20 "$out_log" | grep -E 'eta:|iter:|total_loss|loss_|bbox/AP|Average Precision|Total training time|Finished training' || tail -n 20 "$out_log"
    else
        echo "Recent OUT: (not created yet)"
    fi

    echo ""

    if [[ -f "$err_log" ]]; then
        if [[ -s "$err_log" ]]; then
            echo "Recent ERR:"
            tail -n 20 "$err_log"
        else
            echo "Recent ERR: (empty)"
        fi
    else
        echo "Recent ERR: (not created yet)"
    fi

    MODEL_PATH="$REPO_DIR/layout-model-training/outputs/yearbook-text-only/fast_rcnn_R_50_FPN/model_final.pth"
    if [[ -f "$MODEL_PATH" ]]; then
        echo ""
        echo "Model found: ${MODEL_PATH}"
    fi

    echo ""

    case "$state" in
        COMPLETED)
            return 10
            ;;
        FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
            return 20
            ;;
        *)
            return 0
            ;;
    esac
}

if [[ "$WATCH" == false ]]; then
    show_once
    exit 0
fi

while true; do
    show_once
    rc=$?
    if [[ $rc -eq 10 ]]; then
        printf '\a'
        echo "✅ Job ${JOB_ID} completed."
        exit 0
    elif [[ $rc -eq 20 ]]; then
        printf '\a'
        echo "❌ Job ${JOB_ID} ended with failure state."
        exit 1
    fi
    sleep "$INTERVAL"
done
