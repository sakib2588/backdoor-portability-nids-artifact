#!/usr/bin/env bash
# Event stream for the sequential queue. One line per transition, then exit.
#
# Coverage matters more than tidiness here. This emits on every terminal state, not only on
# success: a job's END line carries its exit code, and the queue dying without writing QUEUE_DONE
# is itself reported. A watcher that only greps for success is silent through a crash, and silence
# is indistinguishable from still-running.
set -u
cd "$(dirname "$0")/.."
STATUS="logs/queue_status.txt"
prev=0
while true; do
    if [ -f "$STATUS" ]; then
        cur=$(wc -l < "$STATUS")
        if [ "$cur" -gt "$prev" ]; then
            sed -n "$((prev + 1)),${cur}p" "$STATUS"
            prev=$cur
        fi
        if grep -q QUEUE_DONE "$STATUS"; then
            exit 0
        fi
        # The queue holds logs/.queue.lock for its whole life. If the lock is free and the queue
        # never wrote QUEUE_DONE, it died: killed, OOM, or the machine went down.
        if [ "$prev" -gt 0 ] && ! fuser logs/.queue.lock >/dev/null 2>&1; then
            echo "QUEUE_DIED without QUEUE_DONE -- last line: $(tail -1 "$STATUS")"
            exit 1
        fi
    fi
    sleep 20
done
