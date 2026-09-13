#!/usr/bin/env bash
# [SERVER] Collects the recorder's reports and pushes them to the repository.
#
# Why this exists: the reports live in out/ and the logs are *.log, both of
# which are gitignored, so nothing about a running recorder is readable from
# outside the machine. The Mac side reaches this server through git and nothing
# else - no SSH out, no dashboard token - so a report that stays on disk is a
# report nobody reads.
#
# Read-only apart from snapshot/. The recorder is not touched, no service is
# restarted, no data file is rewritten.
#
#     bash out.sh          reports plus the tail of both logs
#     bash out.sh --full   also the whole cost report without truncation

set -u
cd "$(dirname "$0")" || exit 1
OUT=snapshot
mkdir -p "$OUT"

echo "=== out.sh at $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

echo "==> reports"
# Both tools write their own out/*.txt through tee; regenerate so the snapshot
# carries fresh numbers rather than whatever was last run by hand.
./costmap    > /dev/null 2>&1 && cp -f out/costmap.txt "$OUT/costmap.txt" 2>/dev/null
./report     > /dev/null 2>&1 && cp -f out/report.txt  "$OUT/report.txt"  2>/dev/null
echo "    costmap.txt $(wc -l < "$OUT/costmap.txt" 2>/dev/null || echo 0) lines"
echo "    report.txt  $(wc -l < "$OUT/report.txt"  2>/dev/null || echo 0) lines"

echo "==> logs"
tail -n 120 recorder.log > "$OUT/recorder.log.txt" 2>/dev/null
tail -n 120 costmap.log  > "$OUT/costmap.log.txt"  2>/dev/null

echo "==> services and disk"
{
    echo "taken: $(date -u '+%Y-%m-%d %H:%M:%S') UTC"
    echo "commit: $(git log -1 --format='%h %ad %s' --date=short)"
    echo
    for unit in orderbook-recorder orderbook-costmap orderbook-dashboard; do
        if systemctl list-unit-files | grep -q "^${unit}"; then
            printf '%-22s %s  restarts: %s\n' "$unit" \
                "$(systemctl is-active "$unit" 2>/dev/null)" \
                "$(systemctl show "$unit" -p NRestarts --value 2>/dev/null)"
        fi
    done
    echo
    echo "raw capture : $(du -sh data      2>/dev/null | cut -f1)"
    echo "cost map    : $(du -sh data_cost 2>/dev/null | cut -f1)"
    echo
    df -h . | tail -1
    echo
    echo "days on disk:"
    ls data      2>/dev/null | tail -5 | sed 's/^/  raw  /'
    ls data_cost 2>/dev/null | tail -5 | sed 's/^/  cost /'
} > "$OUT/status.txt" 2>&1

echo "==> pushing to the repository"
git add -A "$OUT" >/dev/null 2>&1

if git diff --cached --quiet; then
    echo "    nothing changed since the last snapshot"
    exit 0
fi

# Identity per command: this server has none configured globally, and a commit
# without it fails quietly while the script reports success.
if ! git -c user.name="orderbook-recorder server" \
        -c user.email="server@localhost" \
        commit -q -m "Recorder snapshot $(date -u '+%Y-%m-%d %H:%M UTC')"; then
    echo
    echo "ERROR: commit failed, see above. Nothing was pushed."
    exit 1
fi

# Merge, not rebase: rebase refuses to run with a dirty tree, and a running
# recorder keeps writing its data files by construction.
echo "    syncing with the remote first"
if ! git -c user.name="orderbook-recorder server" \
        -c user.email="server@localhost" \
        pull --no-rebase --no-edit -q; then
    echo
    echo "ERROR: could not merge the remote. The snapshot is committed locally,"
    echo "nothing is lost. Resolve it with:"
    echo "  cd /root/orderbook-recorder && git status"
    exit 1
fi

if ! git push -q; then
    echo
    echo "ERROR: push failed. The commit was made locally, nothing is lost."
    exit 1
fi

echo
echo "========================================"
echo "  DONE - snapshot is in the repository"
echo "========================================"
