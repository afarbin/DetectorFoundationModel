#!/bin/bash
# slac_pull.sh — stream SuperHJD ntuples SLAC -> UTA directly, no staging.
#
# Run ON the UTA server (cn-1e1901), which can reach the SLAC jump host
# outbound (verified 2026-08-24: s3dflogin.slac.stanford.edu:22 open).
# rsync runs over ssh with ProxyJump, so bytes flow
#   SLAC filesystem -> (inner node ->) jump host -> UTA /storage
# in one stream; nothing is staged at SLAC or in home directories.
#
# Duo/MFA: the `auth` step opens one master connection (ControlMaster) that
# is kept alive 12h; every later ssh/rsync multiplexes over it with no
# further prompts.
#
# Usage:
#   ./slac_pull.sh auth
#   ./slac_pull.sh list <remote_dir>
#   ./slac_pull.sh pull <remote_dir> <dest_subdir> [--max-files N]
#                       [--files listfile] [--bwlimit KBPS] [--dry-run]
#
# Examples:
#   ./slac_pull.sh pull /sdf/data/atlas/.../vbf_container vbf_hinv
#   ./slac_pull.sh pull /sdf/data/atlas/.../ttbar_container ttbar --max-files 150
#
# Edit the config block for your account / topology.

set -euo pipefail

# ---- config (edit me) -------------------------------------------------------
SLAC_USER="${SLAC_USER:-mxg1065}"          # your SLAC S3DF username
JUMP="${JUMP:-s3dflogin.slac.stanford.edu}"
INNER="${INNER:-}"                          # interior node — REQUIRED for the
                                            # SuperHJD data (not visible from
                                            # the login node); set to the
                                            # interactive node you use
DEST_BASE="${DEST_BASE:-/storage/mxg1065/superhjd}"
RETRIES=5
CM_SOCK="$HOME/.ssh/cm-slac-%r@%h:%p"
# -----------------------------------------------------------------------------

SSH_BASE=(ssh -o "ControlMaster=auto" -o "ControlPath=$CM_SOCK" \
              -o "ControlPersist=12h" -o "ServerAliveInterval=30")
if [[ -n "$INNER" ]]; then
    TARGET="$SLAC_USER@$INNER"
    SSH_CMD=("${SSH_BASE[@]}" -o "ProxyJump=$SLAC_USER@$JUMP")
else
    TARGET="$SLAC_USER@$JUMP"
    SSH_CMD=("${SSH_BASE[@]}")
fi

usage() { sed -n '2,25p' "$0"; exit 1; }
[[ $# -ge 1 ]] || usage
cmd="$1"; shift || true

case "$cmd" in
auth)
    # opens the master connection: expect one password/Duo prompt, then done
    "${SSH_CMD[@]}" "$TARGET" 'echo "authenticated on $(hostname)"'
    ;;

list)
    [[ $# -ge 1 ]] || usage
    rdir="$1"
    mkdir -p "$DEST_BASE"
    out="$DEST_BASE/manifest_$(basename "$rdir")_$(date +%Y%m%d).txt"
    "${SSH_CMD[@]}" "$TARGET" "ls -la '$rdir'" | tee "$out"
    echo "manifest saved: $out"
    ;;

pull)
    [[ $# -ge 2 ]] || usage
    rdir="$1"; sub="$2"; shift 2
    max_files=0; listfile=""; bwlimit=""; dry=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --max-files) max_files="$2"; shift 2;;
            --files)     listfile="$2";  shift 2;;
            --bwlimit)   bwlimit="--bwlimit=$2"; shift 2;;
            --dry-run)   dry="--dry-run"; shift;;
            *) echo "unknown option $1"; usage;;
        esac
    done
    dest="$DEST_BASE/$sub"
    mkdir -p "$dest"

    # Build the file list on the remote side (root files only), newest last,
    # excluding anything already fully present locally (size match).
    tmplist=$(mktemp)
    if [[ -n "$listfile" ]]; then
        cp "$listfile" "$tmplist"
    else
        "${SSH_CMD[@]}" "$TARGET" \
            "cd '$rdir' && find . -maxdepth 2 -name '*.root*' -type f -printf '%P\n' | sort" \
            > "$tmplist"
    fi
    if [[ "$max_files" -gt 0 ]]; then
        head -n "$max_files" "$tmplist" > "$tmplist.cut" && mv "$tmplist.cut" "$tmplist"
    fi
    n=$(wc -l < "$tmplist")
    echo "pulling $n files from $TARGET:$rdir -> $dest"

    # Resumable, verified stream; retry on transient network failure.
    attempt=1
    # note: no --ignore-existing — rsync's size+mtime quick-check skips
    # complete files but still resumes interrupted ones (--partial --inplace)
    until rsync -av --partial --inplace $dry $bwlimit \
            --files-from="$tmplist" \
            -e "$(printf '%q ' "${SSH_CMD[@]}")" \
            "$TARGET:$rdir/" "$dest/"; do
        rc=$?
        if (( attempt >= RETRIES )); then
            echo "rsync failed after $RETRIES attempts (rc=$rc)"; exit "$rc"
        fi
        echo "rsync interrupted (rc=$rc) — retrying in 30s ($((++attempt))/$RETRIES)"
        sleep 30
    done
    rm -f "$tmplist"

    # Post-transfer manifest for the repo's records.
    ls -la "$dest" > "$dest/MANIFEST_local_$(date +%Y%m%d).txt"
    echo "done. local manifest written in $dest"
    ;;

*)
    usage
    ;;
esac
