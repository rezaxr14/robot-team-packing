#!/usr/bin/env bash
# Resumable, hash-verified model download (no hf_hub/xet required, survives dropped connections).
#
#   ops/download_model.sh Qwen/Qwen3.5-4B                 # -> $RTP_MODELS/Qwen3.5-4B
#   ops/download_model.sh Qwen/Qwen3.5-4B /data/models/qwen
#   ops/download_model.sh Qwen/Qwen3.5-4B --workers 2
#
# Why not `hf download` alone: on a flaky link the CLI process dies mid-shard and a restart can
# abandon a multi-GB partial. This script
#   * reads the expected size + LFS sha256 for every file from the HF API,
#   * seeds each download from the largest matching `*.incomplete` partial left by `hf download`,
#   * resumes with `curl -C -` in a retry loop until the byte count matches,
#   * verifies sha256 before moving the file into place, and never re-downloads a verified file.
# Exit status is non-zero if any file is still unverified.
set -euo pipefail
MODEL=""; DEST=""; WORKERS=1; positional=()
while [ $# -gt 0 ]; do
  case "$1" in
    --workers) WORKERS="${2:?}"; shift 2 ;;
    --workers=*) WORKERS="${1#*=}"; shift ;;
    --dest) DEST="${2:?}"; shift 2 ;;
    --dest=*) DEST="${1#*=}"; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    -*) echo "unknown argument: $1" >&2; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done
MODEL="${positional[0]:-Qwen/Qwen3.5-4B}"
DEST="${DEST:-${positional[1]:-${RTP_MODELS:-$HOME/models}/${MODEL##*/}}}"

BASE="https://huggingface.co/${MODEL}/resolve/main"
API="https://huggingface.co/api/models/${MODEL}?blobs=true"
CACHE="$DEST/.cache/huggingface/download"
mkdir -p "$DEST"

META="$(mktemp)"; trap 'rm -f "$META" "$META.tsv" "$META.sorted"' EXIT
echo "== $MODEL -> $DEST"
curl -fsSL "$API" -o "$META" || { echo "cannot read $API (gated/private? set HF_TOKEN)"; exit 3; }
python3 - "$META" > "$META.tsv" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
for s in j.get("siblings", []):
    name = s.get("rfilename")
    if not name or name.endswith("/"):
        continue
    size = s.get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    print(f"{name}\t{size}\t{sha}")
PY

sha_of() { sha256sum "$1" | cut -d' ' -f1; }
size_of() { stat -c%s "$1" 2>/dev/null || echo 0; }

fetch_one() {  # name size sha
  local name="$1" size="$2" sha="$3"
  local out="$DEST/$name" part="$DEST/$name.part"
  mkdir -p "$(dirname "$out")"

  if [ -f "$out" ]; then
    if { [ "$size" = 0 ] || [ "$(size_of "$out")" = "$size" ]; } && \
       { [ -z "$sha" ] || [ "$(sha_of "$out")" = "$sha" ]; }; then
      echo "ok      $name ($(size_of "$out") B)"
      return 0
    fi
    echo "redo    $name (size/hash mismatch)"
  fi

  # seed from the largest hf-download partial for this blob
  if [ ! -s "$part" ]; then
    local seed=""
    if [ -n "$sha" ] && [ -d "$CACHE" ]; then
      seed="$(ls -S "$CACHE"/*".$sha"*.incomplete 2>/dev/null | head -1 || true)"
    fi
    if [ -n "$seed" ] && [ -s "$seed" ]; then
      echo "seed    $name <- $(basename "$seed") ($(size_of "$seed") B)"
      mv -f "$seed" "$part"
    fi
  fi
  : >> "$part"

  local attempt=1 max=200
  while [ "$attempt" -le "$max" ]; do
    local have; have="$(size_of "$part")"
    if [ "$size" != 0 ] && [ "$have" = "$size" ]; then break; fi
    if [ "$size" != 0 ] && [ "$have" -gt "$size" ]; then
      echo "trunc   $name (partial larger than expected; restarting)"; : > "$part"; have=0
    fi
    printf '  %-46s %3d%%  %s/%s B (attempt %d)\n' "$name" \
      "$(( size ? have * 100 / size : 0 ))" "$have" "$size" "$attempt"
    curl -fL --retry 6 --retry-delay 2 --retry-all-errors --connect-timeout 20 \
         --speed-limit 1024 --speed-time 120 \
         -C - -o "$part" "$BASE/$name" || true
    attempt=$((attempt + 1))
  done

  if [ "$size" != 0 ] && [ "$(size_of "$part")" != "$size" ]; then
    echo "FAIL    $name: got $(size_of "$part") of $size B" >&2; return 1
  fi
  if [ -n "$sha" ]; then
    local got; got="$(sha_of "$part")"
    if [ "$got" != "$sha" ]; then
      echo "FAIL    $name: sha256 $got != $sha (corrupt partial; delete $part and rerun)" >&2; return 1
    fi
  fi
  mv -f "$part" "$out"
  echo "done    $name ($(size_of "$out") B)"
}

fail=0
# big blobs first so a flaky link spends its time on what matters
sort -t$'\t' -k2,2nr "$META.tsv" > "$META.sorted"
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  fetch_one "$name" "$size" "$sha" || fail=1
done < "$META.sorted"
[ "$fail" = 0 ] || echo "WARNING: some files failed; rerun the same command to resume" >&2

echo
echo "== verify $DEST"
python3 - "$META.tsv" "$DEST" <<'PY' || exit 1
import hashlib, os, sys
bad = []
for line in open(sys.argv[1]):
    name, size, sha = (line.rstrip("\n").split("\t") + ["", ""])[:3]
    if not name:
        continue
    p = os.path.join(sys.argv[2], name)
    if not os.path.exists(p):
        bad.append(f"missing {name}"); continue
    if size and os.path.getsize(p) != int(size):
        bad.append(f"size {name}"); continue
    if sha:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            bad.append(f"sha {name}")
print("\n".join(bad) if bad else f"all files verified in {sys.argv[2]}")
sys.exit(1 if bad else 0)
PY
