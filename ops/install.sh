#!/usr/bin/env bash
# One idempotent installer for every dependency this project needs, per hardware profile.
#
#   ops/install.sh --profile local_4090                # train/eval/eyes/retrieval env
#   ops/install.sh --profile local_4050                # QLoRA env (adds bitsandbytes)
#   ops/install.sh --profile kaggle_t4x2               # fp16 + SDPA env (no bf16, no FA2)
#   ops/install.sh --profile local_4050 --dry-run      # print the plan, change nothing
#   ops/install.sh --profile local_4090 --check        # verify an existing env, install nothing
#   ops/install.sh --profile kaggle_t4x2 --wheelhouse /kaggle/input/rtp-wheels-...  # offline
#   ops/install.sh --profile kaggle_t4x2 --print-packages  # machine-readable plan (build_wheelhouse)
#   ops/install.sh --isaac                             # SEPARATE Isaac Sim / Newton venv
#   ops/install.sh --tex                               # TinyTeX, user-level, no sudo
#
# Idempotent: a stamp (venv/.rtp_install.json) records the recipe hash. Re-running with the same
# profile/packages is a no-op unless --force. Every profile gets its own venv, so two profiles never
# share a torch build, an adapter or a cache. After installing, the script verifies what is really
# importable and prints the attention/dtype decision that rtp.profiles.resolve_runtime() will make.
#
# FlashAttention-2 is attempted only on profiles that ask for it and only on cc >= 8.0. A failed
# build is not fatal: it is recorded, and training falls back to SDPA with a printed reason.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTP_HOME="${RTP_HOME:-$ROOT}"          # everything project-local; no external work tree
VENVS_ROOT="${RTP_VENVS:-$HOME/venvs}" # the single venv folder for all profiles
PY="${RTP_PYTHON:-3.12}"
TORCH_INDEX="${RTP_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
DRY=0; FORCE=0; CHECK=0; ISAAC=0; TEX=0; FLASH=auto; PROFILE=""; VENV=""; DRIVER_MAJOR=""; WHEELHOUSE=""; PRINT_PKGS=0
ISAACLAB_REF="${ISAACLAB_REF:-v3.0.0}"   # D-25: Isaac Lab 3.0 (GA) + Newton only

# Print the leading comment block (after the shebang) as the help text: editing the header never
# needs a matching line range to be updated.
usage() { awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="${2:?--profile needs a value}"; shift 2 ;;
    --profile=*) PROFILE="${1#*=}"; shift ;;
    --venv) VENV="${2:?--venv needs a path}"; shift 2 ;;
    --venv=*) VENV="${1#*=}"; shift ;;
    --dry-run) DRY=1; shift ;;
    --force) FORCE=1; shift ;;
    --check) CHECK=1; shift ;;
    --isaac) ISAAC=1; shift ;;
    --tex) TEX=1; shift ;;
    --no-flash-attn) FLASH=off; shift ;;
    --flash-attn) FLASH=on; shift ;;
    --driver-major) DRIVER_MAJOR="${2:?}"; shift 2 ;;
    --torch-index) TORCH_INDEX="${2:?}"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="${2:?--wheelhouse needs a directory}"; shift 2 ;;
    --wheelhouse=*) WHEELHOUSE="${1#*=}"; shift ;;
    --print-packages) PRINT_PKGS=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 2 ;;
  esac
done

say()  { printf '\n=== %s ===\n' "$*"; }
run()  { if [ "$DRY" = 1 ]; then printf '  [dry-run]'; printf ' %q' "$@"; printf '\n'; else "$@"; fi; }

# ---------------------------------------------------------------- profile recipe
# Keep in sync with rtp/profiles.py (that module is the source of truth for training values;
# the package list below is what an environment needs to execute those runs).
QLORA=0; WANT_FLASH=0; CC_MIN=0; DTYPE=none
case "$PROFILE" in
  local_4050)        CC_MIN=8; DTYPE=bf16; WANT_FLASH=1; QLORA=1 ;;
  local_4090)        CC_MIN=8; DTYPE=bf16; WANT_FLASH=1; QLORA=0 ;;
  kaggle_t4x2)       CC_MIN=7; DTYPE=fp16; WANT_FLASH=0; QLORA=0 ;;
  pro6000)           CC_MIN=8; DTYPE=bf16; WANT_FLASH=1; QLORA=0 ;;
  tpu_v5e8|tpu_v6e1|tpu_v5e1) CC_MIN=0; DTYPE=bf16; WANT_FLASH=0; QLORA=0
      echo "NOTE: $PROFILE is a declared stub (supported=False): no XLA install path exists yet." >&2 ;;
  "") : ;;
  *)  echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

if [ -z "$PROFILE" ] && [ "$ISAAC" = 0 ] && [ "$TEX" = 0 ]; then
  echo "--profile is required (one of: local_4050 local_4090 kaggle_t4x2 pro6000 tpu_v5e8 tpu_v6e1 tpu_v5e1)" >&2
  usage 2
fi

VENV="${VENV:-${PROFILE:+$VENVS_ROOT/$PROFILE}}"
PYBIN="${VENV:+$VENV/bin/python}"

if [ -n "$WHEELHOUSE" ] && [ ! -d "$WHEELHOUSE" ]; then
  echo "--wheelhouse $WHEELHOUSE: not a directory (mount the Kaggle dataset or build the wheels first)" >&2
  exit 2
fi

CORE=(torch torchvision)
ML=(transformers trl peft datasets accelerate "huggingface_hub[cli]" safetensors sentencepiece
    protobuf einops)
RETRIEVAL=("qdrant-client[fastembed]")
AGENTS=(langgraph langchain-core)
EYES=(ultralytics opencv-python-headless)
SCIENCE=(numpy pandas pyarrow pillow matplotlib)
DEV=(pytest pyflakes)
PKGS=("${CORE[@]}" "${ML[@]}" "${RETRIEVAL[@]}" "${AGENTS[@]}" "${EYES[@]}" "${SCIENCE[@]}" "${DEV[@]}")
EXTRA=()
[ "$QLORA" = 1 ] && EXTRA+=(bitsandbytes)

if [ "$PRINT_PKGS" = 1 ]; then
  # The package set lives here and nowhere else: ops/build_wheelhouse.py reads this instead of
  # keeping a second copy that drifts. `core` comes from the torch index, `pkgs`/`extra` from PyPI.
  [ -n "$PROFILE" ] || { echo "--print-packages needs --profile" >&2; exit 2; }
  printf 'profile=%s\n'   "$PROFILE"
  printf 'venv=%s\n'      "$VENV"
  printf 'python=%s\n'    "$PY"
  printf 'torch_index=%s\n' "$TORCH_INDEX"
  printf 'flash=%s\n'     "$([ "$WANT_FLASH" = 1 ] && [ "$FLASH" != off ] && echo 1 || echo 0)"
  printf 'core=%s\n'      "${CORE[*]}"
  printf 'pkgs=%s\n'      "${ML[*]} ${RETRIEVAL[*]} ${AGENTS[*]} ${EYES[*]} ${SCIENCE[*]} ${DEV[*]}"
  printf 'extra=%s\n'     "${EXTRA[*]:-}"
  exit 0
fi

# driver -> warp pin for Newton (README: Warp >=1.18 needs driver R580+; older: warp-lang==1.17.0)
if [ -z "$DRIVER_MAJOR" ] && command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
fi
DRIVER_MAJOR="${DRIVER_MAJOR:-0}"

RECIPE="profile=$PROFILE;py=$PY;torch=$TORCH_INDEX;pkgs=$(IFS=,; echo "${PKGS[*]}");extra=$(IFS=,; echo "${EXTRA[*]:-}");flash=$FLASH;isaac=$ISAAC;tex=$TEX;offline=${WHEELHOUSE:+1}"
STAMP="$VENV/.rtp_install.json"
RECIPE_HASH="$(printf '%s' "$RECIPE" | sha256sum | cut -c1-16)"
FROZEN_HASH=""
if [ -f "$STAMP" ]; then
  FROZEN_HASH="$(sed -n 's/.*"recipe_hash": *"\([^"]*\)".*/\1/p' "$STAMP" | head -1)"
fi

say "plan"
cat <<EOF
  repo        : $ROOT
  profile     : $PROFILE   (QLoRA=$QLORA, wanted flash-attn=$WANT_FLASH, dtype=$DTYPE)
  venv        : $VENV
  python      : $PY
  torch index : $TORCH_INDEX
  recipe hash : $RECIPE_HASH${FROZEN_HASH:+   (installed: $FROZEN_HASH)}
  mode        : dry-run=$DRY check=$CHECK force=$FORCE isaac=$ISAAC tex=$TEX flash=${FLASH}
  wheelhouse  : ${WHEELHOUSE:-none (online: PyPI + the torch index)}
EOF

if [ -n "$PROFILE" ] && [ "$CHECK" = 0 ] && [ "$DRY" = 0 ] && [ -n "$FROZEN_HASH" ] && [ "$FROZEN_HASH" = "$RECIPE_HASH" ] && [ "$FORCE" = 0 ]; then
  say "already installed (recipe unchanged) — use --force to reinstall"
else
  if [ "$CHECK" = 0 ] && [ -n "$PROFILE" ]; then
    say "python env"
    if command -v uv >/dev/null 2>&1; then
      [ -d "$VENV" ] || run uv venv "$VENV" --python "$PY" --seed
    else
      [ -d "$VENV" ] || run python3 -m venv "$VENV"
    fi
    PIP=(uv pip install --python "$PYBIN")           # uv talks to the venv's interpreter directly
    command -v uv >/dev/null 2>&1 || PIP=("$PYBIN" -m pip install)
    TORCH_FLAGS=(--index-url "$TORCH_INDEX")
    if [ -n "$WHEELHOUSE" ]; then
      # offline mode: resolve only from the mounted wheelhouse (Kaggle dataset or a local build)
      PIP+=(--no-index --find-links "$WHEELHOUSE")
      TORCH_FLAGS=()
    fi

    say "torch ($DTYPE build for this profile)"
    run "${PIP[@]}" "${CORE[@]}" "${TORCH_FLAGS[@]}"

    say "training + retrieval + agents + eyes"
    run "${PIP[@]}" "${ML[@]}" "${RETRIEVAL[@]}" "${AGENTS[@]}" "${EYES[@]}" "${SCIENCE[@]}" "${DEV[@]}"
    if [ "${#EXTRA[@]}" -gt 0 ]; then
      say "qlora extra"
      run "${PIP[@]}" "${EXTRA[@]}"
    fi

    if [ "$FLASH" = off ] || [ "$WANT_FLASH" = 0 ]; then
      say "flash-attn: skipped (profile/flag)"
    else
      say "flash-attn (cc >= 8.0 only; a failed build falls back to SDPA)"
      if [ "$DRY" = 1 ]; then
        run "${PIP[@]}" --no-build-isolation flash-attn
      elif [ -n "$WHEELHOUSE" ]; then
        # a wheelhouse run must not reach PyPI: install the prebuilt wheel or fall back to SDPA
        "${PIP[@]}" flash-attn || {
          echo "  WARNING: no flash-attn wheel in $WHEELHOUSE (sm tag?); training will use sdpa" >&2
          echo "  build it once with: MAX_JOBS=2 FLASH_ATTN_CUDA_ARCHS=<cc> uv pip wheel flash-attn --no-build-isolation" >&2
        }
      elif "${PIP[@]}" --no-build-isolation flash-attn; then
        echo "  flash-attn installed"
      else
        echo "  WARNING: flash-attn did not build; training will use sdpa (reason is printed by --check)" >&2
      fi
    fi

    say "record the recipe"
    if [ "$DRY" = 0 ]; then
      mkdir -p "$ROOT/requirements"
      printf '%s\n' "${PKGS[@]}" "${EXTRA[@]}" > "$ROOT/requirements/$PROFILE.in"
      { "$PYBIN" -m pip freeze --all 2>/dev/null || uv pip freeze --python "$PYBIN"; } \
        > "$ROOT/requirements/$PROFILE.lock"
      "$PYBIN" - "$STAMP" "$PROFILE" "$RECIPE_HASH" "$ROOT/requirements/$PROFILE.lock" <<'PY'
import hashlib, json, sys, time
stamp, profile, recipe_hash, lock = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    frozen = hashlib.sha256(open(lock, "rb").read()).hexdigest()[:16]
except OSError:
    frozen = None
json.dump({"profile": profile, "recipe_hash": recipe_hash, "lock_hash": frozen,
           "lock": lock, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(stamp, "w"), indent=2, sort_keys=True)
print("  stamp:", stamp)
PY
    fi
  fi
fi

# ---------------------------------------------------------------- optional: Isaac Lab 3.0 + Newton (D-25)
if [ "$ISAAC" = 1 ] && [ "$CHECK" = 0 ]; then
  IVENV="${RTP_ISAAC_VENV:-$VENVS_ROOT/rtp}"
  say "Isaac Lab ${ISAACLAB_REF} + Newton venv ($IVENV)  [D-25: the only simulator stack]"
  echo "  NOTE: Python 3.12 is required by Isaac Lab 3.0 [V:S-15]; Isaac Lab's own requirement files"
  echo "        at the tag pin the rest (docs/SIMULATION.md §2). There is no isaacsim[all] pip step here"
  echo "        any more: that was the removed 2.3-era stack."
  echo "  NOTE: cc 7.5+ (Turing or newer). The tag pins warp-lang 1.17.0 and newton[sim]==1.6.1;"
  echo "        the pinned torch is CUDA 13, which needs driver >= 580.65.06 [V:S-19, tag pyproject/uv.lock]."
  if command -v uv >/dev/null 2>&1; then
    [ -d "$IVENV" ] || run uv venv "$IVENV" --python "$PY" --seed
  else
    [ -d "$IVENV" ] || run python3 -m venv "$IVENV"
  fi
  if [ ! -d "$RTP_HOME/IsaacLab/.git" ]; then
    run git clone https://github.com/isaac-sim/IsaacLab.git "$RTP_HOME/IsaacLab"
  fi
  run git -C "$RTP_HOME/IsaacLab" fetch --tags
  run git -C "$RTP_HOME/IsaacLab" checkout "$ISAACLAB_REF"
  # The tag's own uv.lock is the pin (newton 1.6.1, warp 1.17.0, torch+cu130). Installing
  # `pip install newton` by hand picks a newer Newton and silently diverges from the tag;
  # `./isaaclab.sh --install` is deprecated at v3.0.0 in favour of `uv run isaaclab`.
  say "uv sync at $ISAACLAB_REF into $IVENV (the tag's lock; no Isaac Sim, Newton only)"
  UV_PROJECT_ENVIRONMENT="$IVENV" run uv sync --project "$RTP_HOME/IsaacLab" --no-dev
  echo "  next: UV_PROJECT_ENVIRONMENT=$IVENV uv run --project $RTP_HOME/IsaacLab isaaclab --help"
  echo "  next: $IVENV/bin/python -c 'import newton, warp, isaaclab; print(newton.__version__, warp.__version__)'"
  echo "  next: source ops/env.sh && python ops/plan.py run p2   # gate G-3 (rgb+depth on this GPU)"
  echo "  UNVERIFIED until G-3: the exact install flag that selects Newton inside Isaac Lab at this tag."
fi

# ---------------------------------------------------------------- optional: TinyTeX (for paper/)
if [ "$TEX" = 1 ] && [ "$CHECK" = 0 ]; then
  say "TinyTeX (user-level)"
  if command -v pdflatex >/dev/null 2>&1; then
    echo "  pdflatex already present: $(command -v pdflatex)"
  else
    run bash -c 'curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh'
    echo "  next: add \$HOME/.TinyTeX/bin/*/ to PATH, then: tlmgr install booktabs natbib hyperref" >&2
  fi
fi

# ---------------------------------------------------------------- verify what is really there
if [ "$DRY" = 1 ]; then
  say "verification skipped in --dry-run"; exit 0
fi
if [ -z "$PROFILE" ]; then
  say "no --profile given: nothing to verify"; exit 0
fi
if [ ! -x "$PYBIN" ]; then
  echo "no interpreter at $PYBIN — run without --check to install" >&2; exit 3
fi

say "verify"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYBIN" - "$PROFILE" <<'PY'
import importlib, json, sys
profile = sys.argv[1]
out = {"profile": profile, "imports": {}, "cuda": {}, "runtime": {}}
for mod in ("torch", "transformers", "trl", "peft", "datasets", "accelerate",
            "qdrant_client", "langgraph", "ultralytics", "cv2", "bitsandbytes", "flash_attn"):
    try:
        m = importlib.import_module(mod)
        out["imports"][mod] = getattr(m, "__version__", "ok")
    except Exception as e:                                   # noqa: BLE001
        out["imports"][mod] = f"MISSING ({type(e).__name__})"
try:
    import torch
    out["cuda"] = {"available": torch.cuda.is_available(),
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "cc": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
                   "bf16": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
                   "torch_cuda": torch.version.cuda}
except Exception as e:                                       # noqa: BLE001
    out["cuda"] = {"error": f"{type(e).__name__}: {e}"}
try:
    from rtp.profiles import detect_caps, get_profile, resolve_runtime
    p = get_profile(profile)
    attn, dtype, notes = resolve_runtime(p, detect_caps())
    out["runtime"] = {"attn": attn, "dtype": dtype, "notes": notes,
                      "tuning": p.tuning, "paths": p.paths.as_dict(),
                      "config_hash": p.config_hash()}
except Exception as e:                                       # noqa: BLE001
    out["runtime"] = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps(out, indent=2))
missing = [k for k, v in out["imports"].items() if str(v).startswith("MISSING")]
if missing:
    print(f"\nMISSING imports: {missing}\n  (bitsandbytes is only needed for QLoRA; flash_attn only for FA2)",
          file=sys.stderr)
PY
