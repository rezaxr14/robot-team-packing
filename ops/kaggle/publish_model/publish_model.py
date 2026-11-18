#!/usr/bin/env python3
"""Kaggle-side publish of the Qwen3.5-4B snapshot as our private dataset.

Run **on Kaggle**, never on the laptop: the 9.4 GB model is downloaded by Kaggle and published as a
private dataset from Kaggle's own disk, so this machine never uploads it. The resulting dataset slug
is exactly ``kaggle_slug(ac.model_spec(...))``, which is what ``rtp.remotes.RemoteStore.resolve``
looks for on the next notebook.

Credentials, in order:
  1. ``KAGGLE_API_TOKEN`` (or ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``) already in the kernel env;
  2. Kaggle Secrets: add a secret named ``KAGGLE_API_TOKEN`` to this notebook and it is read here;
  3. HF token is optional: the base model is public, so no secret is needed to download it.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e8"
DATASET_SLUG = "rtp-model-qwen35-4b-851bf6e8"      # == assetcache slug for this spec
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
DEST = WORK / DATASET_SLUG


def say(*a):
    print(*a, flush=True)


def creds() -> None:
    if os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME")
                                              and os.environ.get("KAGGLE_KEY")):
        say("kaggle credentials: present in the environment")
        return
    try:
        from kaggle_secrets import UserSecretsClient
    except Exception:
        raise SystemExit("no Kaggle credentials: attach a secret named KAGGLE_API_TOKEN")
    token = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")
    os.environ["KAGGLE_API_TOKEN"] = token
    say("kaggle credentials: read from Kaggle Secrets (KAGGLE_API_TOKEN)")


def download() -> None:
    from huggingface_hub import snapshot_download
    DEST.mkdir(parents=True, exist_ok=True)
    say(f"downloading {MODEL}@{REVISION} -> {DEST}")
    snapshot_download(MODEL, revision=REVISION, local_dir=str(DEST),
                      token=os.environ.get("HF_TOKEN"))
    # The hub keeps bookkeeping under <local_dir>/.cache; a symlinked blob dir would break the
    # upload if we deleted it, so materialise any symlinks first.
    cache = DEST / ".cache"
    links = [p for p in DEST.rglob("*") if p.is_symlink()]
    for link in links:
        real = link.resolve()
        tmp = link.with_suffix(link.suffix + ".real")
        shutil.copy2(real, tmp)
        link.unlink()
        tmp.rename(link)
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    total = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    say(f"dataset files: {sum(1 for p in DEST.rglob('*') if p.is_file())}, {total / 1e9:.2f} GB, "
        f"symlinks materialised: {len(links)}")


def write_metadata() -> None:
    (DEST / "dataset-metadata.json").write_text(json.dumps({
        "title": "rtp model qwen35 4b 851bf6e8",
        "id": f"{os.environ.get('KAGGLE_USERNAME') or _username()}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": (f"Private snapshot of {MODEL} at revision {REVISION}, published from Kaggle "
                        "so the laptop never uploads 9.4 GB. Consumed by "
                        "rtp.remotes.RemoteStore.resolve."),
    }, indent=2))
    say("wrote dataset-metadata.json (isPrivate=true)")


def _username() -> str:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api.config_values["username"]
    except Exception as exc:                                  # noqa: BLE001
        raise SystemExit(f"cannot determine the Kaggle username ({type(exc).__name__})")


def publish() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    owner = api.config_values["username"]
    os.environ.setdefault("KAGGLE_USERNAME", owner)
    try:
        api.dataset_status(DATASET_SLUG)
        exists = True
    except Exception:                                        # noqa: BLE001
        exists = False
    say(f"dataset {owner}/{DATASET_SLUG}: {'exists, adding a version' if exists else 'creating'}")
    if exists:
        api.dataset_create_version(folder=str(DEST), version_notes=f"{MODEL}@{REVISION}",
                                   quiet=True, convert_to_csv=False, dir_mode="skip")
    else:
        api.dataset_create_new(folder=str(DEST), public=False, quiet=True,
                               convert_to_csv=False, dir_mode="skip")
    say("published")


def main() -> int:
    creds()
    if not (DEST / "config.json").exists():
        download()
    write_metadata()
    publish()
    subprocess.run([sys.executable, "-c", "print('done')"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
