"""Remote asset backends for the local -> Kaggle -> Hugging Face workflow.

The plan (§11, D-32) and the user's workflow: look for data/model bytes **locally**, then in a
private Kaggle dataset attached to the notebook, then download from one of **our** private
Kaggle datasets, then from our private Hugging Face repo. Publish goes the same way: Kaggle
datasets for big blobs (so Kaggle's small working directory is never used) and HF for weights,
traces and resumable checkpoints.

Everything here is injectable: pass ``kaggle``/``hf`` clients to :class:`RemoteStore` to test the
ordering offline. Real clients are built only by :func:`from_env`, which never logs token values.
The HF repo id is fixed to ``<account>/robot-team-packing``; we never enumerate or modify other
repos. The Kaggle slug is derived from the asset spec, so two profiles/assets cannot collide.

Tokens come from ``.env`` (HF_TOKEN, KAGGLE_API_TOKEN) and are stripped of CR/LF: a token pasted
with Windows line endings is otherwise rejected as an illegal HTTP header. Kaggle needs the local
proxy (``RTP_KAGGLE_PROXY``, default ``http://localhost:10808``) in this environment.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Mapping

from rtp import assetcache as ac
from rtp.assetcache import AssetSpec, Profile

HF_REPO_NAME = "robot-team-packing"
KAGGLE_PROXY_ENV = "RTP_KAGGLE_PROXY"
DEFAULT_KAGGLE_PROXY = "http://localhost:10808"


class RemoteError(RuntimeError):
    """A remote is configured but the requested operation failed."""


def read_env(path: Path | str = ".env") -> dict:
    """Parse ``KEY=value`` with CRLF stripped. Never prints or returns secrets to logs."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def load_env(path: Path | str = ".env") -> None:
    """Export the parsed env into ``os.environ`` for client libraries that read it directly."""
    for key, value in read_env(path).items():
        os.environ.setdefault(key, value)


def kaggle_proxy(env: Mapping[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return env.get(KAGGLE_PROXY_ENV) or env.get("HTTPS_PROXY") or DEFAULT_KAGGLE_PROXY


def configure_kaggle_proxy(proxy: str | None = None) -> str:
    """Set the proxy for both requests-style and curl-style Kaggle clients; return the URL."""
    url = proxy or kaggle_proxy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = url
    return url


def kaggle_slug(spec: AssetSpec) -> str:
    """A Kaggle dataset id must be lower-case alnum/dash and owner/slug."""
    return ac.slugify(spec.slug)


def hf_repo_id(env: Mapping[str, str] | None = None, client=None) -> str:
    """Only our repo; the account is read from the authenticated token, never guessed."""
    env = os.environ if env is None else env
    owner = env.get("HF_USERNAME")
    if owner and client is None:
        return f"{owner}/{HF_REPO_NAME}"
    client = client or _hf_api()
    return f"{client.whoami()['name']}/{HF_REPO_NAME}"


def _hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def _kaggle_api(retries: int = 4):
    import time
    from kaggle.api.kaggle_api_extended import KaggleApi
    last = None
    for attempt in range(retries):
        try:
            api = KaggleApi()
            api.authenticate()
            return KaggleClient(api)
        except Exception as exc:                                    # noqa: BLE001, PERF401
            # The local proxy intermittently closes TLS during token introspection; a fresh
            # client succeeds on retry. Never print the exception: it may quote the token.
            last = type(exc).__name__
            if attempt + 1 < retries:
                time.sleep(1 + attempt)
    raise RemoteError(f"Kaggle authentication failed after {retries} attempts ({last}); "
                      "check the proxy and KAGGLE_API_TOKEN")


class KaggleClient:
    """Adapter: the real KaggleApi's method names to the small interface RemoteStore expects.

    Kept outside the store so the offline tests can inject their own doubles without importing
    the Kaggle SDK. Dataset creation is private by default in kaggle 2.2 (no ``-u`` flag).
    """

    def __init__(self, api):
        self.api = api

    @property
    def username(self) -> str:
        return self.api.config_values.get("username", "")

    def dataset_exists(self, slug: str, owner: str) -> bool:
        try:
            self.api.dataset_status(slug)
            return True
        except Exception:                                       # noqa: BLE001
            return False

    def dataset_create(self, slug: str, owner: str, root: Path, message: str) -> None:
        self.api.dataset_create_new(folder=str(root), public=False, quiet=True)

    def dataset_version(self, slug: str, owner: str, root: Path, message: str) -> None:
        self.api.dataset_create_version(folder=str(root), version_notes=message, quiet=True)

    def dataset_download(self, slug: str, dest: str, owner: str) -> None:
        self.api.dataset_download_files(f"{owner}/{slug}", path=dest, unzip=True, quiet=True)


def from_env(env_path: Path | str = ".env", *, with_kaggle: bool = True,
             with_hf: bool = True) -> "RemoteStore":
    """Build real clients from ``.env`` (with the Kaggle proxy) — the only place that reads tokens."""
    load_env(env_path)
    kaggle, hf, repo = None, None, ""
    if with_kaggle:
        configure_kaggle_proxy()
        kaggle = _kaggle_api()
    if with_hf:
        hf = _hf_api()
        repo = hf_repo_id(client=hf)
    return RemoteStore(kaggle=kaggle, hf=hf, hf_repo=repo)


class RemoteStore:
    """Push/pull asset directories through our private Kaggle datasets and HF repo.

    ``kaggle`` must implement CLI-style ``dataset_create/dataset_version/dataset_download`` or the
    injected test double; ``hf`` must implement ``create_repo/upload_folder/snapshot_download``.
    """

    def __init__(self, *, kaggle=None, hf=None, hf_repo: str = "", say: Callable[[str], None] = print):
        self.kaggle = kaggle
        self.hf = hf
        self.hf_repo = hf_repo
        self.say = say

    # ---- resolution order ------------------------------------------------------------------
    def resolve(self, spec: AssetSpec, profile: Profile, *, attached=None, verify: str = "size",
                fetch_kaggle: bool = True, fetch_hf: bool = True,
                log: Callable[[str], None] | None = None) -> Path:
        """Local cache -> attached dataset -> Kaggle -> HF; returns a verified cache directory."""
        say = log or self.say
        dest = ac.asset_dir(spec, profile)
        if not ac.verify_manifest(dest, spec, verify=verify):
            say(f"{spec.slug}: local cache hit ({dest})")
            return dest

        # assetcache handles attach + atomic install + verification; the fetch callback is the
        # first remote that actually has the bytes.
        def fetch(tmp: Path) -> None:
            slug = kaggle_slug(spec)
            if fetch_kaggle and self.kaggle is not None:
                try:
                    owner = self.kaggle_owner()
                    self.kaggle.dataset_download(slug, str(tmp), owner)
                    say(f"{spec.slug}: downloaded from Kaggle dataset {owner}/{slug}")
                    ac.write_manifest(tmp, spec, deep=True)
                    return
                except Exception as exc:                          # noqa: BLE001
                    say(f"{spec.slug}: Kaggle fetch failed ({type(exc).__name__}); trying HF")
            if fetch_hf and self.hf is not None:
                self.hf_download(spec, tmp)
                say(f"{spec.slug}: downloaded from HF {self.hf_repo}")
                ac.write_manifest(tmp, spec, deep=True)
                return
            raise ac.AssetUnavailable(f"{spec.slug}: no local/attach/Kaggle/HF copy")

        return ac.resolve(spec, profile, attached=attached, fetch=fetch, verify=verify, log=say)

    def kaggle_owner(self) -> str:
        if getattr(self.kaggle, "username", None):
            return self.kaggle.username
        # The real KaggleApi sets config_values["username"] after authenticate(), but exposes no
        # property for it; test doubles implement get_unclaimed_value()/username instead.
        config = getattr(self.kaggle, "config_values", None)
        if isinstance(config, Mapping) and config.get("username"):
            return config["username"]
        if hasattr(self.kaggle, "get_unclaimed_value"):
            return self.kaggle.get_unclaimed_value()
        raise RemoteError("Kaggle client has no authenticated username; call authenticate() first")

    # ---- publish ---------------------------------------------------------------------------
    def publish(self, root: Path, spec: AssetSpec, *, to_kaggle: bool = True, to_hf: bool = True,
                message: str = "", verify: str = "size") -> dict:
        """Publish a verified asset directory to both remotes; returns what happened."""
        root = Path(root)
        problems = ac.verify_manifest(root, spec, verify=verify)
        if problems:
            raise ac.ManifestError(f"refusing to publish unverified {spec.slug}: {problems[0]}")
        out = {"slug": spec.slug, "kaggle": None, "hf": None}
        if to_kaggle:
            if self.kaggle is None:
                raise RemoteError("no Kaggle client configured")
            slug = kaggle_slug(spec)
            owner = self.kaggle_owner()
            ac.write_export_metadata(root, spec, owner)
            exists = self.kaggle.dataset_exists(slug, owner)
            if exists:
                self.kaggle.dataset_version(slug, owner, root, message or f"update {spec.slug}")
            else:
                self.kaggle.dataset_create(slug, owner, root, message or f"publish {spec.slug}")
            out["kaggle"] = f"{owner}/{slug}"
        if to_hf:
            if self.hf is None:
                raise RemoteError("no HF client configured")
            self.hf_upload(root, spec)
            out["hf"] = f"{self.hf_repo}/tree/main/{self.hf_prefix(spec)}"
        return out

    # ---- HF helpers (kept separate so the injectable doubles are tiny) ----------------------
    @staticmethod
    def hf_prefix(spec: AssetSpec) -> str:
        return f"{spec.kind}/{spec.slug}"

    def hf_upload(self, root: Path, spec: AssetSpec) -> None:
        self.hf.create_repo(repo_id=self.hf_repo, repo_type="model", private=True, exist_ok=True)
        self.hf.upload_folder(folder_path=str(root), repo_id=self.hf_repo, repo_type="model",
                              path_in_repo=self.hf_prefix(spec),
                              commit_message=f"publish {spec.slug}")

    def hf_download(self, spec: AssetSpec, dest: Path) -> None:
        self.hf.snapshot_download(repo_id=self.hf_repo, repo_type="model",
                                  allow_patterns=[f"{self.hf_prefix(spec)}/**"],
                                  local_dir=str(dest), force_download=True)
        inner = dest / self.hf_prefix(spec)
        if inner.exists():
            import shutil
            for child in list(inner.iterdir()):
                shutil.move(str(child), str(dest / child.name))
            inner.rmdir()

    # ---- resumable training checkpoints -----------------------------------------------------
    @staticmethod
    def checkpoint_prefix(profile_name: str, stage: str, step: int | None = None) -> str:
        base = f"checkpoints/{profile_name}/{stage}"
        return base if step is None else f"{base}/step-{step:09d}"

    def push_checkpoint(self, root: Path, profile_name: str, stage: str, step: int,
                        message: str = "") -> str:
        """Upload a training checkpoint under a deterministic, sortable remote prefix."""
        prefix = self.checkpoint_prefix(profile_name, stage, step)
        self.hf.create_repo(repo_id=self.hf_repo, repo_type="model", private=True, exist_ok=True)
        self.hf.upload_folder(folder_path=str(root), repo_id=self.hf_repo, repo_type="model",
                              path_in_repo=prefix,
                              commit_message=message or f"{profile_name}/{stage} step {step}")
        return f"{self.hf_repo}/tree/main/{prefix}"

    def list_checkpoints(self, profile_name: str, stage: str) -> list[int]:
        """Every completed step number present in HF for this profile/stage, ascending."""
        prefix = self.checkpoint_prefix(profile_name, stage) + "/step-"
        steps = set()
        for path in self.hf.list_repo_files(repo_id=self.hf_repo, repo_type="model"):
            if not path.startswith(prefix):
                continue
            token = path[len(prefix):].split("/")[0]
            if token.isdigit():
                steps.add(int(token))
        return sorted(steps)

    def latest_checkpoint(self, profile_name: str, stage: str) -> int | None:
        """The greatest completed step in HF, or None — the resume point before training starts."""
        steps = self.list_checkpoints(profile_name, stage)
        return steps[-1] if steps else None

    def download_checkpoint(self, profile_name: str, stage: str, step: int, dest: Path | str,
                            *, force: bool = False) -> Path:
        """Fetch one checkpoint into ``dest`` (the resume directory handed to the trainer)."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self.hf.snapshot_download(repo_id=self.hf_repo, repo_type="model",
                                  allow_patterns=[f"{self.checkpoint_prefix(profile_name, stage, step)}/**"],
                                  local_dir=str(dest), force_download=force)
        inner = dest / self.checkpoint_prefix(profile_name, stage, step)
        if inner.exists():
            import shutil
            for child in list(inner.iterdir()):
                target = dest / child.name
                if target.exists():
                    shutil.rmtree(target) if target.is_dir() else target.unlink()
                shutil.move(str(child), str(target))
            inner.rmdir()
        return dest

    def prune_checkpoints(self, profile_name: str, stage: str, keep: int = 2) -> list[int]:
        """Keep the ``keep`` newest steps and delete older remote folders; returns deleted steps.

        A 4.66 B-parameter run's adapter+optimizer checkpoint is ~0.4-1.4 GB, so an unpruned
        every-50-steps push would fill the repo. The newest ``keep`` are always preserved.
        """
        if keep < 1:
            raise ValueError("keep must be >= 1")
        steps = self.list_checkpoints(profile_name, stage)
        doomed = steps[:-keep] if len(steps) > keep else []
        for step in doomed:
            self.hf.delete_folder(repo_id=self.hf_repo,
                                  folder_path=self.checkpoint_prefix(profile_name, stage, step),
                                  repo_type="model")
        return doomed
