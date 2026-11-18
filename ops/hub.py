#!/usr/bin/env python3
"""Publish or fetch one asset through our private storage (local -> Kaggle -> HF).

    ops/hub.py status                                   # account/repo identity + credentials
    ops/hub.py publish DIR --kind model --name qwen35-4b --revision 851bf6e8
    ops/hub.py publish DIR --kind data  --name sft --revision <sha> --kaggle-only
    ops/hub.py fetch   --kind model --name qwen35-4b --revision 851bf6e8 --profile local_4050
    ops/hub.py checkpoint-push DIR --profile local_4050 --stage sft --step 120
    ops/hub.py checkpoint-latest --profile local_4050 --stage sft

Publishing writes a content manifest (sizes + sha256) first and refuses an unverified directory.
Nothing outside our own `<account>/robot-team-packing` HF repo and `<account>/*` Kaggle datasets
is touched; no credential value is ever printed. Kaggle connections use the proxy from
``RTP_KAGGLE_PROXY`` (default http://localhost:10808).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtp import assetcache as ac                                  # noqa: E402
from rtp import remotes                                            # noqa: E402


def spec_for(kind: str, name: str, revision: str) -> ac.AssetSpec:
    if kind == "model":
        return ac.model_spec("local", revision=revision, name=name)
    return ac.data_spec_kind(kind, name, revision)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("publish")
    p.add_argument("dir", help="verified local asset directory (manifest is written if absent)")
    p.add_argument("--kind", choices=["model", "data", "adapters", "wheelhouse"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--revision", default="local")
    p.add_argument("--no-kaggle", action="store_true")
    p.add_argument("--no-hf", action="store_true")
    p.add_argument("--profile", default="local_4050", help="profile whose cache tree is used")
    p = sub.add_parser("fetch")
    p.add_argument("--kind", choices=["model", "data", "adapters", "wheelhouse"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--revision", default="local")
    p.add_argument("--profile", default="local_4050")
    p.add_argument("--no-kaggle", action="store_true")
    p.add_argument("--no-hf", action="store_true")
    p.add_argument("--verify", default="size", choices=["size", "sha256"])
    p.add_argument("--attached", default="", help="optional /kaggle/input or local attach root")
    p = sub.add_parser("checkpoint-push")
    p.add_argument("dir")
    p.add_argument("--profile", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--step", type=int, required=True)
    p = sub.add_parser("checkpoint-drop")
    p.add_argument("--profile", required=True)
    p.add_argument("--stage", required=True)
    p = sub.add_parser("checkpoint-latest")
    p.add_argument("--profile", required=True)
    p.add_argument("--stage", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "status":
        store = remotes.from_env()
        print(json.dumps({"hf_repo": store.hf_repo, "kaggle_user": store.kaggle_owner(),
                          "kaggle_proxy": remotes.kaggle_proxy(),
                          "hf_token_set": bool(__import__("os").environ.get("HF_TOKEN")),
                          "kaggle_token_set": bool(__import__("os").environ.get("KAGGLE_API_TOKEN"))},
                         indent=2))
        return 0

    if a.cmd in ("publish", "fetch"):
        store = remotes.from_env(with_kaggle=not a.no_kaggle, with_hf=not a.no_hf)
        if a.kind == "model":
            spec = ac.model_spec("local", revision=a.revision, name=a.name)
        elif a.kind == "wheelhouse":
            tags = ac.env_tags()
            spec = ac.wheelhouse_spec(python=tags["python"], torch=tags.get("torch", "?"),
                                      cuda=tags.get("cuda", "?"), sm=tags.get("sm", "?"))
        elif a.kind == "adapters":
            from rtp.profiles import get_profile
            spec = ac.adapters_spec(get_profile(a.profile), a.name, a.revision)
        else:
            spec = ac.data_spec_kind(a.kind, a.name, a.revision)
        _expected_slug = spec.slug

        if a.cmd == "publish":
            root = Path(a.dir).expanduser()
            if not root.is_dir():
                print(f"not a directory: {root}", file=sys.stderr)
                return 2
            manifest = ac.read_manifest(root)
            if manifest:
                # An existing cache manifest is the identity: its source/revision/hashes must not be
                # replaced by CLI guesses (the safety check in publish() caught exactly that).
                spec = ac.AssetSpec(kind=manifest.get("kind", spec.kind), slug=manifest.get("slug", spec.slug),
                                    source=manifest.get("source", spec.source),
                                    revision=manifest.get("revision", spec.revision),
                                    files=tuple(ac.FileSpec(**f) for f in manifest.get("files", ())),
                                    tags=manifest.get("tags", {}) or {})
                if spec.slug != _expected_slug or spec.revision != a.revision:
                    print(f"refusing to publish {root}: manifest slug/revision "
                          f"({spec.slug}/{spec.revision}) != requested ({_expected_slug}/{a.revision})",
                          file=sys.stderr)
                    return 2
            else:
                ac.write_manifest(root, spec, deep=True)
            out = store.publish(root, spec, to_kaggle=not a.no_kaggle, to_hf=not a.no_hf)
            print(json.dumps(out, indent=2))
            return 0

        from rtp.profiles import get_profile
        path = store.resolve(spec, get_profile(a.profile), attached=a.attached or None,
                             verify=a.verify, fetch_kaggle=not a.no_kaggle, fetch_hf=not a.no_hf)
        print(json.dumps({"slug": spec.slug, "path": str(path)}, indent=2))
        return 0

    if a.cmd.startswith("checkpoint-"):
        # Checkpoints live only in our HF repo; no Kaggle auth needed.
        store = remotes.from_env(with_kaggle=False)
    if a.cmd == "checkpoint-push":
        print(store.push_checkpoint(Path(a.dir).expanduser(), a.profile, a.stage, a.step))
        return 0
    if a.cmd == "checkpoint-drop":
        store = remotes.from_env(with_kaggle=False)
        print(json.dumps({"profile": a.profile, "stage": a.stage,
                          "deleted_steps": store.drop_checkpoint_stage(a.profile, a.stage)}))
        return 0
    if a.cmd == "checkpoint-latest":
        step = store.latest_checkpoint(a.profile, a.stage)
        print(json.dumps({"profile": a.profile, "stage": a.stage, "latest_step": step}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
