"""Ship training checkpoints to our private HF repo and fetch the latest one before training.

Two rules from the plan drive this module:

* **Resumable mid-epoch.** A run may be interrupted at any optimizer step; the next machine fetches
  the newest checkpoint from HF and continues at the next micro-batch (see ``rtp.trainer``).
* **Never lose the run.** The local checkpoint is written first, then pushed; a failed push leaves a
  resumable local directory and logs the reason instead of killing training. Old remote steps are
  pruned so a long run cannot fill the repo.

Checkpoints live only in HF (big, versioned, resumable); bulk data and the base model live in Kaggle
datasets. ``CheckpointPolicy`` knows nothing about ``Trainer`` internals beyond ``TrainState.step``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from rtp.trainer import TrainState, save_checkpoint


class CheckpointPolicy:
    """``on_checkpoint`` callback: save locally every ``every`` steps and push to HF.

    ``every=0`` disables periodic saving. Set ``push=False`` for a local-only run (tests, smoke).
    """

    def __init__(self, *, root: Path | str, profile: str, stage: str, store=None,
                 every: int = 50, keep: int = 2, push: bool = True,
                 tokenizer=None, extra: dict | None = None,
                 log: Callable[[str], None] = print):
        self.root = Path(root)
        self.profile = profile
        self.stage = stage
        self.store = store
        self.every = every
        self.keep = keep
        self.push = push and store is not None
        self.tokenizer = tokenizer
        self.extra = extra or {}
        self.say = log or (lambda _m: None)
        self.model = None
        self.optimizer = None
        self.pushed: list[int] = []

    def bind(self, model, optimizer) -> "CheckpointPolicy":
        self.model = model
        self.optimizer = optimizer
        return self

    def __call__(self, state: TrainState) -> Path | None:
        if not self.every or state.step % self.every:
            return None
        if self.model is None:
            raise RuntimeError("CheckpointPolicy.bind(model, optimizer) was never called")
        path = save_checkpoint(self.root / f"step-{state.step:09d}", model=self.model,
                               optimizer=self.optimizer, state=state,
                               tokenizer=self.tokenizer, extra=self.extra)
        self.say(f"checkpoint {state.step} -> {path}")
        if self.push:
            try:
                url = self.store.push_checkpoint(path, self.profile, self.stage, state.step)
                self.pushed.append(state.step)
                self.say(f"pushed {state.step} -> {url}")
                removed = self.store.prune_checkpoints(self.profile, self.stage, keep=self.keep)
                if removed:
                    self.say(f"pruned remote steps {removed}")
            except Exception as exc:                     # noqa: BLE001 - a push must not kill a run
                self.say(f"push failed at step {state.step} ({type(exc).__name__}); "
                         f"local checkpoint kept at {path}")
        return path


def fetch_latest(store, profile: str, stage: str, dest: Path | str,
                 *, log: Callable[[str], None] = print) -> tuple[int | None, Path | None]:
    """Download the newest remote checkpoint for ``profile/stage``; ``(None, None)`` if none.

    Call this *before* constructing the trainer's resume argument. A missing remote is normal on a
    first run, so it is not an error.
    """
    say = log or (lambda _m: None)
    step = store.latest_checkpoint(profile, stage)
    if step is None:
        say(f"no remote checkpoint for {profile}/{stage}: starting from step 0")
        return None, None
    path = store.download_checkpoint(profile, stage, step, dest)
    say(f"fetched {profile}/{stage} step {step} -> {path}")
    return step, path
