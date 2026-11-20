"""HARD boundary rule: this repo is the only writable local workspace.

The user explicitly forbids creating, modifying or deleting files anywhere on the
machine outside ``~/Projects/robot_team``.  The single authorised exception is the
venv folder ``~/venvs``, which the user ordered for the toolchain migration
(2026-11-20).  A failure here means a hard rule was broken; stop all work.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path.home() / "Projects" / "robot_team"


def test_project_root_is_the_only_workspace():
    assert ROOT.resolve() == EXPECTED_ROOT.resolve(), (
        f"tests are running from {ROOT}, expected {EXPECTED_ROOT}"
    )


def test_old_external_work_tree_is_gone():
    assert not (Path.home() / "rtp_work").exists(), "the forbidden ~/rtp_work tree exists again"


def test_env_points_only_inside_project_or_the_single_venvs_folder():
    env = (ROOT / "ops" / "env.sh").read_text()
    def default(name: str) -> str:
        m = re.search(rf'^export {name}="([^"]+)"', env, re.M)
        assert m, f"ops/env.sh has no export {name}"
        return m.group(1)
    assert "$RTP_ROOT" in default("RTP_HOME")
    assert "$HOME/venvs" in default("VENVS")
    assert default("ISAACLAB_DIR").startswith("$RTP_ROOT/")
    assert "$RTP_ROOT/.hf_cache" in default("HF_HOME")
    assert "$VENVS/isaac_newton/bin/python" in default("ISAAC_PY")


def test_no_tracked_file_still_points_at_the_deleted_tree():
    # CLAUDE.md is the rule itself and is allowed to name the forbidden path.
    out = subprocess.run(
        ["git", "grep", "-l", "rtp_work", "--", ":!CLAUDE.md"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert out.returncode == 1 and not out.stdout.strip(), (
        "tracked files still reference the forbidden external work tree:\n" + out.stdout
    )


def test_vendored_isaac_lab_lives_inside_the_project():
    isaac = ROOT / ".isaac" / "IsaacLab"
    assert isaac.is_dir(), "Isaac Lab source must be inside the project (.isaac/IsaacLab)"


def test_no_python_cache_escaped_to_home_root():
    # A common failure mode is `pip install --user` or running python from $HOME.
    assert not list(Path.home().glob("robot_team*.egg-info"))
