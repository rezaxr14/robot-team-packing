"""The wheelhouse builder: tags, name matching, and a strict "wheels only" check.

A wheelhouse is only useful if it is (a) labelled with the tags it was built for and (b) complete.
Both are cheap to get wrong in a way that only shows up on Kaggle, offline, with no compiler:
an sdist silently shipped inside it, or a directory named for the wrong ABI. These tests pin the
pure parts; the download itself is exercised by `ops/build_wheelhouse.py --only ...`.
"""
import subprocess
from pathlib import Path

from rtp import assetcache as ac

from ops.build_wheelhouse import (ROOT, SOURCE_BUILT, build_command, install_plan, normalize,
                                 parse_plan, probe_tags, req_name, scan, verify_dir, wheel_dist)


def test_requirement_names_are_normalised_like_pep_503():
    assert req_name("huggingface_hub[cli]") == "huggingface-hub"
    assert req_name("qdrant-client[fastembed]") == "qdrant-client"
    assert req_name("opencv-python-headless") == "opencv-python-headless"
    assert req_name("warp-lang==1.17.0") == "warp-lang"
    assert req_name("torch>=2.6,<3") == "torch"
    assert req_name("numpy; python_version<'3.13'") == "numpy"
    assert normalize("Flash_Attn") == "flash-attn" == normalize("flash.attn")


def test_a_wheel_filename_names_its_distribution_and_an_sdist_does_not():
    assert wheel_dist("torch-2.15.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl") == "torch"
    assert wheel_dist("opencv_python_headless-4.12.0.88-cp37-abi3-manylinux2014_x86_64.whl") \
        == "opencv-python-headless"
    assert wheel_dist("torch-2.15.0.tar.gz") is None


def test_a_missing_wheel_or_a_shipped_sdist_is_a_problem(tmp_path):
    (tmp_path / "pyflakes-4.1.0-py2.py3-none-any.whl").write_bytes(b"x")
    (tmp_path / "transformers-5.23.0.tar.gz").write_bytes(b"x")
    (tmp_path / ac.MANIFEST).write_text("{}")
    have, strays = scan(tmp_path)
    assert have == {"pyflakes"} and strays == ["transformers-5.23.0.tar.gz"]

    problems = verify_dir(tmp_path, ["pyflakes", "transformers"])
    assert any("sdist/non-wheel transformers-5.23.0.tar.gz" in p for p in problems)
    assert not any("no wheel for pyflakes" in p for p in problems), problems
    assert any("no wheel for transformers" in p for p in problems)


def test_a_package_with_no_pypi_wheel_is_called_out_not_counted_as_missing(tmp_path):
    problems = verify_dir(tmp_path, sorted(SOURCE_BUILT))
    assert len(problems) == len(SOURCE_BUILT)
    assert all("build it on that box" in p for p in problems)
    assert not any("no wheel for" in p for p in problems)


def test_a_directory_that_does_not_exist_says_so(tmp_path):
    assert verify_dir(tmp_path / "nope", ["pyflakes"]) == [f"{tmp_path / 'nope'} is not a directory"]


def test_the_installer_is_the_single_source_of_truth_for_the_package_set():
    out = subprocess.run(["bash", str(ROOT / "ops" / "install.sh"), "--profile", "local_4050",
                          "--print-packages"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    plan = parse_plan(out.stdout)
    assert plan["profile"] == "local_4050"
    assert plan["core"] == ["torch", "torchvision"]
    assert "transformers" in plan["pkgs"] and "qdrant-client[fastembed]" in plan["pkgs"]
    assert plan["extra"] == ["bitsandbytes"], "QLoRA's extra only on the 6 GB profile"
    assert plan["flash"] is True and plan["venv"].endswith("local_4050")

    t4 = install_plan("kaggle_t4x2")
    assert t4["extra"] == [] and t4["flash"] is False


def test_the_plan_refuses_to_guess_a_profile():
    out = subprocess.run(["bash", str(ROOT / "ops" / "install.sh"), "--print-packages"],
                         capture_output=True, text=True)
    assert out.returncode == 2 and "--profile" in out.stderr


def test_a_cross_build_pins_the_abi_and_never_reaches_pypi_implicitly(tmp_path):
    py = Path("/usr/bin/false")
    core = build_command(py, tmp_path, ["torch"], index_url="https://download.pytorch.org/whl/cu128",
                         target={"python": "3.12", "platform": "manylinux2014_x86_64"})
    assert "--only-binary=:all:" in core and "--index-url" in core
    assert core[-2:] == ["cp312", "--platform"] or "cp312" in core
    assert "manylinux2014_x86_64" in core and "3.12" in core
    pypi = build_command(py, tmp_path, ["pyflakes"], index_url=None, target=None)
    assert "--index-url" not in pypi and "--platform" not in pypi


def test_tags_come_from_an_interpreter_and_report_its_python():
    tags = probe_tags(Path(__import__("sys").executable))
    assert len(tags["python"].split(".")) == 2
    assert set(tags) <= {"python", "torch", "cuda", "sm"}
    assert "torch" in tags, "the test venv has torch; a probe that loses it cannot tag a wheelhouse"


def test_a_wheelhouse_slug_carries_the_tags_that_make_it_valid():
    spec = ac.wheelhouse_spec(python="3.12", torch="2.15.0", cuda="13.2", sm="89")
    assert spec.slug == "rtp-wheels-py3.12-torch2.15.0-cu13.2-sm89"
    other = ac.wheelhouse_spec(python="3.11", torch="2.15.0", cuda="12.1", sm="75")
    assert other.slug != spec.slug and spec.tags["sm"] == "89"
