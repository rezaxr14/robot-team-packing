"""Preflight: detect GPU architecture/VRAM and the system, and recommend which install branch to take.

  python ops/preflight.py            # prints a report, writes reports/preflight.json, exit 1 on a hard failure
Pure functions (parse_nvidia_smi, arch_of, recommend) are unit-tested; the probes need the real machine.
"""
from __future__ import annotations
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class GPU:
    name: str
    vram_gb: float
    cc: tuple | None
    driver: str


def parse_nvidia_smi(text: str) -> list:
    """Parse `nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader`."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        name = parts[0]
        m = re.search(r"([\d.]+)\s*MiB", parts[1])
        vram = float(m.group(1)) / 1024 if m else 0.0
        cc = None
        if len(parts) > 2 and re.fullmatch(r"\d+\.\d+", parts[2]):
            maj, mnr = parts[2].split(".")
            cc = (int(maj), int(mnr))
        gpus.append(GPU(name, round(vram, 1), cc, parts[3] if len(parts) > 3 else ""))
    return gpus


def arch_of(g: GPU) -> str:
    cc = g.cc
    if cc:
        if cc[0] == 12:
            return "blackwell"           # RTX 50xx / RTX PRO Blackwell (sm_120)
        if cc[0] == 10:
            return "blackwell_datacenter"
        if cc == (9, 0):
            return "hopper"
        if cc == (8, 9):
            return "ada"
        if cc[0] == 8:
            return "ampere"
    n = g.name.lower()
    if "blackwell" in n or "rtx pro" in n or "5090" in n or "5080" in n:
        return "blackwell"
    if "ada" in n or "4090" in n or "6000 ada" in n:
        return "ada"
    return "unknown"


def recommend(g: GPU) -> dict:
    """Install-branch recommendation under D-25: Isaac Lab 3.0 + Newton, no other physics stack."""
    arch = arch_of(g)
    notes, warns = [], []
    if arch in ("blackwell", "blackwell_datacenter"):
        notes.append("Blackwell (sm_120): use CUDA 12.8+ PyTorch wheels (cu128); fluid-kernel builds need the CUDA 12.8+ toolkit.")
    if arch == "ada":
        notes.append("Ada (sm_89): cu128 wheels also work; no special handling.")
    if arch == "unknown":
        warns.append("could not classify the GPU architecture; check `nvidia-smi --query-gpu=compute_cap --format=csv`")

    # Newton's floor is Turing, cc 7.5 [V:S-17]; Warp >= 1.18 needs driver R580+, else pin warp-lang 1.17.0 [V:S-19].
    cc = g.cc
    newton_ok = cc is not None and cc >= (7, 5)
    major = int(re.match(r"\d+", g.driver).group()) if re.match(r"\d+", g.driver or "") else 0
    warp_pin = None if (not major or major >= 580) else "1.17.0"
    if not newton_ok:
        warns.append(f"cc {cc} is below Newton's floor (7.5): Isaac Lab 3.0 + Newton cannot run on this GPU [V:S-17]")
    if warp_pin:
        notes.append(f"driver {g.driver} < R580: pin warp-lang=={warp_pin} for Newton [V:S-19]")

    vram = g.vram_gb
    return {
        "arch": arch,
        "torch_index": "cu128",
        "sim_stack": "isaac_lab_3.0_newton",       # D-25; no mjsim/PhysX line exists any more
        "isaac_line": "3.0",
        "newton_min_cc": "7.5",
        "newton_ok": newton_ok,
        "warp_pin": warp_pin,
        "vllm_gpu_memory_utilization_4b": round(min(0.5, 18.0 / vram), 2) if vram else 0.3,
        "qlora_4b_fits": vram >= 6,
        "train_lora_4b_fits": vram >= 24,
        "can_run_isaac_and_vllm_together": vram >= 40,
        "can_run_isaac_vllm_and_train_together": vram >= 90,
        "notes": notes, "warnings": warns,
    }


def _run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:                                                  # noqa: BLE001
        return ""


def system_checks() -> list:
    out = []
    osr = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                osr[k] = v.strip('"')
    except OSError:
        pass
    ver = osr.get("VERSION_ID", "?")
    out.append(("os", osr.get("ID") == "ubuntu" and ver in ("22.04", "24.04"), f"{osr.get('PRETTY_NAME', 'unknown')}",
                "warn"))
    try:
        mem_gb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)) / 1024 / 1024
    except Exception:                                                  # noqa: BLE001
        mem_gb = 0.0
    out.append(("ram >= 32 GB (sim workstation; dev laptops are fine)", mem_gb >= 32, f"{mem_gb:.0f} GB", "warn"))
    free = shutil.disk_usage(".").free / 1e9
    out.append(("disk free >= 100 GB (models, renders, Isaac checkout)", free >= 100, f"{free:.0f} GB", "warn"))
    for tool in ("git", "curl", "gcc", "nvidia-smi"):
        out.append((f"tool: {tool}", shutil.which(tool) is not None, shutil.which(tool) or "missing", "fail" if tool in ("git", "nvidia-smi") else "warn"))
    out.append(("uv or conda", bool(shutil.which("uv") or shutil.which("conda")), shutil.which("uv") or shutil.which("conda") or "missing (ops/01_install_uv.sh installs uv)", "warn"))
    return out


def main() -> int:
    smi = _run("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader")
    gpus = parse_nvidia_smi(smi)
    report = {"gpus": [], "system": [], "hard_fail": False}
    print("== GPUs ==")
    if not gpus:
        print("no GPU found by nvidia-smi")
        report["hard_fail"] = True
    for g in gpus:
        r = recommend(g)
        report["gpus"].append({**asdict(g), **r})
        print(f"{g.name}: {g.vram_gb} GB, cc {g.cc}, driver {g.driver} -> arch {r['arch']}")
        for n in r["notes"]:
            print("  note:", n)
        for w in r["warnings"]:
            print("  WARN:", w)
            report["hard_fail"] |= w.startswith("FAIL")
        print(f"  plan: Isaac Lab {r['isaac_line']}, torch index {r['torch_index']}, vLLM 4B utilization {r['vllm_gpu_memory_utilization_4b']}, "
              f"isaac+vllm together: {r['can_run_isaac_and_vllm_together']}, +training together: {r['can_run_isaac_vllm_and_train_together']}")
    print("== system ==")
    for name, ok, detail, sev in system_checks():
        print(f"[{'ok' if ok else sev.upper()}] {name}: {detail}")
        report["system"].append({"check": name, "ok": ok, "detail": detail, "severity": sev})
        report["hard_fail"] |= (not ok and sev == "fail")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/preflight.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nRESULT:", "HARD FAIL" if report["hard_fail"] else "ok (read the warnings)")
    return 1 if report["hard_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
