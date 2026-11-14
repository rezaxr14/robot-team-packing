"""Render Eye training data: images from both cameras + SCENE JSON targets from simulator state.
  python isaac/bringup/06_render_eye_data.py --headless --enable_cameras --seeds 2000 --out data/eye
Writes data/eye/images/seed{S}_view{A|B}.png, data/eye/eye_sft.jsonl (LLaMA-Factory style) and data/eye/eye_grpo.jsonl.
Seeds match rtp.datagen.sample_world(seed), so counterfactual credit replays the same scene symbolically."""
import json
from pathlib import Path
from _launch import launch, check
args, app = launch("render Eye data", extra=[(["--seeds"], dict(type=int, default=100)), (["--seed0"], dict(type=int, default=100000))])

import torch
from PIL import Image
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaac.packing_sim import PackingSim, LAYOUT, _region
from isaac.cameras import PackSceneWithCamsCfg, EYE_POSES
from rtp.perception import EyeConfig
from rtp.eye_rewards import REQUIRED

from rtp.vlm_eye import EYE_PROMPT
PROMPT = "<image>" + EYE_PROMPT                      # identical wording to what VLMEye asks at inference time

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
scene = InteractiveScene(PackSceneWithCamsCfg(num_envs=1, env_spacing=3.0))
sim.reset()
ps = PackingSim(sim, scene)
out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True)
sft = open(out / "eye_sft.jsonl", "w"); grpo = open(out / "eye_grpo.jsonl", "w")
for seed in range(args.seed0, args.seed0 + args.seeds):
    ps.reset(seed)
    objs = []
    for s in ps.object_states():
        o = ps.truth.objs[s.name]
        objs.append({"name": s.name, "cls": s.cls, "region": _region(s, LAYOUT), "size": o.size, "mass_guess": round(o.mass, 3),
                     "fragility_guess": round(o.fragility, 3), "occluded": False, "confidence": 0.9})
    for view, (eye, tgt) in EYE_POSES.items():
        cam = scene["camera_" + view]
        cam.set_world_poses_from_view(torch.tensor([eye], device=sim.device), torch.tensor([tgt], device=sim.device))
        for _ in range(3):
            ps._step()
        cam.update(ps.dt)
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        path = out / "images" / f"seed{seed}_view{view}.png"
        Image.fromarray(rgb).save(path)
        sft.write(json.dumps({"messages": [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": json.dumps(objs)}],
                              "images": [str(path)]}) + "\n")
        grpo.write(json.dumps({"prompt": PROMPT, "images": [str(path)], "seed": seed, "view": view}) + "\n")
sft.close(); grpo.close()
check("rendered images", len(list((out / "images").glob("*.png"))) == 2 * args.seeds)
app.close()
