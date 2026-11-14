"""Eye -> Brain pipeline on VIKI-Bench (Track A), with oracle-Eye and no-Eye controls.

VIKI gives one image + instruction and asks for a plan. Our split: an Eye (VLM) turns the image into a short text
scene description; a text-only Brain plans from instruction + description using the UPSTREAM system prompt and
scorer. Three conditions isolate where performance is lost:
  eye    : real Eye description (needs a VLM server)
  oracle : description built from ground truth (assets and their candidate start positions)
  none   : no description (instruction only)
oracle - none  = value of perfect perception;  oracle vs eye = perception loss;  upstream VLM vs oracle = what
end-to-end vision costs the planner. Positions in VIKI ground truth are candidate SETS, so the oracle description
lists candidates; an Eye sees one.
"""
from __future__ import annotations
import concurrent.futures
import json

from .trackA import Upstream, build_messages, score_l2, to_native

CONDITIONS = ("eye", "oracle", "none")
EYE_PROMPT = ("List every object and robot you can see as a compact JSON list of "
              '{"name": ..., "position": ...} items, where position is the named place it is at or on. Output only JSON.')


def oracle_scene(gt: dict) -> list:
    """Assets from ground-truth init_pos (robots excluded) with their candidate positions."""
    out = []
    for name, pos in gt.get("init_pos", {}).items():
        if name.startswith("R") and name[1:].isdigit():
            continue
        out.append({"name": name, "position": pos if len(pos) > 1 else pos[0]})
    return out


def eye_messages(row: dict) -> list:
    import base64
    img = base64.b64encode(row["images"][0]["bytes"]).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                                         {"type": "text", "text": EYE_PROMPT}]}]


def brain_messages(row: dict, up: Upstream, scene_text: str | None) -> tuple:
    """Upstream L2 system prompt, text-only user turn (the image is replaced by the description)."""
    msgs, gt, task = build_messages(2, row, up)
    text = f"Task Description: {task}\n"
    if scene_text is not None:
        text = f"Scene description (from the vision module): {scene_text}\n" + text
    return [msgs[0], {"role": "user", "content": text}], gt, task


def run_pipeline(rows: list, up: Upstream, condition: str, brain_chat, eye_chat=None, workers: int = 8,
                 limit=None) -> list:
    assert condition in CONDITIONS
    if condition == "eye" and eye_chat is None:
        raise ValueError("condition 'eye' needs eye_chat")
    rows = rows[:limit] if limit else rows

    def one(i):
        try:
            gt = to_native(rows[i]["reward_model"]["ground_truth"])
            if condition == "eye":
                scene = eye_chat(eye_messages(rows[i]))
            elif condition == "oracle":
                scene = json.dumps(oracle_scene(gt))
            else:
                scene = None
            msgs, gt, task = brain_messages(rows[i], up, scene)
            resp = brain_chat(msgs)
            return {"idx": i, "condition": condition, "task_id": gt.get("task_id", f"task_{i}"), "scene": scene,
                    "response": resp, **score_l2(resp, gt, up, i)}
        except Exception as e:                                                  # noqa: BLE001
            return {"idx": i, "condition": condition, "error": f"{type(e).__name__}: {e}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, range(len(rows))))
