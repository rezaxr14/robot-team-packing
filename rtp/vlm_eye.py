"""VLMEye: the real Eye. Renders its view, asks a vision-language model for SCENE JSON, parses it.

`render(view) -> PNG bytes` comes from the simulator (isaac PackingSim.render_eye); `chat(messages) -> str` is any
OpenAI-style chat function (ChatClient.chat on a vLLM server). A reply that does not parse becomes an EMPTY report
(counted in `stats`), so perception failures show up as missing objects and are attributed to the Eye by the loop.
"""
from __future__ import annotations
import base64

from .eye_rewards import parse_scene_completion
from .messages import Message

EYE_PROMPT = ("List every object you can see on the table or in the bag as a JSON list of "
              '{"name","cls","region","size","mass_guess","fragility_guess","occluded","confidence"}. Output only JSON.')


def eye_messages(png_bytes: bytes) -> list:
    img = base64.b64encode(png_bytes).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                                         {"type": "text", "text": EYE_PROMPT}]}]


class VLMEye:
    def __init__(self, name: str, chat, render):
        self.name, self.chat, self.render = name, chat, render
        self.stats = {"calls": 0, "parse_failures": 0, "errors": 0}

    def report(self, world, t: int = 0) -> Message:
        self.stats["calls"] += 1
        objs = None
        try:
            objs = parse_scene_completion(self.chat(eye_messages(self.render(self.name))))
        except Exception:                                                   # noqa: BLE001  (server/render failure)
            self.stats["errors"] += 1
        if objs is None:
            self.stats["parse_failures"] += 1
            objs = []
        return Message("SCENE", f"eye_{self.name}", "all", {"objects": objs, "view": self.name}, t=t)
