"""Typed message layer: JSON payload plus a short text note, with a token-budget bus."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict

# type -> required content keys
SCHEMA = {
    "ABILITY":   {"zones", "payload", "force_limit", "state"},
    "SCENE":     {"objects", "view"},
    "CLAIM":     {"task"},
    "OFFER":     {"task"},
    "DECLINE":   {"task", "reason"},
    "STATUS":    {"task", "progress"},
    "BLOCKED":   {"task", "reason"},
    "TOUCH":     {"obj", "measured_weight", "slip", "grasp_ok", "peak_force"},
    "LOOK":      {"query"},
    "CHALLENGE": {"obj", "evidence"},
    "HANDOVER":  {"obj", "phase"},
    "LOCK":      {"zone", "action"},
    "AMEND":     {"text"},
}
MSG_TYPES = tuple(SCHEMA)


class MessageError(ValueError):
    pass


@dataclass
class Message:
    type: str
    sender: str
    receiver: str
    content: dict = field(default_factory=dict)
    note: str = ""
    t: int = 0

    def validate(self) -> "Message":
        if self.type not in SCHEMA:
            raise MessageError(f"unknown message type {self.type}")
        missing = SCHEMA[self.type] - set(self.content)
        if missing:
            raise MessageError(f"{self.type} missing keys {sorted(missing)}")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(s: str) -> "Message":
        return Message(**json.loads(s)).validate()

    def n_tokens(self) -> int:
        # tokenizer-free proxy (~4 chars per token); swap in the real tokenizer for training
        return max(1, len(self.to_json()) // 4)


class MessageBus:
    """Delivers messages, charges token cost plus a per-message cost, enforces a budget."""

    def __init__(self, token_budget: int = 10**9, per_message_cost: int = 0):
        self.token_budget = token_budget
        self.per_message_cost = per_message_cost
        self.log: list[Message] = []
        self.tokens_used = 0
        self.dropped = 0

    def send(self, msg: Message) -> bool:
        msg.validate()
        cost = msg.n_tokens() + self.per_message_cost
        if self.tokens_used + cost > self.token_budget:
            self.dropped += 1
            return False
        self.tokens_used += cost
        self.log.append(msg)
        return True

    def inbox(self, receiver: str, types: tuple | None = None) -> list:
        return [m for m in self.log
                if m.receiver in (receiver, "all") and (types is None or m.type in types)]

    def by_type(self, t: str) -> list:
        return [m for m in self.log if m.type == t]
