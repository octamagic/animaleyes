"""Animal identification with Claude.

Every request has the same prefix: a frozen system prompt and the reference photos, marked
for prompt caching so they are billed once per cache lifetime. The most recent camera frames
and the question come after the cache breakpoint. Output is constrained to a JSON schema so
the state machine never has to parse prose.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from PIL import Image

from .camera import Frame
from .detect import BedVerdict, FeedingVerdict, Identifier, Verdict
from .personas import TARGET_KEY, Persona
from .store import Store

# Re-exported for backward compatibility: callers still do `from .vision import Verdict, ...`.
__all__ = ["ClaudeIdentifier", "FeedingVerdict", "Identifier", "Verdict", "estimate_cost_usd"]

log = logging.getLogger(__name__)

ANIMALS = ("grrr", "bowie", "cat")
ANIMAL_DESCRIPTIONS = {
    "grrr": "Grrr: a tiny, old, black schnoodle. Small body, short legs, scruffy coat. "
    "The ONLY animal that may be fed.",
    "bowie": "Bowie: a medium-sized, blonde, lanky dog. Much larger and longer-legged than Grrr.",
    "cat": "The cat. Never fed.",
}
REFERENCE_MAX_EDGE = 640
VERDICT_MAX_TOKENS = 300

SYSTEM_PROMPT = """You identify which household animal, if any, is at a pet food bowl, from
security-camera frames. You are given labelled reference photos of each animal, then the
most recent frames from the camera (oldest first).

Rules:
- "unsure" is a valid and preferred answer when frames are dark, blurry, partial, or the
  animal is too far away to judge. Never guess between animals; choose "unsure".
- At night the camera uses infrared and the image is greyscale: a black coat is NOT a cue.
  Use body size, leg length, body proportions, ear and muzzle shape, and posture.
- Bowie is much larger and lankier than Grrr. Grrr is small and low to the ground.
- POC fallback when few or no reference photos are provided: identify by species and size
  alone. The ONLY small dog in this home is Grrr, so classify any clearly small dog as
  "grrr", a clearly larger or lankier dog as "bowie", and a cat as "cat". A cat is not a
  small dog: tell them apart by the cat's shorter muzzle, triangular upright ears, long
  tail, and lighter, more fluid gait. If you cannot tell a small dog from a cat, choose
  "unsure" — never feed on a guess.
- "at_bowl" is true only when the animal's head is at or in the bowl area, not merely nearby.
- Answer with a single JSON object matching the schema you were given, nothing else."""


BED_SYSTEM_PROMPT = """You watch a night-vision camera at floor level pointed at a small dog's
sleeping spot. There are two beds: a fluffy bed in the foreground, and a second bed behind it,
underneath a white chair, where she usually sleeps. The dog, Grrr, is tiny with a black curly
coat. In infrared (greyscale) she shows up as a dark, curly-textured lump, often mostly hidden
behind the front bed, and the space under the chair is dark too, so look closely for curly fur
texture against the pale bedding. Other animals in the home: Bowie, a much larger pale
short-coated dog, and a cat.

Rules:
- When a reference photo of the empty beds is given, compare the current frame with it,
  especially the space under the chair.
- Answer "unsure" rather than guess when the frame is too dark or obscured to tell.
- Answer with a single JSON object matching the schema you were given, nothing else."""

BED_QUESTION = (
    "Is Grrr lying in either bed right now? occupant: 'grrr', 'other' (a different animal is "
    "in a bed), 'empty' (no animal in either bed), or 'unsure'. where: 'under_chair', "
    "'front_bed', or 'none'."
)

BED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "occupant": {"type": "string", "enum": ["grrr", "other", "empty", "unsure"]},
        "where": {"type": "string", "enum": ["under_chair", "front_bed", "none"]},
        "reason": {"type": "string"},
    },
    "required": ["occupant", "where", "reason"],
    "additionalProperties": False,
}


def identify_schema(keys: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "animal": {"type": "string", "enum": [*keys, "none", "unsure"]},
            # The structured-output API rejects minimum/maximum on numbers; the prompt states
            # the 0..1 range and the machine clamps when reading confidence.
            "confidence": {"type": "number"},
            "at_bowl": {"type": "boolean"},
            "other_animals_present": {"type": "array", "items": {"type": "string", "enum": keys}},
            "reason": {"type": "string"},
        },
        "required": ["animal", "confidence", "at_bowl", "other_animals_present", "reason"],
        "additionalProperties": False,
    }


def feeding_schema(other_keys: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "grrr_at_bowl": {"type": "boolean"},
            "bowl": {"type": "string", "enum": ["food", "empty", "unsure"]},
            "other_animals_present": {
                "type": "array",
                "items": {"type": "string", "enum": other_keys},
            },
            "reason": {"type": "string"},
        },
        "required": ["grrr_at_bowl", "bowl", "other_animals_present", "reason"],
        "additionalProperties": False,
    }


def identify_question(target_name: str) -> str:
    return (
        "Which animal, if any, is in the most recent frames, and is it at the bowl? "
        f"The animal that may be fed is {target_name}. "
        "List any other animals visible in other_animals_present."
    )


def feeding_question(target_name: str) -> str:
    return (
        f"The feeder lid is open. Is {target_name} (the small black dog) still at the bowl? "
        "Is there still food visible in the open bowl, or has it been eaten? "
        "List any other animals in other_animals_present if they are visible."
    )


def verify_question(target_name: str) -> str:
    return (
        "The feeder lid just opened to serve a plate. Look only at the bowl under the open lid "
        "and judge what was SERVED, ignoring the animal: set bowl='food' only if the bowl clearly "
        "contains wet food, bowl='empty' if the bowl is clearly bare/empty (a plate with no "
        "food), and bowl='unsure' if you cannot tell (dark, obscured, blurry). When unsure, say "
        "'unsure' — do NOT guess 'empty', because that would rotate the tray and waste a good "
        f"plate. Set grrr_at_bowl true if {target_name} is eating from it."
    )


# USD per million tokens: (input, output). Cache writes with a 1h TTL cost 2x input,
# cache reads cost 0.1x input. Used for the dashboard estimate only.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
CACHE_TTL = "1h"


def estimate_cost_usd(model: str, usage: dict[str, int]) -> float:
    price_in, price_out = PRICES.get(model, PRICES["claude-opus-5"])
    return (
        usage.get("input_tokens", 0) * price_in
        + usage.get("cache_creation_input_tokens", 0) * price_in * 2.0
        + usage.get("cache_read_input_tokens", 0) * price_in * 0.1
        + usage.get("output_tokens", 0) * price_out
    ) / 1_000_000


def to_jpeg(data: bytes, max_edge: int = REFERENCE_MAX_EDGE) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((max_edge, max_edge))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=80)
    return out.getvalue()


def image_block(jpeg: bytes) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(jpeg).decode("ascii"),
        },
    }


def load_reference_blocks(
    reference_dir: Path, personas: list[Persona]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Content blocks for every reference photo, grouped and labelled per persona."""
    blocks: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for persona in personas:
        paths = sorted(
            p
            for p in (reference_dir / persona.key).glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        counts[persona.key] = len(paths)
        label = persona.description or persona.name
        blocks.append({"type": "text", "text": f"Reference photos of {label}"})
        for path in paths:
            blocks.append(image_block(to_jpeg(path.read_bytes())))
        if not paths:
            blocks.append({"type": "text", "text": "(no reference photos yet)"})
    blocks.append(
        {
            "type": "text",
            "text": "End of reference photos.",
            "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL},
        }
    )
    return blocks, counts


class ClaudeIdentifier:
    def __init__(
        self,
        reference_dir: Path,
        store: Store,
        model: Callable[[], str],
        clock: Callable[[], datetime] = datetime.now,
        client: anthropic.Anthropic | None = None,
        personas: Callable[[], list[Persona]] | None = None,
    ):
        self.reference_dir = reference_dir
        self.store = store
        self.model = model
        self.clock = clock
        self.client = client or anthropic.Anthropic()
        self.personas = personas or (lambda: [])
        self.reference_blocks: list[dict[str, Any]] = []
        self.reference_counts: dict[str, int] = {}
        self._personas_used: list[dict] | None = None
        self.identify_schema: dict[str, Any] = identify_schema([])
        self.feeding_schema: dict[str, Any] = feeding_schema([])
        self.target_name: str = "Grrr"
        self.reload_references()

    def reload_references(self) -> None:
        personas = self.personas()
        self.reference_blocks, self.reference_counts = load_reference_blocks(
            self.reference_dir, personas
        )
        keys = [p.key for p in personas]
        other_keys = [p.key for p in personas if p.key != TARGET_KEY]
        self.identify_schema = identify_schema(keys)
        self.feeding_schema = feeding_schema(other_keys)
        self.target_name = next((p.name for p in personas if p.key == TARGET_KEY), "Grrr")
        self._personas_used = [p.to_dict() for p in personas]
        log.info("reference photos: %s", self.reference_counts)

    def _maybe_reload(self) -> None:
        if [p.to_dict() for p in self.personas()] != self._personas_used:
            self.reload_references()  # a dashboard edit → rebuild the cached prefix + schemas

    def identify(self, frames: list[Frame]) -> Verdict:
        self._maybe_reload()
        q = identify_question(self.target_name)
        data = self._ask("identify", frames, q, self.identify_schema)
        return Verdict.from_json(data) if data else Verdict(reason="llm call failed")

    def feeding_check(self, frames: list[Frame]) -> FeedingVerdict:
        self._maybe_reload()
        q = feeding_question(self.target_name)
        data = self._ask("feeding", frames, q, self.feeding_schema)
        return FeedingVerdict.from_json(data) if data else FeedingVerdict(reason="llm call failed")

    def verify_food(self, frames: list[Frame]) -> FeedingVerdict:
        self._maybe_reload()
        # "llm call failed" -> bowl defaults to "unsure", so a failed call never rotates the tray.
        data = self._ask("verify", frames, verify_question(self.target_name), self.feeding_schema)
        return FeedingVerdict.from_json(data) if data else FeedingVerdict(reason="llm call failed")

    def bed_check(self, frames: list[Frame], empty_reference: bytes | None) -> BedVerdict:
        # Its own prompt and no animal reference photos: those are overhead shots of the bowl.
        # What helps here is a picture of the same beds with nobody in them.
        prefix: list[dict[str, Any]] = []
        if empty_reference:
            prefix = [
                {"type": "text", "text": "Reference photo, same camera, BOTH beds empty:"},
                image_block(empty_reference),
            ]
        data = self._ask(
            "bed", frames, BED_QUESTION, BED_SCHEMA, system=BED_SYSTEM_PROMPT, prefix=prefix
        )
        return BedVerdict.from_json(data) if data else BedVerdict(reason="llm call failed")

    def _ask(
        self,
        purpose: str,
        frames: list[Frame],
        question: str,
        schema: dict[str, Any],
        *,
        system: str = SYSTEM_PROMPT,
        prefix: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if not frames:
            return None
        model = self.model()
        content: list[dict[str, Any]] = list(self.reference_blocks if prefix is None else prefix)
        content.append({"type": "text", "text": "Current camera frames, oldest first:"})
        content.extend(image_block(frame.jpeg) for frame in frames)
        content.append({"type": "text", "text": question})
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=VERDICT_MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL},
                    }
                ],
                messages=[{"role": "user", "content": content}],
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.APIError as exc:
            log.warning("llm %s call failed: %s", purpose, exc)
            self.store.add_llm_call(self.clock(), purpose, model, {"error": str(exc)}, {}, 0.0)
            return None
        usage = response.usage.model_dump(exclude_none=True)
        usage = {k: v for k, v in usage.items() if isinstance(v, int)}
        cost = estimate_cost_usd(model, usage)
        if response.stop_reason == "refusal":
            verdict: dict[str, Any] = {"error": "refusal"}
        else:
            text = next((b.text for b in response.content if b.type == "text"), "{}")
            verdict = _parse_json(text)
        log.info("llm %s -> %s (cost $%.4f, usage %s)", purpose, verdict, cost, usage)
        self.store.add_llm_call(self.clock(), purpose, model, verdict, usage, cost)
        return None if "error" in verdict else verdict


def _parse_json(text: str) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"error": f"non-json response: {text[:200]}"}
    return parsed if isinstance(parsed, dict) else {"error": "non-object response"}
