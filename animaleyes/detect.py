"""Detector-neutral types and the Identifier interface.

These live apart from any specific backend (Claude, YOLO, …) so a detector implementation
only depends on this module, not on the cloud client. The state machine is written against
`Identifier`; swapping backends is a config choice, not a code change. See `build_identifier`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .camera import Frame

# The identities the state machine reasons about. A detector maps whatever it sees onto these.
ANIMALS = ("grrr", "bowie", "cat")


@dataclass
class Verdict:
    animal: str = "unsure"
    confidence: float = 0.0
    at_bowl: bool = False
    other_animals_present: list[str] = field(default_factory=list)
    reason: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Verdict:
        return cls(
            animal=str(data.get("animal", "unsure")),
            # Clamp to 0..1 since the LLM schema can no longer enforce the range.
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            at_bowl=bool(data.get("at_bowl", False)),
            other_animals_present=[str(a) for a in data.get("other_animals_present", [])],
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "animal": self.animal,
            "confidence": self.confidence,
            "at_bowl": self.at_bowl,
            "other_animals_present": self.other_animals_present,
            "reason": self.reason,
        }

    def is_grrr(self, min_confidence: float, require_at_bowl: bool = True) -> bool:
        return (
            self.animal == "grrr"
            and self.confidence >= min_confidence
            and (self.at_bowl or not require_at_bowl)
            and not self.other_animals_present
        )


@dataclass
class FeedingVerdict:
    grrr_at_bowl: bool = False
    bowl: str = "unsure"
    other_animals_present: list[str] = field(default_factory=list)
    reason: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FeedingVerdict:
        return cls(
            grrr_at_bowl=bool(data.get("grrr_at_bowl", False)),
            bowl=str(data.get("bowl", "unsure")),
            other_animals_present=[str(a) for a in data.get("other_animals_present", [])],
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grrr_at_bowl": self.grrr_at_bowl,
            "bowl": self.bowl,
            "other_animals_present": self.other_animals_present,
            "reason": self.reason,
        }


@dataclass
class BedVerdict:
    """What the bed camera shows. `occupant` is "grrr", "other" (a different animal is in a
    bed), "empty", or "unsure"; `where` is "under_chair", "front_bed" or "none"."""

    occupant: str = "unsure"
    where: str = "none"
    reason: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BedVerdict:
        return cls(
            occupant=str(data.get("occupant", "unsure")),
            where=str(data.get("where", "none")),
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"occupant": self.occupant, "where": self.where, "reason": self.reason}


class Identifier(Protocol):
    """What the state machine needs from any detector backend."""

    def identify(self, frames: list[Frame]) -> Verdict: ...
    def feeding_check(self, frames: list[Frame]) -> FeedingVerdict: ...

    # Right after a plate is served: does the served bowl actually contain food? Reuses the
    # FeedingVerdict shape (its `bowl` field is "food"/"empty"/"unsure"). Only a backend that
    # can judge food (Claude) answers meaningfully; others return "unsure".
    def verify_food(self, frames: list[Frame]) -> FeedingVerdict: ...

    # The bed camera: is Grrr in one of her beds? `empty_reference` is a JPEG from the same
    # camera with both beds empty, to compare against (None when none has been saved). Only a
    # backend that can read a dark, half-hidden dog (Claude) answers; others return "unsure".
    def bed_check(self, frames: list[Frame], empty_reference: bytes | None) -> BedVerdict: ...

    # Optional, for backends with reference imagery (Claude). YOLO no-ops these.
    def reload_references(self) -> None: ...
