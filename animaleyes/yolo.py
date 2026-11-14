"""Local YOLO detector backend — a free, real-time alternative to the cloud LLM.

A pretrained COCO model gives us dog / cat / person with bounding boxes. We map that onto the
project's identities: a cat is the cat, a person is ignored, and a dog is Grrr or Bowie by
box size (Grrr is tiny, Bowie medium/lanky — at a fixed camera distance the box area separates
them). The mapping is a pure function (`verdict_from_detections`) so it is unit-tested without
loading PyTorch; the model is lazy-imported only when this backend is actually selected.

Identity: with YOLO_CLASSIFIER set (a model trained by tools/train_classifier.py on the frames
the human tagged), each detected animal's crop is classified as grrr / bowie / cat / ..., so a
dog close to the camera is no longer "Bowie" just because its box is big. Below YOLO_CLS_MIN_CONF
a dog is "unsure" — which never opens the feeder, and which the cascade hands to Claude. Without
a classifier the old size heuristic applies (Grrr is tiny, Bowie medium/lanky).
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from .camera import Frame
from .detect import BedVerdict, FeedingVerdict, Verdict

log = logging.getLogger(__name__)

# COCO class names we care about.
DOG, CAT, PERSON = "dog", "cat", "person"


CROP_MARGIN = 0.12  # pad a detection box by this fraction of its size before classifying


def crop_box(img, box: tuple[float, float, float, float]):
    """Crop a detection box out of a PIL image with a margin, clamped to the frame. Shared by
    training and inference so the classifier sees the same framing both times."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
    w, h = img.size
    return img.crop(
        (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(w, int(x2 + mx)), min(h, int(y2 + my)))
    )


@dataclass
class Detection:
    label: str
    confidence: float
    box_fraction: float  # bbox area / image area, 0..1
    in_bowl_zone: bool = True
    box: tuple[float, float, float, float] | None = None  # pixels, for the identity crop
    identity: str | None = None  # classifier's answer for this animal (persona key)
    identity_conf: float = 0.0


@dataclass
class YoloConfig:
    min_conf: float = 0.4
    # A dog whose box covers at most this fraction of the frame is Grrr (tiny); larger is Bowie.
    # Only used when no classifier identity is available.
    grrr_max_box_fraction: float = 0.18
    classifier_path: str = ""  # trained identity model; empty = size heuristic
    cls_min_conf: float = 0.6  # below this the classifier's answer is "unsure"


def verdict_from_detections(dets: list[Detection], cfg: YoloConfig) -> Verdict:
    """Pure mapping from detections to a Verdict. No model, no I/O — fully testable."""
    animals = [d for d in dets if d.confidence >= cfg.min_conf and d.label in (DOG, CAT)]
    if not animals:
        return Verdict(animal="none", confidence=0.0, at_bowl=False, reason="no animal detected")

    cats = [d for d in animals if d.label == CAT]
    dogs = [d for d in animals if d.label == DOG]

    def who(d: Detection) -> tuple[str, float]:
        """Identity + confidence for one animal: the classifier's answer when it is sure, else
        the old size rule for dogs / "cat" for cats. An unsure dog is "unsure", never a guess."""
        if d.identity is not None:
            if d.identity_conf >= cfg.cls_min_conf:
                return d.identity, d.identity_conf
            return ("cat" if d.label == CAT else "unsure"), d.identity_conf
        if d.label == CAT:
            return "cat", d.confidence
        return ("grrr" if d.box_fraction <= cfg.grrr_max_box_fraction else "bowie"), d.confidence

    # Choose the subject: the dog if present (that's who we might feed), else the cat.
    best_dog = max(dogs, key=lambda d: d.confidence) if dogs else None
    best = best_dog if best_dog is not None else max(cats, key=lambda d: d.confidence)
    subject, conf = who(best)
    present: list[str] = []
    for d in animals:
        name, _ = who(d)
        if name not in present:
            present.append(name)
    others = [a for a in present if a != subject]
    how = f"id {best.identity_conf:.2f}" if best.identity is not None else "size rule"
    reason = f"yolo: {subject} (det {best.confidence:.2f}, {how}, box {best.box_fraction:.2f})"
    in_zone = best.in_bowl_zone
    return Verdict(
        animal=subject,
        confidence=conf,
        at_bowl=in_zone,
        other_animals_present=others,
        reason=reason,
    )


class YoloIdentifier:
    """Runs a local YOLO model and maps its detections onto the Identifier interface."""

    def __init__(
        self,
        model_path: str,
        cfg: YoloConfig,
        model=None,
        classifier=None,
        settings: Callable[[], YoloConfig] | None = None,
    ):
        self.cfg = cfg
        self.model_path = model_path
        self._model = model  # injectable for tests; lazy-loaded otherwise
        self._classifier = classifier
        self._classifier_path = cfg.classifier_path if classifier is not None else ""
        self._bad_classifier_path = ""  # a path that failed to load: warned once, not retried
        # Re-read before every look so dashboard edits (thresholds, the classifier path after a
        # retrain) take effect on the next tick, like every other knob. A fixed cfg is for tests.
        self._settings = settings
        self.reference_counts: dict[str, int] = {}  # for dashboard parity with Claude

    def _refresh(self) -> None:
        if self._settings is not None:
            self.cfg = self._settings()

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy: only needed when this backend is used

            log.info("loading YOLO model %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def _ensure_classifier(self):
        """The identity classifier, or None when none is configured. Reloaded if the configured
        path changes (a retrain + dashboard edit takes effect without a restart)."""
        path = self.cfg.classifier_path
        if not path or path == self._bad_classifier_path:
            return None
        if self._classifier is None or path != self._classifier_path:
            log.info("loading animal classifier %s", path)
            try:
                self._classifier = self._load_classifier(path)
            except Exception as exc:  # noqa: BLE001 - a missing/bad model must not stop feeding
                log.warning("animal classifier unavailable (%s); using the size rule", exc)
                self._classifier = None
                self._bad_classifier_path = path
                return None
            self._classifier_path = path
        return self._classifier

    def _load_classifier(self, path: str):
        from ultralytics import YOLO

        return YOLO(path)

    def _identify_crop(self, img, det: Detection) -> None:
        clf = self._ensure_classifier()
        if clf is None or det.box is None or det.label not in (DOG, CAT):
            return
        r = clf.predict(crop_box(img, det.box), verbose=False, imgsz=224)[0]
        det.identity = str(r.names[int(r.probs.top1)])
        det.identity_conf = float(r.probs.top1conf)

    def _detect(self, frame: Frame) -> list[Detection]:
        from PIL import Image

        self._refresh()
        img = Image.open(io.BytesIO(frame.jpeg)).convert("RGB")
        w, h = img.size
        area = float(w * h) or 1.0
        started = time.perf_counter()
        results = self._ensure_model().predict(img, verbose=False)
        took_ms = (time.perf_counter() - started) * 1000
        dets: list[Detection] = []
        for r in results:
            names = r.names
            for b in r.boxes:
                label = names[int(b.cls[0])]
                if label not in (DOG, CAT, PERSON):
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                frac = max(0.0, (x2 - x1) * (y2 - y1)) / area
                det = Detection(
                    label=label,
                    confidence=float(b.conf[0]),
                    box_fraction=frac,
                    box=(x1, y1, x2, y2),
                )
                if det.confidence >= self.cfg.min_conf:
                    self._identify_crop(img, det)
                dets.append(det)
        # One line per look, so a missed visit can be explained afterwards: what the gate saw
        # (including detections under YOLO_MIN_CONF, which it then ignores) and how long it took.
        seen = ", ".join(
            f"{d.label} {d.confidence:.2f} box {d.box_fraction:.2f}"
            + (f" -> {d.identity} {d.identity_conf:.2f}" if d.identity else "")
            for d in dets
        )
        log.info(
            "yolo saw: %s (%.0f ms, min conf %.2f)", seen or "nothing", took_ms, self.cfg.min_conf
        )
        return dets

    def identify(self, frames: list[Frame]) -> Verdict:
        if not frames:
            return Verdict(animal="none", reason="no frame")
        return verdict_from_detections(self._detect(frames[-1]), self.cfg)

    def feeding_check(self, frames: list[Frame]) -> FeedingVerdict:
        # YOLO can't judge whether the bowl still has food; report presence only and let the
        # motion/absence timing decide. "bowl: unsure" keeps the eaten/offer-again logic safe.
        if not frames:
            return FeedingVerdict(grrr_at_bowl=False, reason="no frame")
        v = verdict_from_detections(self._detect(frames[-1]), self.cfg)
        return FeedingVerdict(
            grrr_at_bowl=(v.animal == "grrr"),
            bowl="unsure",
            other_animals_present=[a for a in v.other_animals_present if a != "grrr"],
            reason=v.reason,
        )

    def verify_food(self, frames: list[Frame]) -> FeedingVerdict:
        # YOLO/COCO has no food class, so it can't tell a served bowl from an empty plate.
        # "unsure" means the state machine won't rotate the tray on YOLO's say-so.
        return FeedingVerdict(bowl="unsure", reason="yolo cannot judge food in the bowl")

    def bed_check(self, frames: list[Frame], empty_reference: bytes | None) -> BedVerdict:
        # Tried on the real bed camera: a black dog curled up under the chair, half hidden
        # behind the front bed in IR, is not a "dog" to COCO. Say so rather than report "empty".
        return BedVerdict(reason="the local detector cannot judge the bed")

    def reload_references(self) -> None:
        pass  # no reference imagery for the detector backend
