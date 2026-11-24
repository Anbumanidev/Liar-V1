"""FinBERT headline sentiment (ProsusAI/finbert), lazily loaded, never crashing.

* The ``transformers`` text-classification pipeline is created on first use (model download /
  load happens then, not at import). Device ``"auto"`` selects ``cuda:0`` when
  ``torch.cuda.is_available()`` else ``cpu``.
* Headlines are scored in batches; ``sentiment_score = P(positive) - P(negative)`` in
  ``[-1, 1]``; ``sentiment_label`` = arg-max class (positive -> BULLISH, negative -> BEARISH,
  neutral -> NEUTRAL).
* If transformers / torch / the model are unavailable, or inference fails, the scorer marks
  itself ``available = False``, logs the reason once, and returns headlines labelled NEUTRAL
  with ``sentiment_score = None`` (graceful degradation, rule 16).
* Tests inject a fake pipeline via ``pipeline=`` (or ``pipeline_factory=``) - nothing is
  downloaded.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from typing import Any

from loguru import logger

from core.enums import Signal
from core.types import Headline
from storage.audit import record_error

DEFAULT_MODEL = "ProsusAI/finbert"

_LABEL_MAP: dict[str, Signal] = {
    "positive": Signal.BULLISH,
    "negative": Signal.BEARISH,
    "neutral": Signal.NEUTRAL,
}

PipelineFn = Callable[..., Any]
PipelineFactory = Callable[[str, str], PipelineFn]  # (model_name, device) -> pipeline


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to ``"cuda:0"`` / ``"cpu"``; other values pass through."""
    if device != "auto":
        return "cuda:0" if device == "cuda" else device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # torch missing or broken CUDA install
        logger.info(f"torch unavailable for device detection ({exc}); using cpu")
        return "cpu"


def _default_factory(model_name: str, device: str) -> PipelineFn:
    from transformers import pipeline  # heavy import, only on first use

    return pipeline("text-classification", model=model_name, tokenizer=model_name, device=device, top_k=None)


def _probabilities(result: Any) -> dict[str, float]:
    """Normalise one pipeline output (list of {label, score} or a single dict) to label->prob."""
    items = result if isinstance(result, list) else [result]
    probs: dict[str, float] = {}
    for item in items:
        if isinstance(item, dict) and "label" in item and "score" in item:
            probs[str(item["label"]).lower()] = float(item["score"])
    return probs


class FinBERTScorer:
    """Lazy FinBERT scorer. Thread-safe; safe to share across agents."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        *,
        batch_size: int = 16,
        include_summary: bool = False,
        pipeline: PipelineFn | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.include_summary = include_summary
        self._pipe: PipelineFn | None = pipeline
        self._factory = pipeline_factory or _default_factory
        self._lock = threading.Lock()
        self._load_failed = False
        self.last_error: str | None = None

    # ------------------------------------------------------------------ loading

    @property
    def available(self) -> bool:
        """True when the model is loaded or still loadable (no failure so far)."""
        return not self._load_failed

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def _ensure_loaded(self) -> PipelineFn | None:
        if self._pipe is not None:
            return self._pipe
        if self._load_failed:
            return None
        with self._lock:
            if self._pipe is not None or self._load_failed:
                return self._pipe
            try:
                dev = resolve_device(self.device)
                logger.info(f"loading FinBERT model {self.model_name} on {dev}")
                self._pipe = self._factory(self.model_name, dev)
            except Exception as exc:  # missing transformers / no model / no network
                self._mark_failed("load", exc)
        return self._pipe

    def _mark_failed(self, stage: str, exc: BaseException) -> None:
        self._load_failed = True
        self.last_error = f"{stage}: {exc}"
        record_error(None, "features.sentiment", exc, {"stage": stage, "model": self.model_name},
                     severity="WARNING")

    # ------------------------------------------------------------------ scoring

    def _text(self, h: Headline) -> str:
        if self.include_summary and h.summary:
            return f"{h.title}. {h.summary}"
        return h.title

    def score_texts(self, texts: Sequence[str]) -> list[tuple[Signal, float] | None]:
        """Score raw texts. Returns ``(label, score)`` per text, or None when unavailable."""
        if not texts:
            return []
        pipe = self._ensure_loaded()
        if pipe is None:
            return [None] * len(texts)
        out: list[tuple[Signal, float] | None] = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i:i + self.batch_size])
                results = pipe(batch, truncation=True)
                if not isinstance(results, list) or len(results) != len(batch):
                    raise ValueError(f"unexpected pipeline output for batch of {len(batch)}")
                for res in results:
                    probs = _probabilities(res)
                    if not probs:
                        out.append(None)
                        continue
                    score = probs.get("positive", 0.0) - probs.get("negative", 0.0)
                    score = max(-1.0, min(1.0, score))
                    top = max(probs, key=lambda k: probs[k])
                    out.append((_LABEL_MAP.get(top, Signal.NEUTRAL), score))
        except Exception as exc:  # inference failure -> degrade to neutral
            self._mark_failed("inference", exc)
            return [None] * len(texts)
        return out

    def score(self, headlines: Sequence[Headline]) -> list[Headline]:
        """Return copies of ``headlines`` with ``sentiment_label`` / ``sentiment_score`` set.

        When the model is unavailable every headline is labelled NEUTRAL with score None.
        """
        if not headlines:
            return []
        scored = self.score_texts([self._text(h) for h in headlines])
        out: list[Headline] = []
        for h, res in zip(headlines, scored):
            if res is None:
                out.append(h.model_copy(update={"sentiment_label": Signal.NEUTRAL, "sentiment_score": None}))
            else:
                label, score = res
                out.append(h.model_copy(update={"sentiment_label": label, "sentiment_score": round(score, 6)}))
        return out

    @staticmethod
    def aggregate(headlines: Sequence[Headline]) -> float | None:
        """Mean ``sentiment_score`` of scored headlines (None when none are scored)."""
        vals = [h.sentiment_score for h in headlines
                if h.sentiment_score is not None and math.isfinite(h.sentiment_score)]
        if not vals:
            return None
        return max(-1.0, min(1.0, sum(vals) / len(vals)))


def aggregate(headlines: Sequence[Headline]) -> float | None:
    """Module-level alias of :meth:`FinBERTScorer.aggregate`."""
    return FinBERTScorer.aggregate(headlines)
