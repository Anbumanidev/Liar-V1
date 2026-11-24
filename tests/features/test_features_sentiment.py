"""FinBERT scorer with an injected fake pipeline (never downloads a model)."""

from __future__ import annotations

from typing import Any

import pytest

from core.enums import Signal
from core.types import Headline
from features.sentiment import FinBERTScorer, aggregate, resolve_device
from tests.helpers import ist


def _h(title: str) -> Headline:
    return Headline(source="test", title=title, published_at=ist(2026, 11, 24, 9, 0))


class FakePipe:
    """Mimics transformers text-classification with top_k=None (list of label/score per text)."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> list[list[dict[str, float | str]]]:
        assert kwargs.get("truncation") is True
        self.batches.append(list(texts))
        out = []
        for t in texts:
            if "surge" in t:
                probs = {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
            elif "crash" in t:
                probs = {"positive": 0.05, "negative": 0.9, "neutral": 0.05}
            else:
                probs = {"positive": 0.2, "negative": 0.1, "neutral": 0.7}
            out.append([{"label": k, "score": v} for k, v in probs.items()])
        return out


def test_scores_and_labels() -> None:
    pipe = FakePipe()
    scorer = FinBERTScorer(pipeline=pipe)
    out = scorer.score([_h("Profits surge"), _h("Shares crash"), _h("AGM held")])
    assert [h.sentiment_label for h in out] == [Signal.BULLISH, Signal.BEARISH, Signal.NEUTRAL]
    assert out[0].sentiment_score == pytest.approx(0.7)
    assert out[1].sentiment_score == pytest.approx(-0.85)
    assert out[2].sentiment_score == pytest.approx(0.1)
    assert scorer.available and scorer.loaded
    assert FinBERTScorer.aggregate(out) == pytest.approx((0.7 - 0.85 + 0.1) / 3)
    assert aggregate(out) == FinBERTScorer.aggregate(out)


def test_batching() -> None:
    pipe = FakePipe()
    scorer = FinBERTScorer(pipeline=pipe, batch_size=2)
    scorer.score([_h(f"news {i}") for i in range(5)])
    assert [len(b) for b in pipe.batches] == [2, 2, 1]


def test_lazy_load_via_factory() -> None:
    created: list[tuple[str, str]] = []

    def factory(model: str, device: str) -> FakePipe:
        created.append((model, device))
        return FakePipe()

    scorer = FinBERTScorer("ProsusAI/finbert", device="cpu", pipeline_factory=factory)
    assert not created and not scorer.loaded  # nothing loaded at construction
    scorer.score([_h("surge")])
    scorer.score([_h("crash")])
    assert created == [("ProsusAI/finbert", "cpu")]  # loaded exactly once


def test_unavailable_model_degrades_to_neutral() -> None:
    def factory(model: str, device: str) -> Any:
        raise OSError("no internet / model not cached")

    scorer = FinBERTScorer(pipeline_factory=factory, device="cpu")
    out = scorer.score([_h("Profits surge"), _h("x")])
    assert scorer.available is False
    assert all(h.sentiment_label == Signal.NEUTRAL and h.sentiment_score is None for h in out)
    assert FinBERTScorer.aggregate(out) is None
    assert "load" in (scorer.last_error or "")


def test_inference_failure_degrades() -> None:
    def bad_pipe(texts: list[str], **kw: Any) -> Any:
        raise RuntimeError("CUDA OOM")

    scorer = FinBERTScorer(pipeline=bad_pipe)
    out = scorer.score([_h("surge")])
    assert out[0].sentiment_label == Signal.NEUTRAL and out[0].sentiment_score is None
    assert scorer.available is False


def test_single_label_output_supported() -> None:
    def pipe(texts: list[str], **kw: Any) -> list[dict[str, Any]]:
        return [{"label": "Negative", "score": 0.6} for _ in texts]

    out = FinBERTScorer(pipeline=pipe).score([_h("a")])
    assert out[0].sentiment_label == Signal.BEARISH
    assert out[0].sentiment_score == pytest.approx(-0.6)


def test_empty_and_original_untouched() -> None:
    scorer = FinBERTScorer(pipeline=FakePipe())
    assert scorer.score([]) == []
    h = _h("surge")
    scorer.score([h])
    assert h.sentiment_score is None  # copies are returned


def test_include_summary() -> None:
    pipe = FakePipe()
    scorer = FinBERTScorer(pipeline=pipe, include_summary=True)
    h = Headline(source="t", title="Q2 results", summary="profits surge", published_at=ist(2026, 11, 24))
    out = scorer.score([h])
    assert pipe.batches[0] == ["Q2 results. profits surge"]
    assert out[0].sentiment_label == Signal.BULLISH


def test_resolve_device() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda:0"
    assert resolve_device("auto") in ("cpu", "cuda:0")
