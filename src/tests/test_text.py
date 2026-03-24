"""
Unit tests for the text scam detection module.

IntentEmbedder requires sentence-transformers to be installed.
Tests are marked to skip gracefully if the model cannot be downloaded.
"""

import pytest

from modules.text.scam_patterns import SCAM_ARCHETYPES, ARCHETYPE_BY_KEY


# ---------------------------------------------------------------------------
# scam_patterns tests (no model required)
# ---------------------------------------------------------------------------

class TestScamPatterns:
    def test_archetype_count(self):
        assert len(SCAM_ARCHETYPES) >= 9

    def test_required_keys_present(self):
        expected_keys = {
            "investment_fraud",
            "romance_fraud",
            "government_impersonation",
            "parcel_fraud",
            "recovery_fraud",
            "guess_who_i_am",
            "atm_deduction_fraud",
            "job_scam",
            "phishing_link_fraud",
        }
        actual_keys = {a.key for a in SCAM_ARCHETYPES}
        assert expected_keys.issubset(actual_keys)

    def test_each_archetype_has_exemplars(self):
        for archetype in SCAM_ARCHETYPES:
            assert len(archetype.exemplars) >= 4, (
                f"Archetype '{archetype.key}' should have ≥4 exemplars"
            )

    def test_archetype_by_key_lookup(self):
        arch = ARCHETYPE_BY_KEY["government_impersonation"]
        assert arch.name_zh == "假冒公務機關詐騙"


# ---------------------------------------------------------------------------
# IntentEmbedder tests
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer as _ST  # noqa
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

needs_st = pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")


@needs_st
class TestIntentEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        from modules.text.intent_embedder import IntentEmbedder
        e = IntentEmbedder()
        e.load()
        return e

    def test_embed_returns_correct_shape(self, embedder):
        vec = embedder.embed("投資詐騙測試文字")
        assert vec.shape == (384,)

    def test_embed_is_unit_norm(self, embedder):
        import numpy as np
        vec = embedder.embed("test text")
        assert abs(float((vec**2).sum()**0.5) - 1.0) < 1e-3

    def test_government_impersonation_detected(self, embedder):
        text = "您好，刑事局通知您帳戶涉嫌洗錢，請配合轉帳至監管帳戶。"
        result = embedder.compute_scam_distances(text)
        assert result.closest_archetype == "government_impersonation"
        assert result.max_similarity > 0.4

    def test_investment_fraud_detected(self, embedder):
        text = "我們的平台保證每月30%回報，零風險投資機會，快來加入"
        result = embedder.compute_scam_distances(text)
        assert result.closest_archetype == "investment_fraud"

    def test_safe_text_low_similarity(self, embedder):
        text = "今天天氣很好，我想去公園散步。"
        result = embedder.compute_scam_distances(text)
        assert result.max_similarity < 0.6

    def test_parcel_fraud_detected(self, embedder):
        text = "您的包裹因地址不完整無法投遞，請點擊連結更新資料並支付補充運費。"
        result = embedder.compute_scam_distances(text)
        assert result.closest_archetype == "parcel_fraud"


# ---------------------------------------------------------------------------
# TextDetector (integration, requires sentence-transformers)
# ---------------------------------------------------------------------------

@needs_st
class TestTextDetector:
    @pytest.fixture(scope="class")
    def detector(self):
        from modules.text.text_detector import TextDetector
        d = TextDetector()
        d.load()
        return d

    def test_scam_text_classified(self, detector):
        text = "您好，刑事局通知您帳戶涉嫌洗錢，請配合轉帳至監管帳戶。"
        verdict = detector.analyze(text)
        assert verdict.status in ("scam", "suspicious")
        assert verdict.confidence > 0.4

    def test_safe_text_classified(self, detector):
        text = "今天台北天氣晴，最高溫28度，適合戶外活動。"
        verdict = detector.analyze(text)
        assert verdict.status in ("safe", "suspicious")

    def test_verdict_has_all_fields(self, detector):
        text = "限時投資機會，保證月報酬30%！"
        verdict = detector.analyze(text)
        assert verdict.closest_archetype != ""
        assert isinstance(verdict.processing_time_ms, float)
        assert isinstance(verdict.explanation, str)

    def test_empty_like_text(self, detector):
        """Single character should not crash."""
        verdict = detector.analyze("A")
        assert verdict.status in ("scam", "safe", "suspicious")
