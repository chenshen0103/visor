
import pytest
from modules.text.text_detector import TextDetector

@pytest.fixture(scope="module")
def detector():
    d = TextDetector()
    d.load()
    return d

def test_history_influence_guess_who(detector):
    # Standalone message: quite innocent/ambiguous
    text = "我是你大學同學啦，我換電話了，先存一下"
    verdict_no_hist = detector.analyze(text)
    
    # Same message with leading "Guess who I am" context
    history = ["猜猜我是誰？"]
    verdict_with_hist = detector.analyze(text, history=history)
    
    print(f"\nStandalone Score: {verdict_no_hist.confidence if verdict_no_hist.is_scam else 1-verdict_no_hist.confidence:.4f}")
    print(f"With History Score: {verdict_with_hist.confidence if verdict_with_hist.is_scam else 1-verdict_with_hist.confidence:.4f}")
    
    # Score with history should be >= score without history for this scam pattern
    assert verdict_with_hist.confidence >= verdict_no_hist.confidence
    assert "context" in verdict_with_hist.explanation.lower()

def test_history_influence_investment(detector):
    # Innocent question
    text = "你要不要試試看？"
    verdict_no_hist = detector.analyze(text)
    
    # Context of high returns
    history = [
        "我在新加坡工作，賺了很多錢",
        "我有個朋友教我的投資方法，保證每月30%回報"
    ]
    verdict_with_hist = detector.analyze(text, history=history)
    
    print(f"\nInnocent Standalone Score: {verdict_no_hist.confidence if verdict_no_hist.is_scam else 1-verdict_no_hist.confidence:.4f}")
    print(f"Scam Context Score: {verdict_with_hist.confidence if verdict_with_hist.is_scam else 1-verdict_with_hist.confidence:.4f}")
    
    assert verdict_with_hist.status in ("scam", "suspicious")
    assert verdict_no_hist.status == "safe" or verdict_with_hist.confidence > verdict_no_hist.confidence
