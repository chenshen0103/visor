
import pytest
from modules.text.text_detector import TextDetector

@pytest.fixture(scope="module")
def detector():
    d = TextDetector()
    d.load()
    return d

def test_llm_explanation_generation(detector):
    # Test with a clear investment scam
    text = "保證月報酬30%，零風險，趕快加LINE了解更多"
    verdict = detector.analyze(text)
    
    print(f"\nText: {text}")
    print(f"Status: {verdict.status}")
    print(f"LLM Explanation: {verdict.llm_explanation}")
    
    assert verdict.status == "scam"
    assert verdict.llm_explanation is not None
    assert len(verdict.llm_explanation) > 10
    # Should be in Traditional Chinese
    assert "詐騙" in verdict.llm_explanation or "風險" in verdict.llm_explanation

def test_llm_explanation_with_history(detector):
    text = "你要不要試試看？"
    history = ["保證月報酬30%，零風險", "我的投資平台很安全"]
    verdict = detector.analyze(text, history=history)
    
    print(f"\nText with History: {text}")
    print(f"Status: {verdict.status}")
    print(f"LLM Explanation: {verdict.llm_explanation}")
    
    assert verdict.status == "scam"
    assert verdict.llm_explanation is not None
