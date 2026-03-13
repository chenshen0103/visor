
import pytest
from modules.text.text_detector import TextDetector
from modules.text.red_flags import RedFlagAnalyzer

@pytest.fixture(scope="module")
def detector():
    d = TextDetector()
    d.load()
    return d

def test_url_red_flags(detector):
    # Message with suspicious TLD
    text = "請點擊連結領取獎勵: http://win-prize.top"
    verdict = detector.analyze(text)
    
    assert any(rf.key == "suspicious_tld" for rf in verdict.red_flags)
    assert verdict.status == "scam"

def test_urgency_red_flags(detector):
    # Message with urgency
    text = "您的帳戶即將逾期，請馬上處理"
    verdict = detector.analyze(text)
    
    assert any(rf.key == "urgency" for rf in verdict.red_flags)

def test_sensitive_action_red_flags(detector):
    # Message with sensitive actions
    text = "請配合轉帳至安全帳戶，並提供您的驗證碼"
    verdict = detector.analyze(text)
    
    assert any(rf.key == "sensitive_action" for rf in verdict.red_flags)

def test_contact_redirect_red_flags(detector):
    # Message with LINE ID
    text = "想要了解更多請加我的LINE ID: scam123"
    verdict = detector.analyze(text)
    
    assert any(rf.key == "contact_redirect" for rf in verdict.red_flags)

def test_red_flag_boost(detector):
    # A message that might be borderline suspicious, but becomes scam with red flags
    # "Your account has an issue" (ambiguous) + "click this .top link immediately"
    text = "您的帳戶出現問題，請立即點擊連結修復: http://fix-account.top"
    verdict = detector.analyze(text)
    
    print(f"\nText: {text}")
    print(f"Scam Prob: {verdict.confidence:.4f}")
    print(f"Red Flags: {[rf.key for rf in verdict.red_flags]}")
    
    assert verdict.status == "scam"
    assert len(verdict.red_flags) >= 2
