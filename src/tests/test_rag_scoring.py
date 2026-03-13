
import pytest
import numpy as np
from modules.text.text_detector import TextDetector
from modules.text.rag_retriever import RAGRetriever

@pytest.fixture(scope="module")
def detector():
    d = TextDetector()
    d.load()
    return d

def test_rag_scoring_dilution(detector):
    # This message is a known scam pattern described in many official warnings
    text = "請配合轉帳至監管帳戶以示清白"
    verdict = detector.analyze(text)
    
    print(f"\nText: {text}")
    print(f"Status: {verdict.status}")
    print(f"Scam Prob: {verdict.confidence if verdict.is_scam else 1-verdict.confidence:.4f}")
    print(f"Intent Similarity: {verdict.intent_similarity:.4f}")
    print(f"RAG Scam Ratio: {verdict.rag_scam_ratio:.4f}")
    
    # Check labels of retrieved chunks
    labels = [c.label for c in verdict.rag_evidence]
    print(f"RAG Chunk Labels: {labels}")
    
def test_official_warning_not_flagged(detector):
    # This is an exact copy of an official warning in the corpus (c001)
    text = "刑事警察局提醒民眾：政府機關不會要求民眾轉帳至「監管帳戶」或「安全帳戶」，此為常見詐騙話術，請勿上當。"
    verdict = detector.analyze(text)
    
    print(f"\nText: {text}")
    print(f"Status: {verdict.status}")
    print(f"Scam Prob: {verdict.confidence if verdict.is_scam else 1-verdict.confidence:.4f}")
    print(f"RAG Scam Ratio: {verdict.rag_scam_ratio:.4f}")
    
    # It should be safe because it matches an official warning exactly
    assert verdict.status == "safe"
    assert verdict.is_scam is False
