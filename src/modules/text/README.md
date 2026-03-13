# Text Scam Detection Module

This module provides a sophisticated, multimodal approach to detecting fraudulent and scam-related text. It combines semantic AI, real-world knowledge retrieval (RAG), heuristic "red flag" analysis, and conversation context to provide high-accuracy verdicts and human-readable explanations.

## 🚀 Key Features

- **8 Expanded Scam Archetypes**: Targeted detection for Investment, Romance, Government Impersonation, Parcel Fraud, Guess Who I Am, ATM/Deduction Error, Job Scams, and Phishing.
- **Hybrid Scoring Pipeline**: Fuses semantic intent, weighted RAG evidence, and heuristic red-flag boosts.
- **Multi-Turn Context Awareness**: Analyzes conversation history to catch scams that develop over several messages (e.g., "Pig-butchering").
- **Weighted RAG Retriever**: FAISS-backed retrieval over a 280+ chunk corpus of official 165 anti-fraud warnings and scam examples.
- **Explainable AI (XAI)**: Integrated local LLM (`Qwen2.5-0.5B`) to generate natural language warnings in Traditional Chinese.

---

## 🏗 Architecture

The `TextDetector` orchestrates five distinct branches to reach a final verdict:

### 1. Intent Branch (`intent_embedder.py`)
Uses `paraphrase-multilingual-MiniLM-L12-v2` to compute the cosine similarity between the input text and 8 pre-defined scam archetypes. 
- **Archetypes defined in:** `scam_patterns.py`
- **Output:** `intent_score` (Weight: 40%)

### 2. RAG Branch (`rag_retriever.py`)
Performs a vector search against the 165 Anti-Fraud Knowledge Base.
- **Weighted Scoring**:
    - `scam_example`: +1.0
    - `official_warning`: +0.4 (weighted lower as they describe patterns rather than intent)
    - `safe_example`: -1.0
- **Output:** `rag_score` (Weight: 60%)

### 3. Heuristic Branch (`red_flags.py`)
Scans for "Red Flags" that AI embeddings might miss:
- **Suspicious URLs**: Detects `.top`, `.xyz`, `.vip` TLDs and shorteners like `bit.ly` or `reurl.cc`.
- **Urgency/Threats**: Keywords like "Immediately", "Otherwise", "Expired".
- **Sensitive Actions**: Terms like "Wire Transfer", "ATM", "Verification Code".
- **Output:** `heuristic_boost` (Added directly to the fused score)

### 4. Context Branch (`text_detector.py`)
Analyzes the current message in the context of the last 3 history messages.
- **Logic**: `max(current_message_score, concatenated_history_score)`
- **Goal**: Catches scams where the final prompt is innocent but the context is dangerous.

### 5. Explanation Branch (`explainer.py`)
Uses a local LLM to summarize the findings.
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Role**: Translates technical scores and RAG chunks into a helpful warning for the end-user.

---

## 📊 Detection Statuses

- **SCAM** (score ≥ 0.60): High-confidence fraud detected.
- **SUSPICIOUS** (0.30 < score < 0.60): Contains some red flags or low-similarity matches; proceed with caution.
- **SAFE** (score ≤ 0.30): No significant scam markers found.
- **OFFICIAL**: Specialized handling for exact matches to official government warnings (prevents false positives on educational text).

---

## 🛠 Usage

### Loading the Detector
```python
from modules.text.text_detector import TextDetector

detector = TextDetector()
detector.load()  # Loads SBERT, FAISS Index, and prepares LLM
```

### Analyzing Text
```python
text = "您的帳戶出現異常，請立即點擊連結修復: https://bank-verify.top"
history = ["您好，我是銀行客服", "剛才偵測到您的帳號在異地登入"]

verdict = detector.analyze(text, history=history)

print(f"Status: {verdict.status}")
print(f"AI Warning: {verdict.llm_explanation}")
```

---

## 📂 Data & Training

- **Corpus Ingestion**: `training/fetch_gov_data.py` downloads and parses the latest CSV from the Taiwan 165 Anti-Fraud hotline.
- **Index Building**: `training/build_rag_index.py` encodes the JSONL corpus into a FAISS vector index.
- **Corpus Location**: `src/data/scam_corpus/165_chunks.jsonl`

---

## ⚙️ Configuration
Thresholds and model names are centralized in `src/config.py`:
- `TEXT_INTENT_WEIGHT = 0.40`
- `TEXT_RAG_WEIGHT = 0.60`
- `EXPLAINER_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"`
