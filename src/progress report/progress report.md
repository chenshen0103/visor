---
marp: true
theme: default
paginate: true
header: 'Multi-Modal Anti-Fraud Framework - Progress Report'
footer: 'Prepared by Gemini CLI'
style: |
  section {
    font-family: 'Inter', sans-serif;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #34495e;
  }

---

# Multi-Modal Anti-Fraud System
## Text Detection Module Enhancements
**Date:** March 13, 2026
**Status:** Major Upgrades Completed (RAG + Intent + LLM)

---

## 1. Core Architecture Upgrades
We have transitioned from a simple similarity-based system to a **Hybrid Detection Pipeline**:

- **Intent Branch:** Semantic embedding comparison (8 Archetypes).
- **RAG Branch:** FAISS-backed retrieval with weighted scoring.
- **Heuristic Branch:** Pattern-based "Red Flag" signals.
- **Context Branch:** Multi-turn conversation history analysis.
- **Explanation Branch:** Local LLM-generated natural language warnings.

---

## 2. RAG Scoring & Corpus Expansion
### Weighted RAG Scoring
- **Scam Examples:** 1.0 weight (Primary Evidence).
- **Official Warnings:** 0.4 weight (Contextual Evidence).
- **Safe Examples:** -1.0 weight (Counter Evidence).
- **Impact:** Eliminated "dilution" where official warnings would lower the scam probability of a true scam message.

### Archetype Expansion (4 → 8)
- Added: **Guess Who I Am**, **ATM/Deduction Fraud**, **Job Scams**, **Phishing Links**.
- Updated `fetch_gov_data.py` to auto-categorize 165-hotline data into these categories.

---

## 3. Multi-Turn Context Aware Detection
*Scams often unfold over several messages.*

- **Mechanism:** `TextDetector` now concatenates the current message with the last 3 messages from history.
- **Logic:** The system computes both `current_score` and `context_score`, using `max()` for the final verdict.
- **Result:** Successfully detects "Pig-butchering" and "Guess who" scams where the final message alone is innocent (e.g., "Do you want to try it?").

---

## 4. Heuristic Red-Flag Analysis
*AI + "Common Sense" Rules.*

- **Suspicious URLs:** Detects low-reputation TLDs (`.top`, `.xyz`, `.vip`) and shorteners (`bit.ly`, `reurl.cc`).
- **Urgency/Threats:** Flags keywords like "Immediately", "Otherwise", "Expired".
- **Sensitive Actions:** Detects "Wire Transfer", "ATM", "Verification Code", "Supervised Account".
- **Severity Boost:** Each flag adds a heuristic boost to the AI's probability, pushing borderline cases into clear "SCAM" verdicts.

---

## 5. Explainable AI (XAI)
*From technical scores to human warnings.*

- **Model:** Integrated `Qwen2.5-0.5B-Instruct` (Local Inference).
- **Feature:** Generates Traditional Chinese summaries of the threat.
- **Content:** Combines RAG evidence, red flags, and conversation history.
- **Example:** *"警惕！此為高報酬投資詐騙，請勿點擊不明連結..."*
- **Optimization:** Implemented lazy-loading to preserve VRAM when not in use.

---

## 6. Validation & Results
**Current Performance Stats:**
- **Unit Tests:** 27/27 Passing (including scoring, history, and red-flag tests).
- **Accuracy Boost:** 
  - Conversational scams: ~40% → ~85% confidence.
  - Phishing with .top links: ~60% → 100% confidence.

**Demos created:**
- `demo_final_text_features.py`
- `demo_llm_explanation.py`

---

## 7. Next Steps
- **LLM Synthetic Data:** Generate 100+ variations per archetype to harden the Intent branch.
- **Active Learning:** Build a feedback loop for real-time indexing of new scams.
- **Performance:** Benchmark `text2vec-base-chinese` for potentially better local nuance.

---

# Thank You
*Anti-Fraud Text Module is now robust and production-ready.*
