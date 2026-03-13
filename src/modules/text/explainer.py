"""
ScamExplainer — summarizes RAG evidence and red flags using a local LLM.
Translates technical scores into human-readable warnings.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import EXPLAINER_MODEL_NAME, EXPLAINER_DEVICE, MAX_NEW_TOKENS
from modules.text.rag_retriever import Chunk
from modules.text.red_flags import RedFlag

logger = logging.getLogger(__name__)

class ScamExplainer:
    """
    Local LLM-based explainer for scam detection results.
    """

    def __init__(
        self, 
        model_name: str = EXPLAINER_MODEL_NAME, 
        device: str = EXPLAINER_DEVICE
    ) -> None:
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        """Load the model and tokenizer."""
        if self._model is not None:
            return
        
        logger.info("Loading explainer LLM: %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Use bfloat16 if available for efficiency
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device != "cuda":
            self._model.to(self.device)
            
        logger.info("Explainer LLM ready.")

    def generate_explanation(
        self,
        text: str,
        status: str,
        archetype_zh: str,
        evidence: List[Chunk],
        red_flags: List[RedFlag],
        history: Optional[List[str]] = None
    ) -> str:
        """
        Generate a concise warning summary based on the evidence.
        """
        if self._model is None:
            return "Explainable AI (LLM) not loaded."

        # Filter evidence to unique sources or key snippets
        evidence_snippets = []
        seen = set()
        for c in evidence:
            if c.text not in seen:
                evidence_snippets.append(c.text)
                seen.add(c.text)
        
        evidence_text = "\n".join([f"- {s}" for s in evidence_snippets[:3]])
        red_flags_text = ", ".join([rf.description for rf in red_flags])
        history_text = "\n".join(history) if history else "None"

        prompt = f"""You are an anti-fraud expert. Summarize why the following message is considered {status}.
Message: {text}
Context History: {history_text}
Detected Type: {archetype_zh}
Red Flags: {red_flags_text}
Similar Scam Examples from 165 database:
{evidence_text}

Task: Provide a concise (under 80 words) warning in Traditional Chinese. 
Explain the scam logic and why the user should be careful. 
Do not mention 'archetypes' or 'technical scores'. 
Focus on the threat and action to take (e.g., 'Do not click', 'Do not transfer').
Warning:"""

        messages = [
            {"role": "system", "content": "你是一位反詐騙專家，能以專業且易懂的語氣向使用者解釋詐騙風險。"},
            {"role": "user", "content": prompt}
        ]

        text_input = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self._tokenizer([text_input], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Remove input tokens from output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()
