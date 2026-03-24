"""speech_analyzer.py
Audio extraction + speech-to-text + scam content analysis for video files.

Pipeline
--------
1. PyAV  — extract audio PCM from video (no system ffmpeg required)
2. faster-whisper — offline speech-to-text (small model, Chinese primary)
3. TextDetector   — scam intent scoring on the transcript
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from modules.text.text_detector import TextDetector
    from modules.text.rag_retriever import Chunk

SAMPLE_RATE = 16_000   # Whisper requires 16 kHz mono
_MIN_AUDIO_SEC = 1.5   # skip analysis if audio shorter than this
_MODEL_SIZE = "small"  # change to "medium" for higher accuracy


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class SpeechAnalysisResult:
    transcript: str
    language: str
    scam_status: str          # "scam" | "suspicious" | "safe" | "no_audio"
    scam_confidence: float    # 0.0 – 1.0
    scam_keywords: List[str] = field(default_factory=list)
    rag_evidence: list = field(default_factory=list)   # List[Chunk]
    all_similarities: dict = field(default_factory=dict)  # archetype_key → float
    processing_time_ms: float = 0.0
    explanation: str = ""
    closest_archetype: str = ""
    closest_archetype_zh: str = ""


# ---------------------------------------------------------------------------
# Audio extraction via PyAV (no system ffmpeg needed)
# ---------------------------------------------------------------------------
def _extract_audio_pcm(video_path: str) -> Optional[np.ndarray]:
    """
    Return float32 mono 16 kHz PCM array, or None if no audio stream found.
    """
    try:
        import av  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "PyAV not installed. Run: pip install av"
        )

    try:
        container = av.open(video_path)
    except Exception as exc:
        raise RuntimeError(f"Cannot open video for audio extraction: {exc}") from exc

    audio_streams = [s for s in container.streams if s.type == "audio"]
    if not audio_streams:
        container.close()
        return None

    resampler = av.audio.resampler.AudioResampler(
        format="fltp",   # float32 planar
        layout="mono",
        rate=SAMPLE_RATE,
    )

    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(audio_streams[0]):
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray()   # shape (1, N) float32
                chunks.append(arr[0])
    except Exception:
        pass  # partial audio still usable
    finally:
        container.close()

    if not chunks:
        return None

    audio = np.concatenate(chunks).astype(np.float32)

    # Normalise peak to ±1
    peak = np.abs(audio).max()
    if peak > 1e-6:
        audio /= peak

    return audio


# ---------------------------------------------------------------------------
# SpeechAnalyzer
# ---------------------------------------------------------------------------
class SpeechAnalyzer:
    """
    Lazy-loads faster-whisper model on first use to avoid slowing startup.
    Pass the shared TextDetector instance at construction time.
    """

    _whisper_model = None   # class-level singleton

    def __init__(self, text_detector: "TextDetector | None" = None) -> None:
        self._text_det = text_detector

    # ------------------------------------------------------------------
    def _ensure_whisper(self) -> None:
        if SpeechAnalyzer._whisper_model is not None:
            return
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
        print(f"[SpeechAnalyzer] Loading Whisper '{_MODEL_SIZE}' model "
              "(first use — may take ~30 s to download)…")
        SpeechAnalyzer._whisper_model = WhisperModel(
            _MODEL_SIZE,
            device="cpu",
            compute_type="int8",   # quantised: fast on CPU
        )
        print("[SpeechAnalyzer] Whisper ready.")

    # ------------------------------------------------------------------
    def analyze(self, video_path: str) -> SpeechAnalysisResult:
        t0 = time.time()

        # ── 1. Extract audio ──────────────────────────────────────────
        try:
            audio = _extract_audio_pcm(video_path)
        except RuntimeError as exc:
            return SpeechAnalysisResult(
                transcript="", language="unknown",
                scam_status="no_audio", scam_confidence=0.5,
                processing_time_ms=(time.time() - t0) * 1000,
                explanation=f"音訊提取失敗：{exc}",
            )

        if audio is None:
            return SpeechAnalysisResult(
                transcript="", language="unknown",
                scam_status="no_audio", scam_confidence=0.5,
                processing_time_ms=(time.time() - t0) * 1000,
                explanation="影片無音訊軌道。",
            )

        if len(audio) < SAMPLE_RATE * _MIN_AUDIO_SEC:
            return SpeechAnalysisResult(
                transcript="", language="unknown",
                scam_status="no_audio", scam_confidence=0.5,
                processing_time_ms=(time.time() - t0) * 1000,
                explanation=f"音訊過短（< {_MIN_AUDIO_SEC:.0f} 秒），跳過分析。",
            )

        # ── 2. Speech-to-text ─────────────────────────────────────────
        try:
            self._ensure_whisper()
        except RuntimeError as exc:
            return SpeechAnalysisResult(
                transcript="", language="unknown",
                scam_status="no_audio", scam_confidence=0.5,
                processing_time_ms=(time.time() - t0) * 1000,
                explanation=str(exc),
            )

        try:
            segments, info = SpeechAnalyzer._whisper_model.transcribe(
                audio,
                language="zh",          # Mandarin/Traditional Chinese primary
                beam_size=5,
                vad_filter=True,        # skip silent sections
                vad_parameters={"min_silence_duration_ms": 500},
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            language = getattr(info, "language", "zh")
        except Exception as exc:
            return SpeechAnalysisResult(
                transcript="", language="unknown",
                scam_status="no_audio", scam_confidence=0.5,
                processing_time_ms=(time.time() - t0) * 1000,
                explanation=f"語音辨識失敗：{exc}",
            )

        if not transcript:
            return SpeechAnalysisResult(
                transcript="", language=language,
                scam_status="no_audio", scam_confidence=0.5,
                processing_time_ms=(time.time() - t0) * 1000,
                explanation="音訊中未偵測到語音。",
            )

        # ── 3. Scam text analysis ──────────────────────────────────────
        if self._text_det is not None:
            try:
                v = self._text_det.analyze(transcript)
                scam_status      = v.status
                scam_confidence  = v.confidence
                rag_evidence     = v.rag_evidence or []
                all_sims         = v.all_similarities or {}
                keywords         = [c.text[:50].strip() + "…" for c in rag_evidence][:3]
                explanation      = v.explanation
                closest_arch     = v.closest_archetype
                closest_arch_zh  = v.closest_archetype_zh
            except Exception as exc:
                scam_status     = "suspicious"
                scam_confidence = 0.5
                rag_evidence    = []
                all_sims        = {}
                keywords        = []
                explanation     = f"文字分析出錯：{exc}"
                closest_arch    = ""
                closest_arch_zh = ""
        else:
            scam_status     = "suspicious"
            scam_confidence = 0.5
            rag_evidence    = []
            all_sims        = {}
            keywords        = []
            explanation     = "（TextDetector 未載入，僅提供轉錄文字）"
            closest_arch    = ""
            closest_arch_zh = ""

        return SpeechAnalysisResult(
            transcript=transcript,
            language=language,
            scam_status=scam_status,
            scam_confidence=scam_confidence,
            scam_keywords=keywords,
            rag_evidence=rag_evidence,
            all_similarities=all_sims,
            processing_time_ms=(time.time() - t0) * 1000,
            explanation=explanation,
            closest_archetype=closest_arch,
            closest_archetype_zh=closest_arch_zh,
        )
