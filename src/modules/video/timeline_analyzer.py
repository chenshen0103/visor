"""timeline_analyzer.py
Sliding-window, multi-person rPPG timeline analysis.

For each sliding window and each tracked person, runs POS rPPG and
produces a real_prob score.  Returns List[SegmentResult] that can be
visualised as a per-person, per-segment heatmap in the demo UI.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

# ── tuning constants ──────────────────────────────────────────────────────────
_DETECT_EVERY_N = 5        # run Haar face-detector every N frames (speed)
_MAX_PERSONS    = 4        # cap tracked identities
_MIN_FACE_PX    = 40       # minimum face bbox side (px)
_EMA_ALPHA      = 0.30     # centroid EMA smoothing
_MAX_GONE_F     = 25       # frames before dropping a lost track
_MAX_DIST_PX    = 150      # max centroid distance to match a track


# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class SegmentResult:
    t_start:      float
    t_end:        float
    person_id:    int
    real_prob:    float          # 0 (fake) … 1 (real)
    status:       str            # "real" | "fake" | "uncertain" | "no_face"
    hr_bpm:       float
    pearson_sync: float
    coverage:     float          # fraction of segment frames where person visible


# ── centroid tracker ──────────────────────────────────────────────────────────
class _PersonTracker:
    """Greedy centroid-based multi-object tracker with EMA smoothing."""

    def __init__(self) -> None:
        self._next_id: int = 0
        self._cents:  Dict[int, np.ndarray] = {}
        self._bboxes: Dict[int, Tuple]      = {}
        self._gone:   Dict[int, int]        = {}

    def update(self, bboxes: List[Tuple]) -> Dict[int, Tuple]:
        """bboxes: list of (x, y, w, h).  Returns {person_id: bbox}."""

        # Age unmatched tracks; return last known positions if no detections
        if not bboxes:
            for pid in list(self._gone):
                self._gone[pid] += 1
                if self._gone[pid] > _MAX_GONE_F:
                    self._cents.pop(pid, None)
                    self._bboxes.pop(pid, None)
                    self._gone.pop(pid, None)
            return dict(self._bboxes)

        new_cents = np.array(
            [(x + w / 2.0, y + h / 2.0) for x, y, w, h in bboxes],
            dtype=np.float32,
        )

        # First detections ever → register all
        if not self._cents:
            for i, (c, bb) in enumerate(zip(new_cents, bboxes)):
                if i >= _MAX_PERSONS:
                    break
                self._cents[self._next_id]  = c
                self._bboxes[self._next_id] = bb
                self._gone[self._next_id]   = 0
                self._next_id += 1
            return dict(self._bboxes)

        existing_ids   = list(self._cents.keys())
        existing_cents = np.array([self._cents[p] for p in existing_ids])

        # Distance matrix (n_existing × n_new)
        dist = np.linalg.norm(
            existing_cents[:, None] - new_cents[None, :], axis=2
        )

        matched_e: set = set()
        matched_n: set = set()

        # Greedy match by ascending distance
        flat_order = np.argsort(dist.ravel())
        for flat_idx in flat_order:
            ei, ni = divmod(int(flat_idx), len(bboxes))
            if ei in matched_e or ni in matched_n:
                continue
            if dist[ei, ni] > _MAX_DIST_PX:
                break
            pid = existing_ids[ei]
            # EMA centroid update
            self._cents[pid]  = (1 - _EMA_ALPHA) * self._cents[pid] + _EMA_ALPHA * new_cents[ni]
            self._bboxes[pid] = bboxes[ni]
            self._gone[pid]   = 0
            matched_e.add(ei)
            matched_n.add(ni)

        # Age unmatched existing
        for ei, pid in enumerate(existing_ids):
            if ei not in matched_e:
                self._gone[pid] += 1
                if self._gone[pid] > _MAX_GONE_F:
                    self._cents.pop(pid, None)
                    self._bboxes.pop(pid, None)
                    self._gone.pop(pid, None)

        # Register new unmatched detections
        for ni in range(len(bboxes)):
            if ni not in matched_n and len(self._cents) < _MAX_PERSONS:
                self._cents[self._next_id]  = new_cents[ni]
                self._bboxes[self._next_id] = bboxes[ni]
                self._gone[self._next_id]   = 0
                self._next_id += 1

        return dict(self._bboxes)


# ── ROI helpers ───────────────────────────────────────────────────────────────
def _face_sub_bgr(
    frame: np.ndarray, bbox: Tuple
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (top_bgr, bottom_bgr) mean of two face sub-regions."""
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    mx = max(1, int(w * 0.15))

    # Top: forehead (5 %–40 % of face height)
    t_y1 = max(0, y + int(h * 0.05))
    t_y2 = max(0, y + int(h * 0.40))
    # Bottom: cheeks (55 %–90 %)
    b_y1 = max(0, y + int(h * 0.55))
    b_y2 = min(frame.shape[0], y + int(h * 0.90))
    x1   = max(0, x + mx)
    x2   = min(frame.shape[1], x + w - mx)

    top_roi = frame[t_y1:t_y2, x1:x2]
    bot_roi = frame[b_y1:b_y2, x1:x2]

    top = top_roi.mean(axis=(0, 1)).astype(np.float32) if top_roi.size >= 3 else None
    bot = bot_roi.mean(axis=(0, 1)).astype(np.float32) if bot_roi.size >= 3 else None
    return top, bot


# ── signal processing ─────────────────────────────────────────────────────────
def _pos_bvp(bgr_series: np.ndarray) -> np.ndarray:
    """POS rPPG (de Haan & Jeanne 2013).  bgr_series: (T, 3) BGR."""
    if len(bgr_series) < 6:
        return np.zeros(len(bgr_series))
    rgb = bgr_series[:, ::-1].astype(np.float64)   # BGR → RGB
    mu  = rgb.mean(axis=0, keepdims=True) + 1e-8
    C   = rgb / mu
    H1  = C[:, 1] - C[:, 2]
    H2  = -2.0 * C[:, 0] + C[:, 1] + C[:, 2]
    bvp = H1 + (H1.std() / (H2.std() + 1e-8)) * H2
    std = bvp.std()
    return ((bvp - bvp.mean()) / (std + 1e-8)).astype(np.float32)


def _bandpass(sig: np.ndarray, fps: float, lo: float = 0.7, hi: float = 3.5) -> np.ndarray:
    nyq = fps / 2.0
    hi  = min(hi, nyq * 0.95)
    if lo >= hi or len(sig) < 18:
        return sig
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 6:
        return 0.0
    if a.std() < 1e-7 or b.std() < 1e-7:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(np.nan_to_num(r))


def _score_segment(
    top_list: List[np.ndarray],
    bot_list: List[np.ndarray],
    fps: float,
) -> Tuple[float, float, float]:
    """Returns (real_prob, hr_bpm, pearson_sync)."""
    if len(top_list) < 30 or len(bot_list) < 30:
        return 0.5, 0.0, 0.0

    top = np.array(top_list)   # (T, 3) BGR
    bot = np.array(bot_list)

    top_bvp = _bandpass(_pos_bvp(top), fps)
    bot_bvp = _bandpass(_pos_bvp(bot), fps)

    pearson_r = _safe_pearson(top_bvp, bot_bvp)

    # HR from top-ROI PSD
    freqs = np.fft.rfftfreq(len(top_bvp), d=1.0 / fps)
    psd   = np.abs(np.fft.rfft(top_bvp)) ** 2
    mask  = (freqs >= 0.7) & (freqs <= 3.5)
    if mask.sum() == 0:
        hr_bpm, snr = 0.0, 0.0
    else:
        hr_bpm  = float(freqs[mask][np.argmax(psd[mask])]) * 60.0
        in_band = float(psd[mask].sum())
        snr     = in_band / (float(psd.sum()) - in_band + 1e-8)

    # Map Pearson r → real_prob, nudge by SNR
    real_prob = (pearson_r + 1.0) / 2.0
    if snr > 0.35:
        real_prob = min(1.0, real_prob + 0.10)
    elif snr < 0.05:
        real_prob = max(0.0, real_prob - 0.15)

    return float(real_prob), hr_bpm, float(pearson_r)


# ── main entry point ──────────────────────────────────────────────────────────
def analyze_timeline(
    mp4_path:    str,
    segment_sec: float = 6.0,
    stride_sec:  float = 2.0,
) -> List[SegmentResult]:
    """
    Analyze a video in sliding windows, tracking multiple persons independently.

    Parameters
    ----------
    mp4_path    : path to MP4 file
    segment_sec : window length in seconds  (default 6 s)
    stride_sec  : stride between windows    (default 2 s)

    Returns
    -------
    List[SegmentResult] — one entry per (segment × tracked person).
    Empty list if no faces found.
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return []

    fps          = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seg_f        = max(1, int(segment_sec * fps))
    stride_f     = max(1, int(stride_sec  * fps))

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    tracker      = _PersonTracker()
    last_tracked: Dict[int, Tuple] = {}

    # {pid: {"top": {frame_idx: bgr}, "bot": {frame_idx: bgr}}}
    person_data: Dict[int, Dict[str, Dict[int, np.ndarray]]] = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── face detection (every N frames) ───────────────────────────────
        if frame_idx % _DETECT_EVERY_N == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                cv2.equalizeHist(gray),
                scaleFactor=1.1, minNeighbors=4,
                minSize=(_MIN_FACE_PX, _MIN_FACE_PX),
            )
            bboxes       = [tuple(int(v) for v in f) for f in faces] if len(faces) else []
            last_tracked = tracker.update(bboxes)

        # ── extract RGB per person using last tracked bboxes ──────────────
        for pid, bbox in last_tracked.items():
            top_bgr, bot_bgr = _face_sub_bgr(frame, bbox)
            if top_bgr is None or bot_bgr is None:
                continue
            if pid not in person_data:
                person_data[pid] = {"top": {}, "bot": {}}
            person_data[pid]["top"][frame_idx] = top_bgr
            person_data[pid]["bot"][frame_idx] = bot_bgr

        frame_idx += 1

    cap.release()

    if not person_data:
        return []

    # ── sliding window scoring ────────────────────────────────────────────
    results: List[SegmentResult] = []
    seg_starts = list(range(0, max(1, total_frames - seg_f + 1), stride_f))

    for seg_start in seg_starts:
        seg_end = seg_start + seg_f
        t_start = seg_start / fps
        t_end   = min(seg_end / fps, total_frames / fps)
        f_range = range(seg_start, min(seg_end, total_frames))

        for pid, data in person_data.items():
            top_list = [data["top"][fi] for fi in f_range if fi in data["top"]]
            bot_list = [data["bot"][fi] for fi in f_range if fi in data["bot"]]
            coverage = len(top_list) / max(1, seg_f)

            if coverage < 0.25 or len(top_list) < 30:
                results.append(SegmentResult(
                    t_start=t_start, t_end=t_end, person_id=pid,
                    real_prob=float("nan"), status="no_face",
                    hr_bpm=0.0, pearson_sync=0.0, coverage=coverage,
                ))
                continue

            real_prob, hr_bpm, pearson_sync = _score_segment(top_list, bot_list, fps)

            if real_prob >= 0.60:
                status = "real"
            elif real_prob <= 0.35:
                status = "fake"
            else:
                status = "uncertain"

            results.append(SegmentResult(
                t_start=t_start, t_end=t_end, person_id=pid,
                real_prob=real_prob, status=status,
                hr_bpm=hr_bpm, pearson_sync=pearson_sync,
                coverage=coverage,
            ))

    return results
