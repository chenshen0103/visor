"""
demo.py — Gradio demo for the Multi-Modal Anti-Fraud Defense Framework.

Usage
-----
    python src/demo.py              # opens browser at http://localhost:7860
    python src/demo.py --port 7861  # custom port
    python src/demo.py --share      # public link via Gradio tunnel
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modules.text.text_detector import TextDetector
from modules.photo.photo_detector import PhotoDetector
from modules.video.video_detector import VideoDetector
from modules.video.timeline_analyzer import analyze_timeline, SegmentResult
from modules.video.speech_analyzer import SpeechAnalyzer

# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------
print("[DEMO] Loading models…")
_text_det = TextDetector()
_text_det.load()

_photo_det = PhotoDetector()

_video_det = VideoDetector()

# SpeechAnalyzer: Whisper model is lazy-loaded on first video analysis
_speech_analyzer = SpeechAnalyzer(text_detector=_text_det)
print("[DEMO] Models ready.")

# ---------------------------------------------------------------------------
# Helper: format result as Markdown
# ---------------------------------------------------------------------------
STATUS_EMOJI = {
    "scam":      "🚨",
    "safe":      "✅",
    "suspicious":"⚠️",
    "real":      "✅",
    "fake":      "🚨",
    "face_swap": "🎭",
    "uncertain": "⚠️",
}

# Thresholds for intent matching display
_INTENT_HIT   = 0.45   # ≥ this → 🚨 命中
_INTENT_WARN  = 0.30   # ≥ this → ⚠️ 輕度相關


def _sim_bar(sim: float, width: int = 10) -> str:
    """ASCII progress bar for similarity score, e.g. ███████░░░ 72%"""
    filled = round(sim * width)
    return "█" * filled + "░" * (width - filled) + f" {sim:.0%}"


def _format_intent_analysis(all_similarities: dict, title: str = "意圖比對") -> str:
    """
    Render a Markdown block showing how the text's intent compares against
    all scam archetypes.  Archetypes are sorted by similarity (high→low).
    Only archetypes above _INTENT_WARN are shown in detail; rest are listed
    as "未命中".
    """
    if not all_similarities:
        return ""
    from modules.text.scam_patterns import ARCHETYPE_BY_KEY  # noqa: PLC0415

    # Sort archetypes high → low similarity
    sorted_items = sorted(all_similarities.items(), key=lambda x: x[1], reverse=True)

    hit_lines   = []   # sim ≥ _INTENT_HIT
    warn_lines  = []   # _INTENT_WARN ≤ sim < _INTENT_HIT
    miss_names  = []   # sim < _INTENT_WARN

    for key, sim in sorted_items:
        arch = ARCHETYPE_BY_KEY.get(key)
        if arch is None:
            continue
        bar = _sim_bar(sim)
        if sim >= _INTENT_HIT:
            hit_lines.append(
                f"#### 🚨 {arch.name_zh}（{arch.name_en}）\n"
                f"`{bar}`\n\n"
                f"**詐騙手法說明：** {arch.description}\n\n"
                f"**典型話術範例：**\n"
                + "\n".join(f"> {ex}" for ex in arch.exemplars[:3])
            )
        elif sim >= _INTENT_WARN:
            warn_lines.append(
                f"#### ⚠️ {arch.name_zh}（輕度相關）\n"
                f"`{bar}`\n\n"
                f"{arch.description[:100]}…"
            )
        else:
            miss_names.append(f"~~{arch.name_zh}~~")

    parts = [f"### 🎯 {title}\n"]

    if hit_lines:
        parts.append("**命中以下詐騙意圖：**\n")
        parts.extend(hit_lines)
    else:
        parts.append("> ✅ 未命中任何高風險詐騙意圖。\n")

    if warn_lines:
        parts.append("\n**輕度相關（僅供參考）：**\n")
        parts.extend(warn_lines)

    if miss_names:
        parts.append(f"\n**未命中：** {' ／ '.join(miss_names)}")

    return "\n".join(parts)


_BADGE_STYLE = {
    "scam":      ("🚨", "SCAM",       "#7f1d1d", "#fca5a5"),
    "safe":      ("✅", "SAFE",       "#14532d", "#86efac"),
    "suspicious":("⚠️", "SUSPICIOUS", "#713f12", "#fde68a"),
    "real":      ("✅", "REAL",       "#14532d", "#86efac"),
    "fake":      ("🚨", "FAKE",       "#7f1d1d", "#fca5a5"),
    "face_swap": ("🎭", "FACE SWAP",  "#4c1d95", "#c4b5fd"),
    "uncertain": ("⚠️", "UNCERTAIN",  "#713f12", "#fde68a"),
}

def _badge(status: str, confidence: float | None = None) -> str:
    emoji, label, fg, bg = _BADGE_STYLE.get(status, ("❓", status.upper(), "#374151", "#e5e7eb"))
    conf_line = f"<div style='font-size:1em;margin-top:6px;opacity:0.85'>可信度 {confidence:.1%}</div>" if confidence is not None else ""
    return (
        f'<div style="background:{bg};color:{fg};border:2px solid {fg};border-radius:12px;'
        f'padding:20px 28px;text-align:center;margin-bottom:12px">'
        f'<div style="font-size:2.2em;font-weight:900;letter-spacing:2px">{emoji} {label}</div>'
        f'{conf_line}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Helper: convert any video format to a temp MP4 using OpenCV
# (avoids the need for system ffmpeg)
# ---------------------------------------------------------------------------
def _ensure_mp4(video_path: str) -> str:
    """
    If *video_path* is already .mp4, return it unchanged.
    Otherwise re-encode every frame into a temp MP4 using OpenCV VideoWriter
    (codec: mp4v).  Returns the path to the (possibly new) MP4 file.
    The caller is responsible for deleting the temp file when done.
    """
    if Path(video_path).suffix.lower() == ".mp4":
        return video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return tmp_path


# ---------------------------------------------------------------------------
# Helper: download a video URL to a temp MP4 using yt-dlp
# ---------------------------------------------------------------------------
def _download_url(url: str, cookies_path: str | None = None) -> str:
    """
    Download *url* (YouTube / direct video / social media) to a temp MP4
    using yt-dlp (no ffmpeg required when the source is already mp4/webm).
    Returns path to the downloaded file.
    Raises RuntimeError on failure.

    cookies_path: optional path to a Netscape-format cookies.txt file.
    """
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    tmp_dir = tempfile.mkdtemp()
    out_template = str(Path(tmp_dir) / "video.%(ext)s")

    base_opts = {
        "outtmpl": out_template,
        # Select only pre-muxed single-file formats (no ffmpeg merge needed).
        "format": (
            "best[ext=mp4][vcodec!='none'][acodec!='none']"
            "/best[ext=webm][vcodec!='none'][acodec!='none']"
            "/best[vcodec!='none'][acodec!='none']"
            "/best"
        ),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    _FB_DOMAINS = ("facebook.com", "fb.com", "instagram.com", "fb.watch")
    needs_cookies = any(d in url for d in _FB_DOMAINS)

    info = None
    last_err: Exception | None = None

    # 0) User-supplied cookies.txt (most reliable for FB/IG)
    if cookies_path and Path(cookies_path).exists():
        cookie_size = Path(cookies_path).stat().st_size
        print(f"[DEMO] Using cookies file: {cookies_path} ({cookie_size} bytes)")
        if cookie_size < 500:
            print("[DEMO] WARNING: cookies.txt looks too small — may be incomplete export")
        try:
            with yt_dlp.YoutubeDL({**base_opts, "cookiefile": cookies_path,
                                    "quiet": False}) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
        except Exception as e:
            print(f"[DEMO] cookies.txt attempt failed: {type(e).__name__}: {e}")
            last_err = e

    # 1) Try browsers — Firefox first (no DPAPI issue on Windows),
    #    then Edge, then Chrome (Chrome 127+ may fail due to App-Bound Encryption).
    if info is None and needs_cookies:
        for browser in ["firefox", "edge", "chrome", "brave"]:
            print(f"[DEMO] Trying cookies from browser: {browser}")
            try:
                with yt_dlp.YoutubeDL(
                    {**base_opts, "cookiesfrombrowser": (browser, None, None, None),
                     "quiet": False}
                ) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)
                print(f"[DEMO] Success with browser cookies: {browser}")
                break
            except Exception as e:
                print(f"[DEMO] {browser} failed: {type(e).__name__}: {e}")
                last_err = e

    # 2) Fallback: no cookies (works for public / non-FB platforms)
    if info is None:
        try:
            with yt_dlp.YoutubeDL(base_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
        except Exception as e:
            last_err = e

    if info is None:
        err_str = f"{type(last_err).__name__}: {last_err}" if last_err else "未知錯誤"
        print(f"[DEMO] All download attempts failed. Last error: {err_str}")
        hint = ""
        if needs_cookies:
            hint = (
                "\n\n💡 **Facebook / Instagram 下載失敗，請擇一解法：**\n\n"
                "**① 手動下載影片後上傳（最簡單）**\n"
                "   用 [SnapSave](https://snapsave.app/zh-tw) 或 [SaveFrom](https://zh.savefrom.net/) 下載影片，"
                "再切到「上傳檔案」tab 分析。\n\n"
                "**② 重新匯出 cookies.txt（需 > 2KB）**\n"
                "   瀏覽器安裝 Get cookies.txt LOCALLY，到 facebook.com **首頁**後匯出（非分享連結頁面）。\n\n"
                "**③ 用 Firefox 登入 FB 後重試（不需上傳 cookies）**\n"
                "   Firefox 不受 DPAPI 限制，yt-dlp 可直接讀取。"
            )
        raise RuntimeError(f"下載失敗：{err_str}{hint}")

    # yt-dlp may have chosen a different extension
    downloaded = Path(filename)
    if not downloaded.exists():
        # Fallback: pick the first file in tmp_dir
        files = list(Path(tmp_dir).iterdir())
        if not files:
            raise RuntimeError("yt-dlp download produced no output file.")
        downloaded = files[0]

    # Convert to mp4 if needed (OpenCV fallback, no ffmpeg)
    mp4_path = _ensure_mp4(str(downloaded))
    return mp4_path


# ---------------------------------------------------------------------------
# Tab 1 — Text Analysis
# ---------------------------------------------------------------------------
def analyze_text(text: str) -> tuple[str, str, str]:
    if not text or not text.strip():
        return "（未輸入文字）", "", ""

    v = _text_det.analyze(text.strip())
    badge = _badge(v.status, v.confidence)

    summary_md = f"""
| 指標 | 數值 |
|------|------|
| **最接近詐騙類型** | {v.closest_archetype_zh} ({v.closest_archetype}) |
| **意圖相似度** | {v.intent_similarity:.3f} |
| **RAG 詐騙比例** | {v.rag_scam_ratio:.1%} |
| **處理時間** | {v.processing_time_ms:.0f} ms |

**說明：** {v.explanation}
"""

    rag_md = _format_intent_analysis(
        v.all_similarities, title="詐騙意圖比對結果"
    ) if v.all_similarities else ""
    return badge, summary_md.strip(), rag_md.strip()


# ---------------------------------------------------------------------------
# Tab 2 — Photo Analysis
# ---------------------------------------------------------------------------
def analyze_photo(image: np.ndarray | None) -> tuple[str, str]:
    if image is None:
        return "（未上傳圖片）", ""

    # Gradio returns RGB numpy array; convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    v = _photo_det.analyze_array(img_bgr)
    badge = _badge(v.status, v.confidence)

    summary_md = f"""
| 指標 | 數值 |
|------|------|
| **鏡頭幾何分數** | {v.geometry_score:.3f} |
| **PRNU 感測器分數** | {v.prnu_score:.3f} |
| **週期性插值偽影** | {"是 ⚠️" if v.has_periodic_artifacts else "否 ✅"} |
| **處理時間** | {v.processing_time_ms:.0f} ms |

**說明：** {v.explanation}
"""
    return badge, summary_md.strip()


# ---------------------------------------------------------------------------
# Timeline heatmap builder
# ---------------------------------------------------------------------------
def _make_timeline_fig(
    results: list[SegmentResult],
    min_active_segs: int = 3,
) -> plt.Figure | None:
    """
    Build a matplotlib heatmap: rows = persons, columns = time segments.
    Color:  red (fake=0) → yellow (uncertain=0.5) → green (real=1).
    Gray cells = person not visible in that segment.
    Only persons with ≥ min_active_segs non-no_face segments are shown.
    Returns None if results is empty.
    """
    if not results:
        return None

    # Filter: only persons that appear in enough segments
    all_person_ids = sorted({r.person_id for r in results})
    person_ids = [
        pid for pid in all_person_ids
        if sum(1 for r in results if r.person_id == pid and r.status != "no_face")
        >= min_active_segs
    ]
    if not person_ids:
        person_ids = all_person_ids  # fallback: show all if none pass filter

    # Renumber persons by order of first appearance for cleaner labels
    first_t = {
        pid: min(r.t_start for r in results if r.person_id == pid)
        for pid in person_ids
    }
    person_ids = sorted(person_ids, key=lambda p: first_t[p])
    t_starts   = sorted({r.t_start   for r in results})
    n_p, n_t   = len(person_ids), len(t_starts)

    pid_idx = {p: i for i, p in enumerate(person_ids)}
    t_idx   = {t: i for i, t in enumerate(t_starts)}

    # Build probability matrix (NaN = no face)
    data = np.full((n_p, n_t), np.nan)
    for r in results:
        # Skip persons not in the display set (filtered out due to <min_active_segs)
        if r.person_id not in pid_idx or r.t_start not in t_idx:
            continue
        if r.status != "no_face":
            data[pid_idx[r.person_id], t_idx[r.t_start]] = r.real_prob

    # ── per-person overall verdict (majority over non-uncertain segs) ──────
    verdicts: list[str] = []
    for pid in person_ids:
        segs = [r for r in results if r.person_id == pid and r.status not in ("no_face", "uncertain")]
        if not segs:
            verdicts.append("⚠️ 不確定")
            continue
        real_frac = sum(1 for r in segs if r.status == "real") / len(segs)
        if real_frac >= 0.5:
            verdicts.append("✅ 真人")
        else:
            verdicts.append("🚨 疑似AI")

    # ── figure ────────────────────────────────────────────────────────────
    fig_w = max(8, min(20, n_t * 0.55 + 2))
    fig_h = max(2.5, n_p * 1.4 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Custom colormap: red → yellow → green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rppg", [(0.85, 0.15, 0.15), (0.95, 0.80, 0.10), (0.15, 0.75, 0.25)], N=256
    )
    cmap.set_bad(color="#3a3a5c")   # gray for NaN (no face)

    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
                   interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0.0, 0.35, 0.60, 1.0])
    cbar.set_ticklabels(["偽造", "不確定", "真實", ""])
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    # Annotate cells with real_prob value
    for pi in range(n_p):
        for ti in range(n_t):
            v = data[pi, ti]
            if np.isfinite(v):
                ax.text(ti, pi, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="white" if v < 0.8 else "#111",
                        fontweight="bold")

    # Y-axis: person labels with verdict
    y_labels = [
        f"Person {pid + 1}  {verdict}"
        for pid, verdict in zip(person_ids, verdicts)
    ]
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(y_labels, color="white", fontsize=10)

    # X-axis: every ~10 s
    stride = t_starts[1] - t_starts[0] if len(t_starts) > 1 else 2.0
    step   = max(1, round(10.0 / stride))
    x_ticks  = list(range(0, n_t, step))
    x_labels = [f"{t_starts[i]:.0f}s" for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, color="white", fontsize=8)

    ax.tick_params(axis="both", colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555577")

    ax.set_xlabel("時間", color="white", fontsize=9)
    ax.set_title(
        "段落級 rPPG 偵測（綠=偵測到心律・紅=未偵測・灰=無人臉）",
        color="white", fontsize=10, pad=8,
    )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 3 — Video Analysis (file upload + URL)
# ---------------------------------------------------------------------------
def _run_video_analysis(video_path: str) -> tuple[str, str, plt.Figure | None]:
    """Core analysis logic — accepts a local file path (mp4 or avi)."""
    tmp_created: str | None = None
    try:
        print(f"[DEMO] _run_video_analysis: {video_path}")
        mp4_path = _ensure_mp4(video_path)
        if mp4_path != video_path:
            tmp_created = mp4_path   # remember to clean up
        print(f"[DEMO] mp4_path ready: {mp4_path}")

        # ── overall verdict ────────────────────────────────────────────────
        print("[DEMO] calling video_det.analyze()...")
        v = _video_det.analyze(mp4_path)
        print(f"[DEMO] analyze() done: status={v.status} confidence={v.confidence:.2f}")

        _VIDEO_LABELS = {
            "real":      "偵測到生理信號（真實人臉）",
            "fake":      "未偵測到生理信號（疑似 AI 生成）",
            "face_swap": "疑似換臉（臉部與頸部信號不一致）",
            "uncertain": "信號不足，無法判定",
        }
        video_label = _VIDEO_LABELS.get(v.status, v.status.upper())
        badge = _badge(v.status, v.confidence) + f"\n\n**{video_label}**"

        disclaimer = ""
        if v.status == "real":
            disclaimer = (
                "\n> ⚠️ **注意：** 本分析僅偵測影片中是否存在真實生理信號（心率/脈搏），"
                "**不代表影片內容安全**。詐騙影片可能使用真人出鏡。"
                "建議搭配「文字分析」功能檢測語音/字幕內容是否涉及詐騙話術。\n"
            )

        summary_md = "*⏳ 正在產生段落時間軸（多人追蹤），請稍候…*"

        # ── segment-level timeline (multi-person) ─────────────────────────
        print("[DEMO] calling analyze_timeline()...")
        seg_results = analyze_timeline(mp4_path, segment_sec=6.0, stride_sec=2.0)
        print(f"[DEMO] analyze_timeline() done: {len(seg_results)} segments")
        timeline_fig = _make_timeline_fig(seg_results)
        print("[DEMO] timeline figure ready")

        # Update summary: only persons with ≥3 active segments
        all_pids = sorted({r.person_id for r in seg_results}) if seg_results else []
        person_ids = [
            pid for pid in all_pids
            if sum(1 for r in seg_results if r.person_id == pid and r.status != "no_face") >= 3
        ] or all_pids  # fallback: show all if none pass
        # sort by first appearance time
        person_ids = sorted(
            person_ids,
            key=lambda p: min(r.t_start for r in seg_results if r.person_id == p),
        )
        if person_ids:
            lines = []
            for pid in person_ids:
                segs = [r for r in seg_results
                        if r.person_id == pid and r.status not in ("no_face", "uncertain")]
                if not segs:
                    lines.append(f"- **Person {pid + 1}**: ⚠️ 資料不足")
                    continue
                real_f = sum(1 for r in segs if r.status == "real") / len(segs)
                fake_f = sum(1 for r in segs if r.status == "fake") / len(segs)
                verdict = "✅ 真人" if real_f >= 0.5 else "🚨 疑似AI"
                lines.append(
                    f"- **Person {pid + 1}**: {verdict} "
                    f"（真實段 {real_f:.0%} / 偽造段 {fake_f:.0%}）"
                )
            person_summary = "\n### 各人物判定\n" + "\n".join(lines)
        else:
            person_summary = "\n> ⚠️ 未偵測到人臉，無法進行段落分析。"

        # ── speech / content analysis ──────────────────────────────────
        print("[DEMO] calling speech_analyzer.analyze()…")
        sp = _speech_analyzer.analyze(mp4_path)
        print(f"[DEMO] speech done: status={sp.scam_status} "
              f"lang={sp.language} t={sp.processing_time_ms:.0f}ms")

        _SPEECH_EMOJI = {
            "scam":      "🚨",
            "suspicious":"⚠️",
            "safe":      "✅",
            "no_audio":  "🔇",
        }
        sp_emoji = _SPEECH_EMOJI.get(sp.scam_status, "❓")

        if sp.scam_status == "no_audio":
            speech_md = f"\n> {sp_emoji} **語音分析：** {sp.explanation}"
        else:
            transcript_preview = (
                sp.transcript[:300] + "…"
                if len(sp.transcript) > 300 else sp.transcript
            )
            intent_detail = _format_intent_analysis(
                sp.all_similarities, title="語音意圖 vs 詐騙資料庫"
            )
            speech_md = f"""
---
## {sp_emoji} 語音內容分析（詐騙風險）

| 指標 | 數值 |
|------|------|
| **詐騙風險** | {sp.scam_status.upper()} |
| **信心度** | {sp.scam_confidence:.1%} |
| **偵測語言** | {sp.language} |
| **處理時間** | {sp.processing_time_ms:.0f} ms |

**風險說明：** {sp.explanation}

{intent_detail}

<details><summary>📝 語音轉錄文字（點擊展開）</summary>

{transcript_preview}

</details>"""

        heartbeat_ok = v.snr_db >= 3.0 and 40 <= v.hr_bpm <= 180
        hb_icon  = "✅" if heartbeat_ok else "❌"
        sync_ok  = v.face_neck_sync >= 0.35 if v.face_neck_sync != 0.0 else None
        sync_icon = ("✅" if sync_ok else "❌") if sync_ok is not None else "—"

        summary_md = f"""
| 檢測項目 | 結果 |
|---------|------|
| **可信心跳偵測** | {hb_icon} {"偵測到（{:.0f} BPM）".format(v.hr_bpm) if heartbeat_ok else "未偵測到可信心跳"} |
| **臉頸生理同步** | {sync_icon} {"同步一致（r={:.2f}）".format(v.face_neck_sync) if sync_ok else ("不一致，疑似換臉或生成（r={:.2f}）".format(v.face_neck_sync) if sync_ok is not None else "頸部 ROI 不足，無法比對")} |
{disclaimer}
**說明：** {v.explanation}
{person_summary}
{speech_md}
"""
        return badge, summary_md.strip(), timeline_fig

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[DEMO] _run_video_analysis EXCEPTION:\n{tb}")
        return "❌ **ERROR**", f"分析失敗：{type(exc).__name__}: {exc}", None
    finally:
        if tmp_created:
            try:
                Path(tmp_created).unlink(missing_ok=True)
            except Exception:
                pass


def analyze_video(video_path: str | None) -> tuple[str, str, plt.Figure | None]:
    """Called when user uploads a local file."""
    if video_path is None:
        return "（未上傳影片）", "", None
    # On Windows, Gradio's temp file may be locked by the file-serving layer.
    # Copy it to our own temp location first to avoid PermissionError.
    tmp_copy: str | None = None
    try:
        suffix = Path(video_path).suffix or ".mp4"
        tmp_fd, tmp_copy = tempfile.mkstemp(suffix=suffix)
        import os; os.close(tmp_fd)
        shutil.copy2(video_path, tmp_copy)
        return _run_video_analysis(tmp_copy)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[DEMO] analyze_video EXCEPTION:\n{tb}")
        return "❌ **ERROR**", f"分析失敗：{type(exc).__name__}: {exc}", None
    finally:
        if tmp_copy:
            try:
                Path(tmp_copy).unlink(missing_ok=True)
            except Exception:
                pass


def analyze_video_url(
    url: str, cookies_file: str | None = None
) -> tuple[str, str, plt.Figure | None]:
    """Called when user submits a URL (+ optional cookies.txt)."""
    url = (url or "").strip()
    if not url:
        return "（未輸入 URL）", "", None

    tmp_dir: str | None = None
    tmp_mp4: str | None = None
    try:
        tmp_mp4 = _download_url(url, cookies_path=cookies_file)
        tmp_dir = str(Path(tmp_mp4).parent)
        return _run_video_analysis(tmp_mp4)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[DEMO] analyze_video_url EXCEPTION:\n{tb}")
        return "❌ **ERROR**", f"分析失敗：{type(exc).__name__}: {exc}", None
    finally:
        if tmp_dir and Path(tmp_dir).exists():
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        if tmp_mp4 and Path(tmp_mp4).exists():
            try:
                Path(tmp_mp4).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tab 5 — Embedding Demo: Whistleblower vs Scammer
# ---------------------------------------------------------------------------
_DEMO_SAMPLES = [
    # (label, text, group)
    # Whistleblowers — natural short-video lawyer tone, no fraud script keywords
    ("Whistleblower A\n(lawyer, case experience)",
     "這種案件我一個月至少接到五件，受害者年齡層越來越廣，不只老人，年輕人一樣會被騙。",
     "lawyer"),
    ("Whistleblower B\n(lawyer, victim support)",
     "很多人被騙之後覺得自己很蠢，其實詐騙手法非常精密，不要怪自己，趕快報案保留證據才是重點。",
     "lawyer"),
    ("Whistleblower C\n(lawyer, action advice)",
     "收到這種電話，你第一個動作不是配合，是掛斷，然後打給你認識的人確認，不要一個人做決定。",
     "lawyer"),
    # Scammers — first-person, explicit target, urgency + threat + action demand
    ("Scammer A\n(friend impersonation)",
     "媽，是我啦！我手機壞掉了借同學的電話，我現在在醫院要繳手術費，你先匯3萬到這個帳號給我，等下我還你。",
     "scammer"),
    ("Scammer B\n(full gov't script)",
     "這裡是台北地檢署，你的帳戶涉及跨國洗錢案，為了保護你的財產安全，請你在今天內將存款全數轉到我們的監管帳戶，千萬不要告訴任何人，否則會影響辦案。",
     "scammer"),
    ("Scammer C\n(identity theft threat)",
     "你的個資已外洩，有人正在盜用你的身分貸款，你必須立刻配合我的指示操作才能阻止損失擴大。",
     "scammer"),
]


def run_embedding_demo() -> tuple[plt.Figure, plt.Figure, str]:
    """Compute intent similarities and return bar chart + PCA space plot + table."""
    from sklearn.decomposition import PCA
    from matplotlib.patches import Ellipse
    import matplotlib.patches as mpatches

    embedder = _text_det._embedder
    LAWYER_COLOR   = "#4A90D9"
    SCAMMER_COLOR  = "#E74C3C"
    ARCHETYPE_COLOR = "#888899"
    BG = "#0F1117"
    AX_BG = "#1A1D27"

    # ── collect embeddings ────────────────────────────────────────────────
    labels, vecs, sims, colors, archetypes, rows = [], [], [], [], [], []
    for label, text, group in _DEMO_SAMPLES:
        r   = embedder.compute_scam_distances(text)
        vec = embedder.embed(text)
        labels.append(label)
        vecs.append(vec)
        sims.append(r.max_similarity)
        colors.append(LAWYER_COLOR if group == "lawyer" else SCAMMER_COLOR)
        archetypes.append(r.closest_name_en)
        verdict = ("🚨 High-risk" if r.max_similarity >= 0.75
                   else "⚠️ Suspicious" if r.max_similarity >= 0.55
                   else "✅ Safe")
        rows.append(
            f"| **{label.replace(chr(10), ' ')}** "
            f"| {r.max_similarity:.3f} | {r.closest_name_en} | {verdict} |"
        )

    # archetype centroids
    arch_keys   = list(embedder._centroids.keys())
    arch_vecs   = [embedder._centroids[k] for k in arch_keys]
    arch_labels = [k.replace("_", " ").title() for k in arch_keys]

    all_vecs   = np.stack(vecs + arch_vecs)          # (6+9, 384)
    all_labels = labels + arch_labels
    all_colors = colors + [ARCHETYPE_COLOR] * len(arch_keys)

    # ── PCA 384 → 2 ───────────────────────────────────────────────────────
    pca  = PCA(n_components=2, random_state=42)
    pts  = pca.fit_transform(all_vecs)               # (15, 2)
    ev   = pca.explained_variance_ratio_

    sample_pts   = pts[:6]
    arch_pts     = pts[6:]

    # ── Figure 1: bar chart ───────────────────────────────────────────────
    fig_bar, ax = plt.subplots(figsize=(10, 5))
    fig_bar.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    N = len(labels)
    ax.barh(range(N), sims, color=colors, height=0.55, zorder=3)
    ax.axvline(0.55, color="#F39C12", lw=1.5, ls="--", zorder=4)
    ax.axvline(0.75, color="#E74C3C", lw=1.5, ls="--", zorder=4)
    ax.text(0.551, N - 0.1, "Suspicious", color="#F39C12", fontsize=7.5, va="top")
    ax.text(0.751, N - 0.1, "High-risk",  color="#E74C3C", fontsize=7.5, va="top")
    for i, (sim, arch) in enumerate(zip(sims, archetypes)):
        ax.text(sim + 0.008, i, f"{sim:.3f}  [{arch}]",
                va="center", ha="left", fontsize=8.5, color="white", zorder=5)
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels, fontsize=9, color="white")
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Max Cosine Similarity to Scam Archetype", color="#AAAAAA", fontsize=10)
    ax.set_title(
        "MiniLM Intent Similarity: Whistleblower vs Scammer\n"
        "paraphrase-multilingual-MiniLM-L12-v2 · 384-dim → cosine distance",
        color="white", fontsize=11, pad=10,
    )
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.grid(axis="x", color="#2A2D3A", zorder=0)
    ax.legend(handles=[
        mpatches.Patch(color=LAWYER_COLOR,  label="Whistleblower — Educational"),
        mpatches.Patch(color=SCAMMER_COLOR, label="Scammer — Targeting victim"),
    ], loc="lower right", facecolor=AX_BG, edgecolor="#444455", labelcolor="white", fontsize=9)
    plt.tight_layout()

    # ── Figure 2: PCA scatter ─────────────────────────────────────────────
    fig_pca, ax2 = plt.subplots(figsize=(9, 7))
    fig_pca.patch.set_facecolor(BG)
    ax2.set_facecolor(AX_BG)

    # draw confidence ellipses for the two sample groups
    for grp_idx, (gcol, gname) in enumerate([(LAWYER_COLOR, "Whistleblower"), (SCAMMER_COLOR, "Scammer")]):
        pts_g = sample_pts[grp_idx * 3: grp_idx * 3 + 3]
        cx, cy = pts_g.mean(axis=0)
        std = pts_g.std(axis=0).clip(min=0.005)
        ell = Ellipse((cx, cy), width=std[0] * 5, height=std[1] * 5,
                      angle=0, color=gcol, alpha=0.12, zorder=1)
        ax2.add_patch(ell)

    # archetype centroids
    ax2.scatter(arch_pts[:, 0], arch_pts[:, 1],
                c=ARCHETYPE_COLOR, marker="D", s=80, zorder=3, alpha=0.6,
                label="Scam Archetype Centroid")
    for i, lbl in enumerate(arch_labels):
        ax2.annotate(lbl, arch_pts[i], fontsize=6.5, color="#888899",
                     xytext=(4, 4), textcoords="offset points")

    # sample points
    ax2.scatter(sample_pts[:, 0], sample_pts[:, 1],
                c=colors, s=160, zorder=5, edgecolors="white", linewidths=0.8)
    for i, lbl in enumerate(labels):
        short = lbl.split("\n")[0]
        ax2.annotate(short, sample_pts[i], fontsize=8, color="white",
                     xytext=(6, 4), textcoords="offset points", fontweight="bold")

    ax2.set_xlabel(f"PC1 ({ev[0]:.1%} variance)", color="#AAAAAA", fontsize=10)
    ax2.set_ylabel(f"PC2 ({ev[1]:.1%} variance)", color="#AAAAAA", fontsize=10)
    ax2.set_title(
        "384-dim Embedding Space — PCA Projection\n"
        "Whistleblower vs Scammer vs Archetype Centroids",
        color="white", fontsize=11, pad=10,
    )
    ax2.tick_params(colors="#AAAAAA")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333344")
    ax2.grid(color="#2A2D3A", zorder=0)
    ax2.legend(handles=[
        mpatches.Patch(color=LAWYER_COLOR,  label="Whistleblower (Educational)"),
        mpatches.Patch(color=SCAMMER_COLOR, label="Scammer (Targeting victim)"),
        mpatches.Patch(color=ARCHETYPE_COLOR, label="Scam Archetype Centroid"),
    ], facecolor=AX_BG, edgecolor="#444455", labelcolor="white", fontsize=9)
    plt.tight_layout()

    table_md = (
        "### Intent Similarity Results\n\n"
        "| Sample | Similarity | Closest Archetype | Verdict |\n"
        "|--------|:----------:|-------------------|--------|\n"
        + "\n".join(rows)
        + "\n\n> **Thresholds:** ✅ Safe < 0.55 · ⚠️ Suspicious 0.55–0.75 · 🚨 High-risk ≥ 0.75"
    )
    return fig_bar, fig_pca, table_md


# ---------------------------------------------------------------------------
# Tab 4 — Unified Analysis (video file OR url, photo, text)
# ---------------------------------------------------------------------------
def analyze_unified(
    video_path: str | None,
    video_url: str,
    image: np.ndarray | None,
    text: str,
) -> str:
    results = []
    confidences = []
    fraud_flags = []

    # ---- Video (file takes priority over URL) ----
    effective_video: str | None = None
    tmp_dir_url: str | None = None
    tmp_mp4_url: str | None = None
    tmp_mp4_conv: str | None = None

    try:
        if video_path:
            effective_video = video_path
        elif (video_url or "").strip():
            try:
                tmp_mp4_url = _download_url(video_url.strip())
                tmp_dir_url = str(Path(tmp_mp4_url).parent)
                effective_video = tmp_mp4_url
            except Exception as exc:
                results.append(f"**影片 URL** ❌ 下載失敗：{exc}")

        if effective_video:
            mp4_path = _ensure_mp4(effective_video)
            if mp4_path != effective_video:
                tmp_mp4_conv = mp4_path
            vv = _video_det.analyze(mp4_path)
            results.append(f"**影片** {_badge(vv.status)} — 可信度 {vv.confidence:.1%} | {vv.explanation}")
            confidences.append(vv.confidence if vv.status == "fake" else 1.0 - vv.confidence)
            fraud_flags.append(vv.status == "fake")

        if image is not None:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            pv = _photo_det.analyze_array(img_bgr)
            results.append(f"**照片** {_badge(pv.status)} — 可信度 {pv.confidence:.1%} | {pv.explanation}")
            confidences.append(pv.confidence if pv.status == "fake" else 1.0 - pv.confidence)
            fraud_flags.append(pv.status == "fake")

        if text and text.strip():
            tv = _text_det.analyze(text.strip())
            results.append(f"**文字** {_badge(tv.status)} — 可信度 {tv.confidence:.1%} | {tv.explanation}")
            confidences.append(tv.confidence if tv.status == "scam" else 1.0 - tv.confidence)
            fraud_flags.append(tv.status == "scam")

        if not results:
            return "⚠️ 請至少提供一種輸入（影片 / 照片 / 文字）"

        fraud_count = sum(fraud_flags)
        avg_fraud_prob = sum(confidences) / len(confidences)

        if fraud_count > len(results) / 2:
            overall = "🚨 **高風險：偵測到詐騙 / 深偽內容**"
        elif fraud_count == 0 and avg_fraud_prob < 0.3:
            overall = "✅ **低風險：未偵測到明顯詐騙跡象**"
        else:
            overall = "⚠️ **中風險：部分指標異常，建議謹慎**"

        detail = "\n\n".join(f"- {r}" for r in results)
        return f"""
## 統合分析結果

{overall}

**綜合詐騙概率：** {avg_fraud_prob:.1%}

---

{detail}
""".strip()

    finally:
        # Clean up temp files
        for p in [tmp_mp4_conv]:
            if p and Path(p).exists():
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
        if tmp_dir_url and Path(tmp_dir_url).exists():
            try:
                shutil.rmtree(tmp_dir_url, ignore_errors=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="gray",
    neutral_hue="slate",
)

DESCRIPTION = """
# 🛡️ 多模態防詐防禦框架
**Multi-Modal Anti-Fraud Defense Framework**

大同大學 Lab605 研究團隊 | 指導教授：許超雲 講座教授

本系統透過三種獨立模組，從**物理層面**驗證數位內容的真實性：
- 📹 **影片**：rPPG 生理信號分析（額頭與臉頰脈動同步性）
- 🖼️ **照片**：鏡頭幾何 + PRNU 感測器指紋驗證
- 💬 **文字**：Transformer 意圖嵌入 + RAG 165 反詐資料庫比對
"""

with gr.Blocks(title="多模態防詐防禦框架") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # ---- Tab 1: Text ----
        with gr.TabItem("💬 文字分析"):
            gr.Markdown("輸入可疑文字訊息，系統將判斷其詐騙意圖並比對已知詐騙劇本。")
            with gr.Row():
                txt_input = gr.Textbox(
                    label="可疑訊息",
                    placeholder="貼上 SMS / LINE / Email 內容…",
                    lines=5,
                )
            txt_btn = gr.Button("🔍 分析", variant="primary")
            txt_badge = gr.Markdown()
            txt_result = gr.Markdown()
            txt_rag = gr.Markdown()

            gr.Examples(
                examples=[
                    ["您好，我是刑事局偵查員，您的帳戶涉嫌洗錢，請立即轉帳至監管帳戶以示清白，否則將發出逮捕令。"],
                    ["我們的加密貨幣平台保證每月30%回報，零風險，已有會員月賺百萬！限時優惠請加LINE：crypto888。"],
                    ["您的包裹因地址不完整無法投遞，請點擊連結更新資料並支付關稅NT$850：http://fake-post.com"],
                    ["嗨！我是在IG認識的新加坡工程師Tom，我有個穩定獲利的投資平台，想介紹你認識，你有興趣嗎？"],
                    ["今天天氣很好，下午我們去公園散步，順便喝杯咖啡怎麼樣？"],
                ],
                inputs=txt_input,
            )

            txt_btn.click(
                analyze_text,
                inputs=txt_input,
                outputs=[txt_badge, txt_result, txt_rag],
            )

        # ---- Tab 2: Photo ----
        with gr.TabItem("🖼️ 照片分析"):
            gr.Markdown("上傳圖片，系統透過鏡頭幾何一致性與 PRNU 感測器噪聲判斷是否為 AI 生成。")
            with gr.Row():
                img_input = gr.Image(label="上傳圖片（JPG / PNG）", type="numpy")
            img_btn = gr.Button("🔍 分析", variant="primary")
            img_badge = gr.Markdown()
            img_result = gr.Markdown()

            img_btn.click(
                analyze_photo,
                inputs=img_input,
                outputs=[img_badge, img_result],
            )

        # ---- Tab 3: Video ----
        with gr.TabItem("📹 影片分析"):
            gr.Markdown(
                "上傳包含人臉的影片（MP4 / AVI），或貼上影片網址，"
                "系統透過 rPPG 生理信號分析判斷是否為深偽影片。"
            )

            with gr.Tabs():
                with gr.TabItem("📁 上傳檔案"):
                    vid_input = gr.Video(label="上傳影片（MP4 / AVI）")
                    vid_file_btn = gr.Button("🔍 分析", variant="primary")
                    vid_file_badge = gr.Markdown()
                    vid_file_result = gr.Markdown()
                    vid_file_timeline = gr.Plot(
                        label="段落時間軸（各人物 rPPG 偵測）",
                        visible=True,
                    )

                    vid_file_btn.click(
                        analyze_video,
                        inputs=vid_input,
                        outputs=[vid_file_badge, vid_file_result, vid_file_timeline],
                    )

                with gr.TabItem("🔗 輸入網址"):
                    gr.Markdown(
                        "支援 YouTube、Facebook、Instagram、TikTok 及直接影片連結（mp4）。"
                    )
                    vid_url_input = gr.Textbox(
                        label="影片網址",
                        placeholder="https://www.youtube.com/watch?v=… 或直接 MP4 URL",
                    )
                    with gr.Accordion("🍪 Facebook / Instagram cookies（選填）", open=False):
                        gr.Markdown(
                            "**Facebook / Instagram 影片需登入才能下載。**\n\n"
                            "**方法：** 在 Chrome/Edge 安裝 "
                            "[Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc) 擴充功能，"
                            "登入 Facebook 後點擴充功能 → Export → 儲存為 `cookies.txt`，上傳至下方。\n\n"
                            "> 💡 YouTube / TikTok 公開影片**不需要**上傳 cookies。"
                        )
                        vid_cookies_file = gr.File(
                            label="cookies.txt（Netscape 格式）",
                            file_types=[".txt"],
                            type="filepath",
                        )
                    vid_url_btn = gr.Button("⬇️ 下載並分析", variant="primary")
                    vid_url_badge = gr.Markdown()
                    vid_url_result = gr.Markdown()
                    vid_url_timeline = gr.Plot(
                        label="段落時間軸（各人物 rPPG 偵測）",
                        visible=True,
                    )

                    vid_url_btn.click(
                        analyze_video_url,
                        inputs=[vid_url_input, vid_cookies_file],
                        outputs=[vid_url_badge, vid_url_result, vid_url_timeline],
                    )

        # ---- Tab 4: Unified ----
        with gr.TabItem("🔗 統合分析"):
            gr.Markdown("同時分析影片 + 照片 + 文字，輸出綜合風險評估。")
            with gr.Row():
                with gr.Column():
                    uni_vid = gr.Video(label="影片（選填，上傳檔案）")
                    uni_vid_url = gr.Textbox(
                        label="影片網址（選填，與上傳擇一）",
                        placeholder="https://…",
                    )
                    uni_img = gr.Image(label="照片（選填）", type="numpy")
                with gr.Column():
                    uni_txt = gr.Textbox(label="文字（選填）", lines=5)
            uni_btn = gr.Button("🔍 統合分析", variant="primary")
            uni_result = gr.Markdown()

            uni_btn.click(
                analyze_unified,
                inputs=[uni_vid, uni_vid_url, uni_img, uni_txt],
                outputs=uni_result,
            )

        # ---- Tab 5: Embedding Demo ----
        with gr.TabItem("🧪 Embedding 示範"):
            gr.Markdown(
                "## Whistleblower vs Scammer — MiniLM Intent Boundary\n\n"
                "**同樣的詐騙主題，不同的語氣立場。** "
                "吹哨者（律師）以**第三人稱描述**手法；詐騙者以**第二人稱命令式**鎖定受害者。\n\n"
                "點擊下方按鈕，即時計算六句話對詐騙 archetype 的 cosine similarity，"
                "驗證 embedding 能區分「談論詐騙」與「正在詐騙」。"
            )
            demo_btn = gr.Button("▶ 執行 Embedding 示範", variant="primary", size="lg")
            with gr.Row():
                demo_plot_bar = gr.Plot(label="Intent Similarity (Cosine Distance)")
                demo_plot_pca = gr.Plot(label="384-dim Embedding Space (PCA Projection)")
            demo_table = gr.Markdown()

            demo_btn.click(
                run_embedding_demo,
                inputs=[],
                outputs=[demo_plot_bar, demo_plot_pca, demo_table],
            )

    gr.Markdown("""
---
**資料來源：** 內政部警政署 165 反詐騙諮詢專線（政府資料開放平台）
**技術說明：** rPPG (Transformer) · PRNU (Wavelet) · Lens Geometry (RANSAC) · MiniLM + FAISS RAG
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=THEME,
    )
