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
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modules.text.text_detector import TextDetector
from modules.photo.photo_detector import PhotoDetector
from modules.video.video_detector import VideoDetector

# ---------------------------------------------------------------------------
# Load models once
# ---------------------------------------------------------------------------
print("[DEMO] Loading models…")
_text_det = TextDetector()
_text_det.load()

_photo_det = PhotoDetector()

_video_det = VideoDetector()
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
    "uncertain": "⚠️",
}


def _badge(status: str) -> str:
    return f"{STATUS_EMOJI.get(status, '❓')} **{status.upper()}**"


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
def _download_url(url: str) -> str:
    """
    Download *url* (YouTube / direct video / social media) to a temp MP4
    using yt-dlp (no ffmpeg required when the source is already mp4/webm).
    Returns path to the downloaded file.
    Raises RuntimeError on failure.
    """
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("yt-dlp not installed. Run: pip install yt-dlp")

    tmp_dir = tempfile.mkdtemp()
    out_template = str(Path(tmp_dir) / "video.%(ext)s")

    ydl_opts = {
        "outtmpl": out_template,
        # Select only pre-muxed single-file formats (no ffmpeg merge needed).
        # YouTube format 22 = 720p mp4 (combined), format 18 = 360p mp4 (combined).
        # Fallback chain: best combined mp4 → best combined webm → absolute best.
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

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

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
    badge = _badge(v.status)
    confidence_pct = f"{v.confidence:.1%}"

    summary_md = f"""
## 分析結果 {badge}

| 指標 | 數值 |
|------|------|
| **可信度** | {confidence_pct} |
| **最接近詐騙類型** | {v.closest_archetype_zh} ({v.closest_archetype}) |
| **意圖相似度** | {v.intent_similarity:.3f} |
| **RAG 詐騙比例** | {v.rag_scam_ratio:.1%} |
| **處理時間** | {v.processing_time_ms:.0f} ms |

**說明：** {v.explanation}
"""

    rag_md = ""
    if v.rag_evidence:
        rows = "\n".join(
            f"| {c.chunk_id} | {c.label} | {c.archetype} | {c.text[:80]}… |"
            for c in v.rag_evidence
        )
        rag_md = f"""
### RAG 檢索證據（Top-{len(v.rag_evidence)}）

| Chunk ID | 標籤 | 詐騙類型 | 內容摘要 |
|----------|------|----------|----------|
{rows}
"""

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
    badge = _badge(v.status)

    summary_md = f"""
## 分析結果 {badge}

| 指標 | 數值 |
|------|------|
| **可信度** | {v.confidence:.1%} |
| **鏡頭幾何分數** | {v.geometry_score:.3f} |
| **PRNU 感測器分數** | {v.prnu_score:.3f} |
| **週期性插值偽影** | {"是 ⚠️" if v.has_periodic_artifacts else "否 ✅"} |
| **處理時間** | {v.processing_time_ms:.0f} ms |

**說明：** {v.explanation}
"""
    return badge, summary_md.strip()


# ---------------------------------------------------------------------------
# Tab 3 — Video Analysis (file upload + URL)
# ---------------------------------------------------------------------------
def _run_video_analysis(video_path: str) -> tuple[str, str]:
    """Core analysis logic — accepts a local file path (mp4 or avi)."""
    tmp_created: str | None = None
    try:
        mp4_path = _ensure_mp4(video_path)
        if mp4_path != video_path:
            tmp_created = mp4_path   # remember to clean up

        v = _video_det.analyze(mp4_path)
        badge = _badge(v.status)

        summary_md = f"""
## 分析結果 {badge}

| 指標 | 數值 |
|------|------|
| **可信度** | {v.confidence:.1%} |
| **估計心率** | {v.hr_bpm:.1f} BPM |
| **跨區域同步 (Pearson r)** | {v.pearson_sync:.3f} |
| **信噪比 (SNR)** | {v.snr_db:.1f} dB |
| **處理時間** | {v.processing_time_ms:.0f} ms |

**說明：** {v.explanation}
"""
        return badge, summary_md.strip()

    except Exception as exc:
        return "❌ **ERROR**", f"分析失敗：{exc}"
    finally:
        if tmp_created:
            try:
                Path(tmp_created).unlink(missing_ok=True)
            except Exception:
                pass


def analyze_video(video_path: str | None) -> tuple[str, str]:
    """Called when user uploads a local file."""
    if video_path is None:
        return "（未上傳影片）", ""
    return _run_video_analysis(video_path)


def analyze_video_url(url: str) -> tuple[str, str]:
    """Called when user submits a URL."""
    url = (url or "").strip()
    if not url:
        return "（未輸入 URL）", ""

    tmp_dir: str | None = None
    tmp_mp4: str | None = None
    try:
        tmp_mp4 = _download_url(url)
        tmp_dir = str(Path(tmp_mp4).parent)
        return _run_video_analysis(tmp_mp4)
    except Exception as exc:
        return "❌ **ERROR**", f"下載或分析失敗：{exc}"
    finally:
        # Clean up temp directory created by yt-dlp
        if tmp_dir and Path(tmp_dir).exists():
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        # Clean up converted mp4 if it was placed outside tmp_dir
        if tmp_mp4 and Path(tmp_mp4).exists():
            try:
                Path(tmp_mp4).unlink(missing_ok=True)
            except Exception:
                pass


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

                    vid_file_btn.click(
                        analyze_video,
                        inputs=vid_input,
                        outputs=[vid_file_badge, vid_file_result],
                    )

                with gr.TabItem("🔗 輸入網址"):
                    gr.Markdown(
                        "支援 YouTube、Facebook、Instagram、TikTok 及直接影片連結（mp4）。"
                    )
                    vid_url_input = gr.Textbox(
                        label="影片網址",
                        placeholder="https://www.youtube.com/watch?v=… 或直接 MP4 URL",
                    )
                    vid_url_btn = gr.Button("⬇️ 下載並分析", variant="primary")
                    vid_url_badge = gr.Markdown()
                    vid_url_result = gr.Markdown()

                    vid_url_btn.click(
                        analyze_video_url,
                        inputs=vid_url_input,
                        outputs=[vid_url_badge, vid_url_result],
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
