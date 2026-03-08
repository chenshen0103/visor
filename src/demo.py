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
# Tab 3 — Video Analysis
# ---------------------------------------------------------------------------
def analyze_video(video_path: str | None) -> tuple[str, str]:
    if video_path is None:
        return "（未上傳影片）", ""

    v = _video_det.analyze(video_path)
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


# ---------------------------------------------------------------------------
# Tab 4 — Unified Analysis
# ---------------------------------------------------------------------------
def analyze_unified(
    video_path: str | None,
    image: np.ndarray | None,
    text: str,
) -> str:
    results = []
    confidences = []
    fraud_flags = []

    if video_path:
        vv = _video_det.analyze(video_path)
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
            gr.Markdown("上傳包含人臉的影片，系統透過 rPPG 生理信號分析判斷是否為深偽影片。")
            with gr.Row():
                vid_input = gr.Video(label="上傳影片（MP4 / AVI）")
            vid_btn = gr.Button("🔍 分析", variant="primary")
            vid_badge = gr.Markdown()
            vid_result = gr.Markdown()

            vid_btn.click(
                analyze_video,
                inputs=vid_input,
                outputs=[vid_badge, vid_result],
            )

        # ---- Tab 4: Unified ----
        with gr.TabItem("🔗 統合分析"):
            gr.Markdown("同時分析影片 + 照片 + 文字，輸出綜合風險評估。")
            with gr.Row():
                with gr.Column():
                    uni_vid = gr.Video(label="影片（選填）")
                    uni_img = gr.Image(label="照片（選填）", type="numpy")
                with gr.Column():
                    uni_txt = gr.Textbox(label="文字（選填）", lines=5)
            uni_btn = gr.Button("🔍 統合分析", variant="primary")
            uni_result = gr.Markdown()

            uni_btn.click(
                analyze_unified,
                inputs=[uni_vid, uni_img, uni_txt],
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
