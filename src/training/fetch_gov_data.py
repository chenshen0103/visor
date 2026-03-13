"""
fetch_gov_data.py — 下載政府開放資料 165 反詐騙闢謠專區 CSV，
轉換為 RAG 所需的 JSONL 格式，並與現有語料合併後存回。

資料來源：
    https://data.gov.tw/dataset/38262
    更新頻率：不定期（觀察為每週）

用法：
    python src/training/fetch_gov_data.py
    python src/training/fetch_gov_data.py --output src/data/scam_corpus/165_chunks.jsonl
    python src/training/fetch_gov_data.py --dry-run   # 只顯示統計，不寫入
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional
import urllib.request

_SRC = Path(__file__).parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import CHUNKS_JSONL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 政府資料 API URL
# ---------------------------------------------------------------------------
GOV_CSV_URL = (
    "https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset"
    "/4F4DF9A5-DF4C-4EE8-A50D-869347D38D9E"
    "/resource/0F46DD38-FA8D-4B20-9E49-9A10604CDD10/download"
)

# ---------------------------------------------------------------------------
# Archetype keyword mapping（關鍵字 → archetype 分數）
# ---------------------------------------------------------------------------
_ARCHETYPE_KEYWORDS: dict[str, list[str]] = {
    "investment_fraud": [
        "投資", "股票", "加密貨幣", "虛擬貨幣", "幣", "外匯",
        "報酬", "獲利", "平台", "基金", "期貨", "債券",
        "操盤", "保本", "高報酬", "低風險", "證券", "財富",
    ],
    "parcel_fraud": [
        "包裹", "物流", "快遞", "宅急便", "運費", "配送",
        "貨物", "海關", "黑貓", "統一速達", "超商取貨", "郵件",
        "DHL", "FedEx", "UPS", "Amazon", "蝦皮",
    ],
    "romance_fraud": [
        "交友", "感情", "愛情", "殺豬", "徵友", "網戀",
        "約會", "戀愛", "相親", "情人", "交往", "男友", "女友",
    ],
    "government_impersonation": [
        "假冒", "公務", "警察", "刑事局", "地檢署", "法院",
        "健保", "稅務", "移民", "司法", "調查局", "海關",
        "政府", "機關", "公務員", "台電", "水費", "電費",
        "客服", "銀行", "信用卡", "金融機構", "紓困",
    ],
    "guess_who_i_am": [
        "猜猜我是誰", "換電話", "換號碼", "借錢", "週轉",
        "急需", "親友", "同學", "老師", "姪子", "換LINE",
    ],
    "atm_deduction_fraud": [
        "解除", "分期付款", "ATM", "操作", "重複扣款",
        "升級", "會員", "批發商", "經銷商", "設定錯誤",
        "網銀", "APP", "訂單錯誤", "扣款",
    ],
    "job_scam": [
        "求職", "工作", "兼職", "打字", "家庭代工",
        "佣金", "提成", "刷單", "點讚", "高薪", "每日只需",
        "材料費", "保證金", "代領", "代寄", "簿子", "提款卡",
    ],
    "phishing_link_fraud": [
        "連結", "網址", "點擊", "登入", "更新資料",
        "逾期", "罰單", "退稅", "異常", "凍結",
        "驗證身分", "修改密碼", "簡訊驗證碼", "OTP",
    ],
}


def classify_archetype(title: str, content: str) -> str:
    """Keyword-score each archetype; return winner (default: government_impersonation)."""
    combined = title + content
    scores: dict[str, int] = {k: 0 for k in _ARCHETYPE_KEYWORDS}
    for archetype, keywords in _ARCHETYPE_KEYWORDS.items():
        for kw in keywords:
            scores[archetype] += combined.count(kw)
    best = max(scores, key=lambda k: scores[k])
    # If all scores are 0, default to government_impersonation
    return best if scores[best] > 0 else "government_impersonation"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch_csv_bytes(url: str) -> bytes:
    logger.info("Downloading: %s", url)
    # Some gov.tw servers have non-standard certs (missing Subject Key Identifier).
    # We disable verification only for this known official URL.
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    logger.warning("SSL verification disabled for gov.tw certificate compatibility.")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (anti-fraud-project/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        data = resp.read()
    logger.info("Downloaded %.1f KB", len(data) / 1024)
    return data


def _decode_csv(raw: bytes) -> str:
    """Try UTF-8-sig → UTF-8 → Big5."""
    for enc in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def gov_csv_to_chunks(raw: bytes) -> list[dict]:
    text = _decode_csv(raw)
    reader = csv.DictReader(io.StringIO(text))
    chunks = []
    for row in reader:
        serial = row.get("編號", "").strip()
        title  = row.get("標題", "").strip()
        ts     = row.get("發佈時間", "").strip()
        body   = row.get("發佈內容", "").strip()

        if not title and not body:
            continue

        # Clean up excessive whitespace / newlines
        body = re.sub(r"\s+", " ", body).strip()

        chunk_text = f"{title}：{body}" if body else title
        archetype = classify_archetype(title, body)

        chunks.append({
            "chunk_id": f"gov_{serial.zfill(4)}",
            "text": chunk_text,
            "source": "165反詐騙諮詢專線闢謠專區",
            "label": "official_warning",
            "archetype": archetype,
            "published": ts,
        })

    logger.info("Parsed %d chunks from government CSV", len(chunks))
    return chunks


def load_existing_chunks(path: Path) -> list[dict]:
    """Load existing JSONL; try UTF-8 then Big5."""
    if not path.exists():
        return []
    for enc in ("utf-8", "utf-8-sig", "big5", "cp950"):
        try:
            with open(path, encoding=enc) as f:
                chunks = [json.loads(ln) for ln in f if ln.strip()]
            logger.info(
                "Loaded %d existing chunks from %s (encoding: %s)",
                len(chunks), path, enc,
            )
            return chunks
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    logger.warning("Could not decode existing chunks at %s; skipping.", path)
    return []


def merge_chunks(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge; government data takes precedence (de-duplicate by chunk_id)."""
    merged: dict[str, dict] = {}
    for c in existing:
        merged[c["chunk_id"]] = c
    for c in new:
        merged[c["chunk_id"]] = c
    result = list(merged.values())
    logger.info("Merged corpus: %d total chunks", len(result))
    return result


def save_chunks(chunks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks → %s", len(chunks), path)


def print_stats(chunks: list[dict]) -> None:
    from collections import Counter
    arch_counts = Counter(c.get("archetype", "unknown") for c in chunks)
    src_counts  = Counter(c.get("source", "unknown") for c in chunks)
    print("\n=== Corpus Stats ===")
    print(f"Total chunks: {len(chunks)}")
    print("\nBy archetype:")
    for k, v in arch_counts.most_common():
        print(f"  {k:35s} {v:4d}")
    print("\nBy source:")
    for k, v in src_counts.most_common():
        print(f"  {k:40s} {v:4d}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download 165 gov scam data and merge into RAG corpus"
    )
    parser.add_argument(
        "--output", default=str(CHUNKS_JSONL),
        help="Output JSONL path (default: config.CHUNKS_JSONL)"
    )
    parser.add_argument(
        "--url", default=GOV_CSV_URL,
        help="Override government CSV URL"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Download and parse but do not write files"
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # 1. Download
    raw = _fetch_csv_bytes(args.url)

    # 2. Parse government data
    gov_chunks = gov_csv_to_chunks(raw)

    # 3. Load existing corpus
    existing = load_existing_chunks(output_path)

    # 4. Merge
    merged = merge_chunks(existing, gov_chunks)

    # 5. Stats
    print_stats(merged)

    if args.dry_run:
        logger.info("--dry-run: skipping write.")
        return

    # 6. Save
    save_chunks(merged, output_path)
    logger.info(
        "Done. Next step: python src/training/build_rag_index.py"
    )


if __name__ == "__main__":
    main()
