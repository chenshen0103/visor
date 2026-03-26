"""
Experiment: Can MiniLM embedding distinguish
"lawyer educating about fraud" vs "actual scammer targeting a victim"?

Run from repo root:
    python src/experiments/intent_boundary_experiment.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# Use a CJK-capable font available on Windows
for _fname in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    _candidates = [f for f in fm.findSystemFonts() if _fname.replace(" ", "").lower() in f.replace(" ", "").lower()]
    if _candidates:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [_fname] + matplotlib.rcParams["font.sans-serif"]
        break

from modules.text.intent_embedder import IntentEmbedder

# ── Test sentences ────────────────────────────────────────────────────────────
SAMPLES = {
    # 3rd-person, descriptive, educational
    "Lawyer A\n(3rd-person, educating)":
        "詐騙者常假冒檢察官，告知民眾帳戶涉及洗錢，要求轉帳至「安全帳戶」配合調查。",
    "Lawyer B\n(3rd-person, analysis)":
        "假冒公務機關詐騙的特徵是以電話施壓，聲稱受害者涉及犯罪，並要求保密不得告知家人。",
    "Lawyer C\n(legal education)":
        "根據刑法第339條，詐欺罪最高可處五年有期徒刑。民眾若接到可疑電話應立即掛斷並撥165。",

    # 2nd-person, imperative, targeting victim
    "Scammer A\n(fake police)":
        "你的帳戶已被列為洗錢共犯，請立即配合檢察官指示，將存款轉入安全帳戶，否則將遭逮捕。",
    "Scammer B\n(investment fraud)":
        "這個內部消息只有你知道，保證獲利300%，今天就要匯款，機會稍縱即逝，不要告訴任何人。",
    "Scammer C\n(ATM deduction)":
        "您好，我是銀行客服，您的信用卡被誤設為商業帳戶，請馬上前往ATM按照我的指示操作解除。",
}

# ── Run embeddings ────────────────────────────────────────────────────────────
print("Loading IntentEmbedder...")
embedder = IntentEmbedder()
embedder.load()
print("Ready.\n")

labels = list(SAMPLES.keys())
texts  = list(SAMPLES.values())

results = [embedder.compute_scam_distances(t) for t in texts]
max_sims = [r.max_similarity for r in results]

# ── Plot ──────────────────────────────────────────────────────────────────────
LAWYER_COLOR = "#4A90D9"   # blue
SCAMMER_COLOR = "#E74C3C"  # red
N = len(labels)

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0F1117")
ax.set_facecolor("#1A1D27")

colors = [LAWYER_COLOR] * 3 + [SCAMMER_COLOR] * 3
bars = ax.barh(range(N), max_sims, color=colors, height=0.55, zorder=3)

# Threshold lines
ax.axvline(0.55, color="#F39C12", lw=1.5, ls="--", zorder=4, label="Suspicious threshold (0.55)")
ax.axvline(0.75, color="#E74C3C", lw=1.5, ls="--", zorder=4, label="High-risk threshold (0.75)")

# Value labels
for i, (bar, sim, r) in enumerate(zip(bars, max_sims, results)):
    ax.text(sim + 0.008, i, f"{sim:.3f}  [{r.closest_name_en}]",
            va="center", ha="left", fontsize=8.5, color="white", zorder=5)

ax.set_yticks(range(N))
ax.set_yticklabels(labels, fontsize=9, color="white")
ax.set_xlim(0, 1.05)
ax.set_xlabel("Max Cosine Similarity to Scam Archetype", color="#AAAAAA", fontsize=10)
ax.set_title(
    "MiniLM Intent Boundary: Lawyer (Education) vs Scammer (Targeting)\n"
    "paraphrase-multilingual-MiniLM-L12-v2 · 9 archetypes",
    color="white", fontsize=12, pad=12
)
ax.tick_params(colors="#AAAAAA")
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")
ax.grid(axis="x", color="#2A2D3A", zorder=0)

legend_patches = [
    mpatches.Patch(color=LAWYER_COLOR, label="Lawyer — Educational (3rd-person, descriptive)"),
    mpatches.Patch(color=SCAMMER_COLOR, label="Scammer — Targeting victim (2nd-person, imperative)"),
]
ax.legend(handles=legend_patches, loc="lower right",
          facecolor="#1A1D27", edgecolor="#444455", labelcolor="white", fontsize=9)

plt.tight_layout()
out = Path(__file__).parent / "intent_boundary.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
