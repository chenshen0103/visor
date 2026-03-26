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
    # Whistleblowers — policy / statistics / court ruling — no script language
    "Whistleblower A\n(psych. analysis)":
        "詐騙集團慣用時間壓力與恐懼感使受害者無法冷靜，這是社會工程攻擊的核心手法。",
    "Whistleblower B\n(court verdict)":
        "臺灣高等法院裁定，本案被告以假冒警察名義詐取被害人存款，判處有期徒刑三年六個月。",
    "Whistleblower C\n(crime statistics)":
        "根據2024年警政署統計，假冒公務機關詐騙案件達3.2萬件，全年損失逾60億元，為各類詐騙之冠。",

    # Scammers — first-person, explicit target, urgency + threat + action demand
    "Scammer A\n(friend impersonation)":
        "媽，是我啦！我手機壞掉了借同學的電話，我現在在醫院要繳手術費，你先匯3萬到這個帳號給我，等下我還你。",
    "Scammer B\n(full gov't script)":
        "這裡是台北地檢署，你的帳戶涉及跨國洗錢案，為了保護你的財產安全，請你在今天內將存款全數轉到我們的監管帳戶，千萬不要告訴任何人，否則會影響辦案。",
    "Scammer C\n(identity theft threat)":
        "你的個資已外洩，有人正在盜用你的身分貸款，你必須立刻配合我的指示操作才能阻止損失擴大。",
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
