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
    # Whistleblowers — natural short-video lawyer tone, no fraud script keywords
    "Whistleblower A\n(case experience)":
        "這種案件我一個月至少接到五件，受害者年齡層越來越廣，不只老人，年輕人一樣會被騙。",
    "Whistleblower B\n(victim support)":
        "很多人被騙之後覺得自己很蠢，其實詐騙手法非常精密，不要怪自己，趕快報案保留證據才是重點。",
    "Whistleblower C\n(action advice)":
        "收到這種電話，你第一個動作不是配合，是掛斷，然後打給你認識的人確認，不要一個人做決定。",

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

# ── PCA scatter ───────────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from modules.text.intent_embedder import IntentEmbedder as _IE
from modules.text.scam_patterns import SCAM_ARCHETYPES

# reuse already-loaded embedder
arch_keys   = list(embedder._centroids.keys())
arch_vecs   = [embedder._centroids[k] for k in arch_keys]
arch_labels = [k.replace("_", " ").title() for k in arch_keys]

sample_vecs = [embedder.embed(t) for t in texts]
all_vecs    = np.stack(sample_vecs + arch_vecs)
pca         = PCA(n_components=2, random_state=42)
pts         = pca.fit_transform(all_vecs)
ev          = pca.explained_variance_ratio_

sample_pts = pts[:len(labels)]
arch_pts   = pts[len(labels):]

ARCHETYPE_COLOR = "#888899"
fig2, ax2 = plt.subplots(figsize=(9, 7))
fig2.patch.set_facecolor("#0F1117")
ax2.set_facecolor("#1A1D27")

from matplotlib.patches import Ellipse
for grp_idx, gcol in enumerate([LAWYER_COLOR, SCAMMER_COLOR]):
    pts_g = sample_pts[grp_idx * 3: grp_idx * 3 + 3]
    cx, cy = pts_g.mean(axis=0)
    std = pts_g.std(axis=0).clip(min=0.005)
    ax2.add_patch(Ellipse((cx, cy), width=std[0]*5, height=std[1]*5,
                          color=gcol, alpha=0.12, zorder=1))

ax2.scatter(arch_pts[:, 0], arch_pts[:, 1], c=ARCHETYPE_COLOR,
            marker="D", s=80, zorder=3, alpha=0.6, label="Scam Archetype Centroid")
for i, lbl in enumerate(arch_labels):
    ax2.annotate(lbl, arch_pts[i], fontsize=6.5, color="#888899",
                 xytext=(4, 4), textcoords="offset points")

ax2.scatter(sample_pts[:, 0], sample_pts[:, 1], c=colors[:len(labels)],
            s=160, zorder=5, edgecolors="white", linewidths=0.8)
for i, lbl in enumerate(labels):
    ax2.annotate(lbl.split("\n")[0], sample_pts[i], fontsize=8, color="white",
                 xytext=(6, 4), textcoords="offset points", fontweight="bold")

ax2.set_xlabel(f"PC1 ({ev[0]:.1%} variance)", color="#AAAAAA", fontsize=10)
ax2.set_ylabel(f"PC2 ({ev[1]:.1%} variance)", color="#AAAAAA", fontsize=10)
ax2.set_title("384-dim Embedding Space — PCA Projection\nWhistleblower vs Scammer vs Archetype Centroids",
              color="white", fontsize=11, pad=10)
ax2.tick_params(colors="#AAAAAA")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333344")
ax2.grid(color="#2A2D3A", zorder=0)
ax2.legend(handles=[
    mpatches.Patch(color=LAWYER_COLOR,   label="Whistleblower (Educational)"),
    mpatches.Patch(color=SCAMMER_COLOR,  label="Scammer (Targeting victim)"),
    mpatches.Patch(color=ARCHETYPE_COLOR,label="Scam Archetype Centroid"),
], facecolor="#1A1D27", edgecolor="#444455", labelcolor="white", fontsize=9)
plt.tight_layout()

out2 = Path(__file__).parent / "embedding_space.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Saved → {out2}")
