"""
Central configuration for the Multi-Modal Anti-Fraud Defense Framework.
All thresholds, paths, and model hyper-parameters live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent

# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------
WEIGHTS_DIR = SRC_DIR / "models" / "weights"
PHYSFORMER_WEIGHTS = WEIGHTS_DIR / "physformer_lite.pth"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = SRC_DIR / "data"
SCAM_CORPUS_DIR = DATA_DIR / "scam_corpus"
CHUNKS_JSONL = SCAM_CORPUS_DIR / "165_chunks.jsonl"
SCAM_INTENTS_JSON = SCAM_CORPUS_DIR / "scam_intents.json"
FAISS_INDEX_PATH = SCAM_CORPUS_DIR / "faiss.index"
FAISS_META_PATH = SCAM_CORPUS_DIR / "faiss_meta.jsonl"

# ---------------------------------------------------------------------------
# Video / rPPG
# ---------------------------------------------------------------------------
VIDEO_FPS = 30
VIDEO_MAX_FRAMES = 300          # 10 seconds at 30 FPS
VIDEO_MIN_FRAMES = 60           # 2 seconds minimum
RPPG_WINDOW_SEC = 8             # sliding window for HR estimation
RPPG_OVERLAP = 0.5              # window overlap fraction

# Butterworth bandpass for rPPG signal
BANDPASS_LOW_HZ = 0.7           # ~42 BPM
BANDPASS_HIGH_HZ = 3.5          # ~210 BPM
BANDPASS_ORDER = 4

# Peak detection
PEAK_MIN_DISTANCE_SEC = 0.3     # minimum seconds between heartbeat peaks
PEAK_PROMINENCE = 0.05

# Sync / authenticity thresholds
# Calibrated against UBFC-rPPG real videos using POS classical rPPG:
#   real videos show POS sync in range [-0.07, 0.65] → threshold at 0.35
#   AI-generated video should show sync ≈ 0 with low SNR
SYNC_REAL_THRESHOLD = 0.35      # Pearson r above this → real region
SYNC_FAKE_THRESHOLD = 0.10      # Pearson r below this → fake region
SNR_MIN_REAL = 3.0              # dB

# Video verdict confidence weights
VIDEO_SYNC_WEIGHT = 0.60
VIDEO_SNR_WEIGHT = 0.40

# QuantumInspiredRPPGTransformer architecture (v8_filtered)
PHYSFORMER_D_MODEL = 32          # embed_dim
PHYSFORMER_N_HEADS = 2           # num_heads
PHYSFORMER_N_LAYERS = 2          # num_layers
PHYSFORMER_DFF = 64              # dim_feedforward
PHYSFORMER_DROPOUT = 0.2         # dropout
PHYSFORMER_MAX_SEQ_LEN = 160     # max_seq_len

# ROI sizes (pixels)
ROI_FOREHEAD_H = 30
ROI_FOREHEAD_W = 60
ROI_CHEEK_H = 25
ROI_CHEEK_W = 35

# ---------------------------------------------------------------------------
# Photo / PRNU
# ---------------------------------------------------------------------------
# Lens geometry branch
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LEN = 50
HOUGH_MAX_LINE_GAP = 10
RANSAC_MAX_TRIALS = 100
RANSAC_MIN_SAMPLES = 3
GEOMETRY_CONSISTENCY_THRESHOLD = 0.15   # normalised RMSE

# PRNU / noise branch
WAVELET = "db4"
WAVELET_LEVEL = 3
PRNU_ENERGY_REAL_THRESHOLD = 0.05       # real images have low structured noise
PERIODIC_ARTIFACT_THRESHOLD = 0.30     # fraction of FFT energy in peaks

# Fusion weights for photo verdict
PHOTO_GEOMETRY_WEIGHT = 0.40
PHOTO_PRNU_WEIGHT = 0.60

# ---------------------------------------------------------------------------
# Text / scam detection
# ---------------------------------------------------------------------------
SENTENCE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Cosine similarity thresholds
SCAM_SIMILARITY_HIGH = 0.75     # above this → high confidence scam
SCAM_SIMILARITY_MID = 0.55      # above this → suspicious

# RAG retrieval
RAG_TOP_K = 5

# Fusion weights for text verdict
TEXT_INTENT_WEIGHT = 0.40
TEXT_RAG_WEIGHT = 0.60

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_V1_PREFIX = "/api/v1"
MAX_UPLOAD_SIZE_MB = 100
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
