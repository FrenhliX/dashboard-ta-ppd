"""
Fungsi-fungsi bantu (data loading & prediksi) untuk Dashboard TA
Analisis Topik & Sentimen Postpartum Depression (IndoBERTweet + BERTopic).

Arsitektur & logic inference disalin PERSIS dari notebook asli kamu
(IndoBERTweet_v2_Improved_Hasil.ipynb, cell 24 & cell 46).

Catatan penting soal ukuran file model (>100MB): checkpoint_best.pt TIDAK
diupload ke GitHub (limit GitHub 100MB/file). Sebagai gantinya, file itu
di-host di Hugging Face Hub dan didownload otomatis saat app pertama kali
jalan. Lihat models/README.md untuk cara setup-nya.
"""
import json
import os
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Backbone publik IndoBERTweet -- dipakai untuk membangun ARSITEKTUR model
# (config + tokenizer) tanpa perlu file besar apa pun di repo. Bobot asli
# (encoder + 2 head klasifikasi) tetap datang dari checkpoint_best.pt kamu,
# yang di-load & menimpa (override) bobot bawaan ini sepenuhnya.
BASE_ENCODER_NAME = "indolem/indobertweet-base-uncased"

ASPECTS = ["Mental Health", "Finansial", "Support System"]

# Label persis seperti di notebook (cell 12 & 46)
SENTIMEN_NAMES_EN = ["Negative", "Neutral", "Positive"]
SENTIMEN_EN_TO_ID = {"Negative": "Negatif", "Neutral": "Netral", "Positive": "Positif"}
ASPEK_NAMES = ["Mental Health", "Finansial", "Support System"]
ASPEK_THRESHOLD = 0.5
MAX_LENGTH = 128


@lru_cache(maxsize=None)
def load_json(name: str):
    with open(DATA_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def get_overview_stats():
    return load_json("overview_stats.json")


def get_split_ratio():
    return load_json("split_ratio.json")


def get_topic_metrics():
    return load_json("topic_metrics.json")


def get_topics():
    return load_json("topics.json")


def get_validation():
    return load_json("validation.json")


# ---------------------------------------------------------------------------
# Sumber checkpoint model asli: LOKAL (models/<folder>/checkpoint_best.pt)
# ATAU Hugging Face Hub (HF_REPO_ID di .streamlit/secrets.toml / env var).
# Ini menghindari perlunya commit file >100MB ke GitHub.
# ---------------------------------------------------------------------------

def _get_secret(key: str):
    try:
        import streamlit as st

        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)


def _find_local_checkpoint():
    """Cari checkpoint_best.pt di folder models/ (dipakai kalau kamu jalankan
    dashboard-nya sepenuhnya lokal, tanpa GitHub/Streamlit Cloud)."""
    if not MODELS_DIR.exists():
        return None, None
    for sub in MODELS_DIR.iterdir():
        ckpt = sub / "checkpoint_best.pt"
        if sub.is_dir() and ckpt.exists():
            # Kalau folder itu juga punya config.json (hasil save_pretrained
            # encoder asli kamu), pakai itu; kalau tidak, pakai backbone publik.
            encoder_source = str(sub) if (sub / "config.json").exists() else BASE_ENCODER_NAME
            return str(ckpt), encoder_source
    return None, None


def _download_hf_checkpoint():
    """Download checkpoint_best.pt dari Hugging Face Hub kalau HF_REPO_ID
    dikonfigurasi (lihat models/README.md)."""
    repo_id = _get_secret("HF_REPO_ID")
    if not repo_id:
        return None, None
    filename = _get_secret("HF_FILENAME") or "checkpoint_best.pt"
    try:
        from huggingface_hub import hf_hub_download

        token = _get_secret("HF_TOKEN")  # hanya perlu jika repo Hugging Face-nya private
        path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        return path, BASE_ENCODER_NAME
    except Exception as e:  # noqa
        print("[WARN] Gagal download checkpoint dari Hugging Face Hub:", e)
        return None, None


def _find_checkpoint():
    path, encoder_source = _find_local_checkpoint()
    if path:
        return path, encoder_source
    return _download_hf_checkpoint()


def real_model_available() -> bool:
    """True kalau ada checkpoint lokal ATAU HF_REPO_ID sudah dikonfigurasi
    (pengecekan optimis -- kegagalan aktual ditangani saat load/predict,
    dengan fallback otomatis ke mode simulasi)."""
    local_path, _ = _find_local_checkpoint()
    if local_path:
        return True
    return bool(_get_secret("HF_REPO_ID"))


def _build_model_class():
    """Definisi arsitektur PERSIS seperti notebook cell 24:
    IndoBERTweetMultiTask(encoder + head_sentimen + head_aspek)."""
    import torch.nn as nn
    from transformers import AutoModel

    class IndoBERTweetMultiTask(nn.Module):
        def __init__(self, model_name, num_sentimen=3, num_aspek=3, dropout=0.3):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            h = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.head_sentimen = nn.Sequential(
                nn.Linear(h, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_sentimen)
            )
            self.head_aspek = nn.Sequential(
                nn.Linear(h, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_aspek)
            )

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = self.dropout(out.last_hidden_state[:, 0, :])
            return self.head_sentimen(cls), self.head_aspek(cls)

    return IndoBERTweetMultiTask


_REAL = {"loaded": False, "tokenizer": None, "model": None, "error": None}


def _load_real_models_uncached():
    ckpt_path, encoder_source = _find_checkpoint()
    if ckpt_path is None:
        return {"tokenizer": None, "model": None, "error": "Checkpoint tidak ditemukan (lokal maupun Hugging Face Hub)."}
    import torch
    from transformers import AutoTokenizer

    ModelClass = _build_model_class()
    tokenizer = AutoTokenizer.from_pretrained(encoder_source)
    model = ModelClass(model_name=encoder_source, dropout=0.3)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "best_epoch": ckpt.get("epoch"),
        "best_avg_f1": ckpt.get("best_avg_f1"),
        "error": None,
    }


def _try_load_real_models():
    """Load sekali & cache. Memakai st.cache_resource jika tersedia (di dalam
    Streamlit) supaya model+checkpoint besar tidak di-download/dimuat ulang
    setiap kali user menekan tombol Prediksi."""
    if _REAL["loaded"]:
        return _REAL
    try:
        import streamlit as st

        cached_loader = st.cache_resource(show_spinner="Memuat model IndoBERTweet...")(_load_real_models_uncached)
        result = cached_loader()
    except Exception:
        result = _load_real_models_uncached()
    _REAL.update(result)
    _REAL["loaded"] = True
    if result.get("error"):
        print("[WARN] Gagal memuat model IndoBERTweet asli, fallback ke simulasi:", result["error"])
    return _REAL


def _predict_real(text: str):
    """Inference PERSIS seperti fungsi predict() di notebook cell 46."""
    import torch

    models = _try_load_real_models()
    tokenizer = models["tokenizer"]
    model = models["model"]
    enc = tokenizer(
        text, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        lg_s, lg_a = model(enc["input_ids"], enc["attention_mask"])
    probs_s = torch.softmax(lg_s, dim=1)[0].numpy()
    probs_a = torch.sigmoid(lg_a)[0].numpy()
    pred_s_idx = int(probs_s.argmax())
    pred_s = SENTIMEN_EN_TO_ID[SENTIMEN_NAMES_EN[pred_s_idx]]
    pred_a = [ASPEK_NAMES[i] for i, p in enumerate(probs_a) if p >= ASPEK_THRESHOLD]
    if not pred_a:
        pred_a = [ASPEK_NAMES[int(probs_a.argmax())]]
    conf = float(probs_s[pred_s_idx])
    return pred_s, conf, pred_a


# ---------------------------------------------------------------------------
# Mode SIMULASI (dipakai selama model asli belum berhasil dimuat)
# ---------------------------------------------------------------------------

_POS_WORDS = [
    "alhamdulillah", "bahagia", "senang", "bersyukur", "terima kasih", "hebat",
    "suka", "baik", "kuat", "semangat", "lega", "sehat", "nyaman", "tenang",
    "peran suami", "kebaikan", "supportif",
]
_NEG_WORDS = [
    "sedih", "menangis", "nangis", "capek", "lelah", "kelelahan", "depresi", "stress",
    "stres", "marah", "kesal", "takut", "cemas", "khawatir", "kurang tidur",
    "sendirian", "sendiri", "gagal", "susah", "berat", "minim", "tolong", "habis",
]


def _simulate_sentiment(text: str):
    t = text.lower()
    pos = sum(1 for w in _POS_WORDS if w in t)
    neg = sum(1 for w in _NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return "Netral", 0.5
    if pos > neg:
        return "Positif", min(0.5 + 0.15 * (pos - neg), 0.97)
    if neg > pos:
        return "Negatif", min(0.5 + 0.15 * (neg - pos), 0.97)
    return "Netral", 0.5


def _simulate_aspects(text: str):
    t = text.lower()
    aspek_hits = {}
    for aspek, subtopik_list in get_topics().items():
        hits = 0
        for st in subtopik_list:
            hits += sum(1 for frasa in st["frasa"] if any(w in t for w in frasa.split()))
        aspek_hits[aspek] = hits
    return [a for a, h in aspek_hits.items() if h > 0]


# ---------------------------------------------------------------------------
# Pencocokan sub-topik (selalu berbasis frasa hasil BERTopic asli di topics.json)
# ---------------------------------------------------------------------------

def _score_aspect_topics(text: str):
    t = text.lower()
    topics = get_topics()
    results = {}
    for aspek, subtopik_list in topics.items():
        best = None
        best_score = 0
        for st in subtopik_list:
            hits = sum(1 for frasa in st["frasa"] if any(w in t for w in frasa.split()))
            if hits > best_score:
                best_score = hits
                best = st
        results[aspek] = {
            "skor": best_score,
            "sub_topik": best["tema"] if best and best_score > 0 else None,
            "frasa_cocok": best["frasa"] if best and best_score > 0 else [],
        }
    return results


def predict_text(text: str) -> dict:
    """Prediksi sentimen + aspek (model asli jika tersedia) + sub-topik
    (selalu heuristik berbasis frasa BERTopic asli)."""
    text = (text or "").strip()
    if not text:
        return {"mode": "kosong"}

    aspek_scores = _score_aspect_topics(text)

    if real_model_available():
        try:
            sentimen, conf, aspek_terdeteksi = _predict_real(text)
            return {
                "mode": "real",
                "sentimen": sentimen,
                "confidence_sentimen": conf,
                "aspek_terdeteksi": aspek_terdeteksi,
                "detail_aspek": aspek_scores,
            }
        except Exception as e:  # noqa
            print("[WARN] Inference model asli gagal, fallback ke simulasi:", e)

    sentimen, conf = _simulate_sentiment(text)
    aspek_terdeteksi = _simulate_aspects(text)
    return {
        "mode": "simulasi",
        "sentimen": sentimen,
        "confidence_sentimen": conf,
        "aspek_terdeteksi": aspek_terdeteksi,
        "detail_aspek": aspek_scores,
    }
