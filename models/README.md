# Folder Model

Berdasarkan notebook asli kamu (`IndoBERTweet_v2_Improved_Hasil.ipynb` &
`BERTopic_Pendekatan_Bigram_FIX_KEYWORD_Hasil.ipynb`), berikut cara menghubungkan
model asli ke dashboard ini.

## 1. IndoBERTweet (sentimen + aspek)

### Opsi A -- Lokal saja (di laptop kamu, tidak di-deploy online)

Taruh folder hasil training kamu (mis. `indobertweet_v2`) di dalam `models/`:
```
models/indobertweet_v2/
  config.json
  model.safetensors
  tokenizer.json / tokenizer_config.json
  checkpoint_best.pt
```
Dashboard otomatis mendeteksi folder apa pun di `models/` yang berisi
`config.json` + `checkpoint_best.pt`. Cocok untuk `streamlit run app.py` di laptop sendiri.

**JANGAN commit folder ini ke GitHub** -- `checkpoint_best.pt` (~423MB) dan
`model.safetensors` (~422MB) jauh di atas limit GitHub 100MB/file. Folder
`models/` sudah dikecualikan lewat `.gitignore` di project ini.

### Opsi B -- Untuk deploy online di Streamlit Cloud (WAJIB dipakai)

Karena GitHub menolak file >100MB, checkpoint model di-host di **Hugging Face
Hub** (gratis, tanpa limit ketat untuk file model) dan didownload otomatis oleh
dashboard saat pertama kali dibuka.

**Langkah setup:**
1. Buat akun gratis di https://huggingface.co/join
2. Klik profil -> **New Model** (https://huggingface.co/new). Beri nama, mis.
   `indobertweet-ppd`. Boleh Public atau Private.
3. Buka tab **"Files and versions"** di repo model tersebut -> **"Add file" ->
   "Upload files"** -> upload **`checkpoint_best.pt`** saja (~423MB, upload di
   browser HF bisa handle file besar, tunggu sampai selesai).
   - Kamu TIDAK perlu upload `config.json` / `model.safetensors` / `tokenizer*`
     -- dashboard sudah otomatis memakai backbone publik
     `indolem/indobertweet-base-uncased` dari Hugging Face untuk arsitektur +
     tokenizer, lalu bobotnya (encoder + head sentimen + head aspek) ditimpa
     total oleh `checkpoint_best.pt` kamu saat load. Hasilnya identik dengan
     kalau kamu punya folder lengkap.
4. Di Streamlit Cloud, buka app kamu -> **Settings -> Secrets**, isi:
   ```toml
   HF_REPO_ID = "username-huggingface-kamu/indobertweet-ppd"
   ```
   (Kalau repo Hugging Face-nya **Private**, tambahkan juga token akses:
   buat di https://huggingface.co/settings/tokens lalu isi `HF_TOKEN = "hf_..."`
   di Secrets yang sama.)
5. Simpan -- app akan otomatis restart dan mendownload checkpoint dari Hugging
   Face Hub saat pertama kali ada yang memakai fitur "Uji Teks Baru" (didownload
   sekali lalu di-cache, tidak berulang setiap prediksi).

Untuk testing lokal dengan cara yang sama (tanpa taruh folder di `models/`),
buat file `.streamlit/secrets.toml` berisi baris `HF_REPO_ID = "..."` yang sama.

Catatan teknis (sudah dikonfirmasi dari notebook kamu):
- `checkpoint_best.pt` = hasil `torch.save({'model_state_dict': ..., 'epoch': ...,
  'best_avg_f1': ...}, ...)` -- berisi **state_dict LENGKAP** (encoder + head
  sentimen + head aspek). Ini satu-satunya file yang benar-benar dibutuhkan.
- Kelas arsitektur `IndoBERTweetMultiTask` ditulis ulang persis di `utils.py`
  sesuai notebook cell 24 (encoder + 2 head: `Linear(h,256)->ReLU->Dropout->Linear(256,3)`).
- Urutan label sentimen: `Negative, Neutral, Positive` (index 0/1/2, softmax + argmax).
- Aspek bersifat multi-label: `Mental Health, Finansial, Support System`, sigmoid
  dengan threshold 0.5; jika tidak ada yang lolos threshold, dashboard mengambil
  skor tertinggi (sama seperti notebook cell 46).

## 2. BERTopic (sub-topik per aspek) -- BELUM bisa dimuat sebagai model asli

Setelah menelusuri notebook `BERTopic_Pendekatan_Bigram_FIX_KEYWORD_Hasil.ipynb`,
tidak ditemukan baris kode yang memanggil `topic_model.save(...)` -- notebook itu
hanya menyimpan **embeddings (`.npy`)** dan **gambar/CSV hasil visualisasi validasi**,
bukan objek `BERTopic` itu sendiri. Artinya model BERTopic per aspek (Mental
Health/Finansial/Support System) kemungkinan **belum pernah disimpan ke disk**
dan hanya ada di memori sesi Colab saat itu (sudah hilang setelah sesi berakhir).

Ada satu kendala teknis tambahan: vectorizer kamu memakai **custom analyzer
berupa fungsi Python biasa** (`analyzer=lambda doc: ngram_tokens_bersih(doc, n=2)`).
Fungsi seperti ini **tidak bisa di-pickle secara default** oleh `BERTopic.save()`
tanpa modifikasi (butuh `dill` atau memindahkan `ngram_tokens_bersih` ke modul
terpisah yang bisa di-import saat load).

**Karena itu, untuk saat ini dashboard memakai pencocokan sub-topik berbasis
frasa** (dari `data/topics.json`, yang berisi frasa c-TF-IDF ASLI hasil BERTopic
kamu) alih-alih memuat ulang model `.pkl`. Ini SELALU aktif, terlepas dari status
model IndoBERTweet di atas, dan sudah ditandai jelas di UI sebagai pencocokan
heuristik, bukan inferensi model BERTopic langsung.

Jika kamu ingin sub-topik benar-benar dihitung ulang oleh BERTopic asli, ini
yang perlu dilakukan di notebook (opsional, butuh re-run):
```python
# pindahkan def ngram_tokens_bersih(...) ke file terpisah, misalnya ngram_utils.py,
# lalu import dari sana (baik saat training maupun saat load nanti)
topic_model.save(
    f"{LOG_DIR}/bertopic_mental_health",
    serialization="pickle",  # perlu 'pickle' karena analyzer kustom, bukan 'safetensors'
    save_embedding_model=True,
)
```
Kemudian upload hasilnya ke Hugging Face Hub juga (sama seperti checkpoint di
atas, karena ukurannya juga kemungkinan >100MB) -- beri tahu saya kalau sudah,
nanti saya sambungkan loading-nya.
