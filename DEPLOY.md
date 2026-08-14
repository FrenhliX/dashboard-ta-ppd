# Panduan Lengkap Deploy Dashboard ke Streamlit Community Cloud

Panduan ini untuk kamu yang belum pernah pakai Git/GitHub sama sekali. Ikuti dari atas ke bawah.

---

## BAGIAN A — Persiapan

### A1. Install Git
1. Download Git for Windows di https://git-scm.com/download/win
2. Install seperti biasa (klik Next terus, opsi default sudah aman).
3. Setelah selesai, buka **Git Bash** (muncul di Start Menu) untuk memastikan berhasil terinstall — ketik:
   ```
   git --version
   ```
   Kalau muncul versi (mis. `git version 2.44.0`), berarti sudah berhasil.

### A2. Buat akun GitHub
1. Buka https://github.com/signup
2. Daftar pakai email kamu, buat username & password.
3. Verifikasi email kamu.

### A3. Set identitas Git (sekali saja, di Git Bash)
```bash
git config --global user.name "Nama Kamu"
git config --global user.email "email_github_kamu@gmail.com"
```

---

## BAGIAN B — Siapkan Folder Project di Komputer

1. Extract `Dashboard_TA_PPD_Streamlit.zip` ke lokasi yang mudah diingat, misalnya:
   `C:\Users\NamaKamu\Documents\dashboard_ta`
2. (Opsional tapi disarankan) Copy folder model asli kamu ke dalam project:
   - Salin folder `indobertweet_v2` (isi: `config`, `model.safetensors`, `tokenizer`, `tokenizer_config`, `checkpoint_best.pt`) ke:
     `C:\Users\NamaKamu\Documents\dashboard_ta\models\indobertweet_v2\`
3. Pastikan struktur akhirnya seperti ini:
   ```
   dashboard_ta/
     app.py
     utils.py
     requirements.txt
     README.md
     data/
       overview_stats.json, ...
     pages/
       1_Analisis_Hasil.py
       2_Uji_Teks_Baru.py
     models/
       README.md
       indobertweet_v2/        <- kamu tambahkan ini
         config
         model.safetensors
         tokenizer
         tokenizer_config
         checkpoint_best.pt
     .streamlit/
       config.toml
   ```

### (Opsional) Coba jalankan dulu secara lokal untuk memastikan tidak ada error
1. Install Python dari https://www.python.org/downloads/ (centang "Add Python to PATH" saat instalasi).
2. Buka Command Prompt / PowerShell di folder `dashboard_ta`, lalu jalankan:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Browser akan otomatis terbuka ke `http://localhost:8501`. Kalau tampil dengan benar, lanjut ke Bagian C.

---

## BAGIAN C — Upload Project ke GitHub

### C1. Buat repository baru
1. Login ke https://github.com
2. Klik tombol **"+"** di kanan atas → **"New repository"**
3. Isi:
   - Repository name: `dashboard-ta-ppd` (bebas, tanpa spasi)
   - Pilih **Public** (atau **Private** kalau tidak ingin kode terlihat orang lain — tetap bisa di-deploy ke Streamlit Cloud)
   - **Jangan** centang "Add a README file" (biar tidak konflik nanti)
4. Klik **Create repository**. Biarkan halaman ini terbuka (nanti ada perintah `git remote add origin ...` yang perlu disalin).

### C2. Push project dari komputer ke GitHub (via Git Bash)
1. Buka Git Bash, arahkan ke folder project:
   ```bash
   cd "C:/Users/NamaKamu/Documents/dashboard_ta"
   ```
2. Jalankan perintah berikut satu per satu:
   ```bash
   git init
   git add .
   git commit -m "Dashboard TA - versi awal"
   git branch -M main
   git remote add origin https://github.com/USERNAME_GITHUB/dashboard-ta-ppd.git
   git push -u origin main
   ```
   (Ganti `USERNAME_GITHUB` dengan username GitHub kamu — URL persisnya ada di halaman repo yang tadi kamu buat, biasanya ada tombol "Copy" di bagian **"...or push an existing repository from the command line"**.)
3. Saat `git push`, akan diminta login — browser akan terbuka untuk otorisasi GitHub, ikuti saja instruksinya (atau masukkan **Personal Access Token** jika diminta password, bukan password akun biasa; buat token di https://github.com/settings/tokens jika perlu).
4. Setelah selesai, refresh halaman repo GitHub kamu — semua file (`app.py`, `pages/`, `data/`, dll) harus sudah muncul di sana.

### Alternatif tanpa Git: Upload manual lewat browser
Kalau tidak mau pakai Git sama sekali:
1. Di halaman repo GitHub yang baru dibuat, klik **"uploading an existing file"**.
2. Drag & drop semua file dan folder dari `dashboard_ta/` (bisa banyak kali upload karena folder harus di-drag terpisah untuk beberapa browser).
3. Klik **Commit changes**.

---

## BAGIAN D — Deploy ke Streamlit Community Cloud

1. Buka https://share.streamlit.io
2. Klik **Sign in with GitHub**, lalu izinkan akses (Authorize).
3. Klik **"New app"** (atau "Create app").
4. Isi form:
   - **Repository**: pilih `USERNAME_GITHUB/dashboard-ta-ppd`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - (Opsional) **App URL**: kamu bisa custom, mis. `dashboard-ta-ppd-namakamu`
5. Klik **Deploy**.
6. Tunggu 2–5 menit — Streamlit Cloud akan membaca `requirements.txt` dan menginstall `streamlit`, `pandas`, `plotly`, `torch`, `transformers` secara otomatis. Kamu bisa lihat progres instalasi di layar log real-time.
7. Kalau berhasil, dashboard langsung tampil dengan URL publik seperti:
   `https://dashboard-ta-ppd-namakamu.streamlit.app`
8. Bagikan link itu ke dosen pembimbing / lampirkan di laporan TA kamu.

---

## BAGIAN E — Update Dashboard di Kemudian Hari

Setiap kali kamu mengubah file secara lokal dan ingin memperbarui dashboard yang sudah online:
```bash
cd "C:/Users/NamaKamu/Documents/dashboard_ta"
git add .
git commit -m "Update dashboard"
git push
```
Streamlit Cloud akan otomatis mendeteksi push baru dan me-restart app dengan versi terbaru (tidak perlu deploy ulang dari awal).

---

## BAGIAN F — Troubleshooting Umum

| Masalah | Penyebab & Solusi |
|---|---|
| `ModuleNotFoundError` saat deploy | Ada package yang kepakai di kode tapi belum ada di `requirements.txt`. Tambahkan nama package-nya, lalu `git push` ulang. |
| Deploy gagal karena ukuran repo terlalu besar | GitHub membatasi file individual 100MB. Kalau model kamu lebih besar dari itu, gunakan [Git LFS](https://git-lfs.com/) atau host model di HuggingFace Hub / Google Drive dan download saat startup. |
| App "stuck" lama di log instalasi | Wajar untuk `torch`/`transformers` (ukurannya besar), tunggu sampai 5–10 menit. |
| App tampil tapi mode masih "Simulasi" padahal sudah upload model | Cek nama folder di `models/` benar-benar berisi `config.json` DAN `checkpoint_best.pt` langsung di dalamnya (bukan di dalam subfolder tambahan). |
| App "Zzzz... has gone to sleep" setelah tidak diakses lama | Normal untuk tier gratis Streamlit Cloud — klik "Yes, wake this app up", tunggu ~30 detik. |
| Git minta password saat `git push` dan ditolak | GitHub sudah tidak menerima password akun biasa untuk Git — buat **Personal Access Token** di https://github.com/settings/tokens (scope: `repo`), lalu pakai token itu sebagai password. |
