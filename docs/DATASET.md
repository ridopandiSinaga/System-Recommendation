# Dataset contract

Sumber: [Kaggle Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
(`arashnic/book-recommendation-dataset`). Dataset ini berasal dari Book-Crossing.

## File wajib

```text
data/raw/Books.csv
data/raw/Ratings.csv
data/raw/Users.csv
```

File gambar yang ikut archive Kaggle tidak dipakai pipeline.

```bash
uv run kaggle datasets download \
  -d arashnic/book-recommendation-dataset \
  -p data/raw --unzip
uv run book-rec check-data --data-dir data/raw
```

Kaggle CLI membaca credential dari lokasi standar seperti
`~/.kaggle/kaggle.json` atau environment variable resmi. Credential tidak boleh
disimpan di repository.

## Snapshot lokal yang diaudit

Checksum berikut adalah file yang dipakai pada audit 10 Juli 2026. Jika Kaggle
memperbarui archive, checksum dapat berubah dan eksperimen harus dianggap memakai
source snapshot berbeda.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Books.csv` | 73,293,635 | `8a406befc83452ab2dded8050dec47bccb6aa8110edb2512b62d8402f1a3d8dd` |
| `Ratings.csv` | 22,633,892 | `fbb819408aa8d42e7675200b26d48d66660b6e712bbd82585e5b564053d29de9` |
| `Users.csv` | 11,017,438 | `d82fe997ee9bfc455d0090b67d235694b82a9479d03e036665945ac9d30981bc` |

Verifikasi PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 `
  data/raw/Books.csv,data/raw/Ratings.csv,data/raw/Users.csv
```

## Raw schema minimum

- Books: `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`,
  `Publisher`, dan URL image.
- Ratings: `User-ID`, `ISBN`, `Book-Rating`.
- Users: `User-ID`, `Location`, `Age`.

Loader menolak missing core columns, identifier invalid, rating non-finite/di
luar 0–10, serta rating ISBN yang tidak memiliki metadata setelah contract check.
Rating `0` adalah implicit/no explicit rating dan tidak masuk model
explicit-feedback.

## Lineage hasil preprocessing

Model manifest tidak hanya mengandalkan checksum raw di dokumen ini. Pipeline
menghitung order-independent fingerprint dari prepared books/ratings/users,
termasuk row count dan dtype schema. Evaluation artifact juga menyimpan source
dataset fingerprint dan training-split fingerprint.

Jalur download development di Streamlit membuat `dataset_manifest.json` berisi
SHA-256 dan ukuran file setelah full staging parse berhasil. File data dan
manifest dipublish sebagai satu directory swap agar tidak menyajikan campuran
snapshot lama dan baru.
