# Dataset

Project ini memakai Kaggle Book Recommendation Dataset:

https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

File yang dibutuhkan:

- `Books.csv`
- `Ratings.csv`
- `Users.csv`

Letakkan ketiga file tersebut di:

```text
data/raw/
```

Cara download dengan Kaggle CLI:

```bash
kaggle datasets download -d arashnic/book-recommendation-dataset -p data/raw --unzip
```

Kaggle CLI membutuhkan credential `kaggle.json`. Simpan credential di lokasi standar
Kaggle, misalnya `~/.kaggle/kaggle.json`, atau gunakan environment variable yang
didukung Kaggle.
