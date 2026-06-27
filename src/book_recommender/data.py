from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BOOKS_FILE = "Books.csv"
RATINGS_FILE = "Ratings.csv"
USERS_FILE = "Users.csv"


@dataclass(frozen=True)
class BookDataset:
    books: pd.DataFrame
    ratings: pd.DataFrame
    users: pd.DataFrame


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected Kaggle CSV files are not available locally."""


def validate_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    missing = [name for name in (BOOKS_FILE, RATINGS_FILE, USERS_FILE) if not (path / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise DatasetNotFoundError(
            f"Missing dataset files in {path.resolve()}: {missing_text}. "
            "Download https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset "
            "and place the CSV files in data/raw."
        )
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def load_raw_data(data_dir: str | Path = "data/raw") -> BookDataset:
    path = validate_data_dir(data_dir)
    return BookDataset(
        books=_read_csv(path / BOOKS_FILE),
        ratings=_read_csv(path / RATINGS_FILE),
        users=_read_csv(path / USERS_FILE),
    )


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    renamed = books.rename(
        columns={
            "ISBN": "isbn",
            "Book-Title": "book_title",
            "Book-Author": "book_author",
            "Year-Of-Publication": "pub_year",
            "Publisher": "publisher",
            "Image-URL-S": "image_s_url",
            "Image-URL-M": "image_m_url",
            "Image-URL-L": "image_l_url",
        }
    ).copy()

    for column in [
        "isbn",
        "book_title",
        "book_author",
        "pub_year",
        "publisher",
        "image_s_url",
        "image_m_url",
        "image_l_url",
    ]:
        if column not in renamed.columns:
            renamed[column] = np.nan

    string_columns = [
        "isbn",
        "book_title",
        "book_author",
        "publisher",
        "image_s_url",
        "image_m_url",
        "image_l_url",
    ]
    for column in string_columns:
        renamed[column] = renamed[column].astype("string").str.strip()

    renamed = renamed.replace({"": np.nan})
    renamed = renamed.dropna(subset=["isbn", "book_title"])
    renamed["book_author"] = renamed["book_author"].fillna("Unknown")
    renamed["publisher"] = renamed["publisher"].fillna("Unknown")
    renamed["pub_year"] = pd.to_numeric(renamed["pub_year"], errors="coerce")

    columns = [
        "isbn",
        "book_title",
        "book_author",
        "pub_year",
        "publisher",
        "image_s_url",
        "image_m_url",
        "image_l_url",
    ]
    return renamed[columns].drop_duplicates(subset=["isbn"]).reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    renamed = ratings.rename(
        columns={
            "User-ID": "user_id",
            "ISBN": "isbn",
            "Book-Rating": "book_rating",
        }
    ).copy()
    required = ["user_id", "isbn", "book_rating"]
    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise ValueError(f"Ratings data is missing columns: {', '.join(missing)}")

    renamed["isbn"] = renamed["isbn"].astype("string").str.strip()
    renamed["book_rating"] = pd.to_numeric(renamed["book_rating"], errors="coerce")
    renamed["user_id"] = pd.to_numeric(renamed["user_id"], errors="coerce")
    renamed = renamed.dropna(subset=required)
    renamed = renamed[(renamed["book_rating"] >= 1) & (renamed["book_rating"] <= 10)]
    renamed["user_id"] = renamed["user_id"].astype("int64")
    renamed["book_rating"] = renamed["book_rating"].astype("float32")
    return renamed[required].drop_duplicates().reset_index(drop=True)


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    renamed = users.rename(
        columns={
            "User-ID": "user_id",
            "Location": "location",
            "Age": "age",
        }
    ).copy()
    for column in ["user_id", "location", "age"]:
        if column not in renamed.columns:
            renamed[column] = np.nan

    renamed["user_id"] = pd.to_numeric(renamed["user_id"], errors="coerce")
    renamed["age"] = pd.to_numeric(renamed["age"], errors="coerce")
    renamed["location"] = renamed["location"].astype("string").str.strip()
    renamed = renamed.dropna(subset=["user_id"])
    renamed["user_id"] = renamed["user_id"].astype("int64")
    return renamed[["user_id", "location", "age"]].drop_duplicates("user_id").reset_index(drop=True)


def filter_sparse_interactions(
    ratings: pd.DataFrame,
    min_user_ratings: int = 1,
    min_book_ratings: int = 1,
    max_rounds: int = 5,
) -> pd.DataFrame:
    filtered = ratings.copy()
    for _ in range(max_rounds):
        before = len(filtered)
        if min_user_ratings > 1:
            user_counts = filtered["user_id"].value_counts()
            filtered = filtered[filtered["user_id"].isin(user_counts[user_counts >= min_user_ratings].index)]
        if min_book_ratings > 1:
            book_counts = filtered["isbn"].value_counts()
            filtered = filtered[filtered["isbn"].isin(book_counts[book_counts >= min_book_ratings].index)]
        if len(filtered) == before:
            break
    return filtered.reset_index(drop=True)


def select_most_rated_books(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    max_books: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_books is None or max_books <= 0 or books["isbn"].nunique() <= max_books:
        return books.reset_index(drop=True), ratings.reset_index(drop=True)

    top_isbns = ratings["isbn"].value_counts().head(max_books).index
    selected_books = books[books["isbn"].isin(top_isbns)].reset_index(drop=True)
    selected_ratings = ratings[ratings["isbn"].isin(top_isbns)].reset_index(drop=True)
    return selected_books, selected_ratings


def prepare_dataset(
    data_dir: str | Path = "data/raw",
    min_user_ratings: int = 1,
    min_book_ratings: int = 1,
    max_books: int | None = None,
    max_ratings: int | None = None,
    random_state: int = 42,
) -> BookDataset:
    raw = load_raw_data(data_dir)
    books = clean_books(raw.books)
    ratings = clean_ratings(raw.ratings)
    users = clean_users(raw.users)

    ratings = ratings[ratings["isbn"].isin(books["isbn"])]
    ratings = filter_sparse_interactions(
        ratings,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
    )
    books = books[books["isbn"].isin(ratings["isbn"].unique()) | (ratings.empty)].reset_index(drop=True)
    books, ratings = select_most_rated_books(books, ratings, max_books=max_books)

    if max_ratings is not None and max_ratings > 0 and len(ratings) > max_ratings:
        ratings = ratings.sample(n=max_ratings, random_state=random_state).reset_index(drop=True)
        books = books[books["isbn"].isin(ratings["isbn"].unique())].reset_index(drop=True)

    return BookDataset(
        books=books.reset_index(drop=True),
        ratings=ratings.reset_index(drop=True),
        users=users.reset_index(drop=True),
    )


def dataset_summary(dataset: BookDataset) -> dict[str, int]:
    return {
        "books": int(len(dataset.books)),
        "ratings": int(len(dataset.ratings)),
        "users": int(len(dataset.users)),
        "rated_books": int(dataset.ratings["isbn"].nunique()) if not dataset.ratings.empty else 0,
        "rating_users": int(dataset.ratings["user_id"].nunique()) if not dataset.ratings.empty else 0,
    }
