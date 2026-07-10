from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd


BOOKS_FILE = "Books.csv"
RATINGS_FILE = "Ratings.csv"
USERS_FILE = "Users.csv"

BOOK_CORE_COLUMNS = ("isbn", "book_title", "book_author", "publisher")
BOOK_OPTIONAL_COLUMNS = ("pub_year", "image_s_url", "image_m_url", "image_l_url")
RATING_COLUMNS = ("user_id", "isbn", "book_rating")
USER_COLUMNS = ("user_id", "location", "age")

_MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class BookDataset:
    books: pd.DataFrame
    ratings: pd.DataFrame
    users: pd.DataFrame


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the expected Kaggle CSV files are not available locally."""


def _renamed_with_required_columns(
    frame: pd.DataFrame,
    *,
    rename_map: dict[str, str],
    required: tuple[str, ...],
    dataset_name: str,
) -> pd.DataFrame:
    """Rename raw columns and reject ambiguous or incomplete input schemas."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{dataset_name} data must be a pandas DataFrame.")

    renamed = frame.rename(columns=rename_map).copy()
    duplicate_columns = sorted(set(renamed.columns[renamed.columns.duplicated()].tolist()))
    if duplicate_columns:
        raise ValueError(
            f"{dataset_name} data has duplicate columns after normalization: "
            f"{', '.join(duplicate_columns)}"
        )

    missing = [column for column in required if column not in renamed.columns]
    if missing:
        raise ValueError(f"{dataset_name} data is missing columns: {', '.join(missing)}")
    return renamed


def _normalized_isbns(values: pd.Series, *, dataset_name: str) -> pd.Series:
    """Normalize ISBN identifiers without imposing unreliable checksum rules."""
    normalized = values.astype("string").str.strip()
    invalid = normalized.isna() | normalized.eq("")
    invalid |= normalized.str.casefold().isin({"nan", "none", "null"}).fillna(False)
    if invalid.any():
        rows = ", ".join(map(str, values.index[invalid][:5].tolist()))
        raise ValueError(
            f"{dataset_name} data contains missing or blank ISBN values at rows: {rows}"
        )
    return normalized


def _valid_user_ids(values: pd.Series, *, dataset_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = pd.Series(np.isfinite(numeric), index=values.index)
    invalid = numeric.isna() | ~finite
    invalid |= numeric.le(0) | numeric.gt(np.iinfo(np.int64).max)
    invalid |= numeric.mod(1).ne(0)
    if invalid.any():
        rows = ", ".join(map(str, values.index[invalid][:5].tolist()))
        raise ValueError(
            f"{dataset_name} data contains invalid user_id values at rows: {rows}. "
            "User IDs must be positive integers."
        )
    return numeric.astype("int64")


def _validate_integer_option(
    name: str,
    value: int | None,
    *,
    minimum: int,
    allow_none: bool = False,
    maximum: int | None = None,
) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")

    normalized = int(value)
    if normalized < minimum or (maximum is not None and normalized > maximum):
        range_text = (
            f" between {minimum} and {maximum}" if maximum is not None else f" >= {minimum}"
        )
        raise ValueError(f"{name} must be{range_text}.")
    return normalized


def _validate_prepare_options(
    *,
    min_user_ratings: int,
    min_book_ratings: int,
    max_books: int | None,
    max_ratings: int | None,
    random_state: int,
) -> tuple[int, int, int | None, int | None, int]:
    return (
        int(_validate_integer_option("min_user_ratings", min_user_ratings, minimum=1)),
        int(_validate_integer_option("min_book_ratings", min_book_ratings, minimum=1)),
        _validate_integer_option("max_books", max_books, minimum=1, allow_none=True),
        _validate_integer_option("max_ratings", max_ratings, minimum=1, allow_none=True),
        int(
            _validate_integer_option(
                "random_state",
                random_state,
                minimum=0,
                maximum=_MAX_RANDOM_STATE,
            )
        ),
    )


def validate_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    missing = [
        name for name in (BOOKS_FILE, RATINGS_FILE, USERS_FILE) if not (path / name).exists()
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise DatasetNotFoundError(
            f"Missing dataset files in {path.resolve()}: {missing_text}. "
            "Download https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset "
            "and place the CSV files in data/raw."
        )
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    read_kwargs = {"low_memory": True}
    if path.name == BOOKS_FILE:
        read_kwargs["dtype"] = {"Year-Of-Publication": "string"}

    try:
        return pd.read_csv(path, **read_kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **read_kwargs)


def load_raw_data(data_dir: str | Path = "data/raw") -> BookDataset:
    path = validate_data_dir(data_dir)
    return BookDataset(
        books=_read_csv(path / BOOKS_FILE),
        ratings=_read_csv(path / RATINGS_FILE),
        users=_read_csv(path / USERS_FILE),
    )


def load_clean_dataset(data_dir: str | Path = "data/raw") -> BookDataset:
    return clean_dataset(load_raw_data(data_dir))


def clean_dataset(dataset: BookDataset) -> BookDataset:
    books = clean_books(dataset.books)
    ratings = clean_ratings(dataset.ratings)
    users = clean_users(dataset.users)

    ratings = ratings[ratings["isbn"].isin(books["isbn"])].reset_index(drop=True)
    return BookDataset(
        books=books.reset_index(drop=True),
        ratings=ratings,
        users=users.reset_index(drop=True),
    )


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    """Normalize book metadata while requiring columns used by the recommenders.

    Publication year and image URLs are optional because recommendation quality does not
    depend on them. ISBN, title, author, and publisher are part of the model input contract.
    Missing author/publisher values are allowed and represented as ``"Unknown"``.
    """
    renamed = _renamed_with_required_columns(
        books,
        rename_map={
            "ISBN": "isbn",
            "Book-Title": "book_title",
            "Book-Author": "book_author",
            "Year-Of-Publication": "pub_year",
            "Publisher": "publisher",
            "Image-URL-S": "image_s_url",
            "Image-URL-M": "image_m_url",
            "Image-URL-L": "image_l_url",
        },
        required=BOOK_CORE_COLUMNS,
        dataset_name="Books",
    )

    for column in BOOK_OPTIONAL_COLUMNS:
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
    renamed["isbn"] = _normalized_isbns(renamed["isbn"], dataset_name="Books")
    if renamed["book_title"].isna().any():
        rows = ", ".join(map(str, renamed.index[renamed["book_title"].isna()][:5].tolist()))
        raise ValueError(f"Books data contains missing or blank book_title values at rows: {rows}")
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
    """Return explicit ratings with one deterministic value per user/ISBN pair.

    Book-Crossing uses rating ``0`` for implicit/no explicit score, so those rows are
    intentionally excluded. If a source contains multiple explicit scores for the same
    user and ISBN, their arithmetic mean is used. The mean is order-independent and avoids
    pretending that an undated row is the "latest" observation.
    """
    renamed = _renamed_with_required_columns(
        ratings,
        rename_map={
            "User-ID": "user_id",
            "ISBN": "isbn",
            "Book-Rating": "book_rating",
        },
        required=RATING_COLUMNS,
        dataset_name="Ratings",
    )

    renamed["isbn"] = _normalized_isbns(renamed["isbn"], dataset_name="Ratings")
    renamed["user_id"] = _valid_user_ids(renamed["user_id"], dataset_name="Ratings")

    numeric_ratings = pd.to_numeric(renamed["book_rating"], errors="coerce")
    finite = pd.Series(np.isfinite(numeric_ratings), index=renamed.index)
    invalid = numeric_ratings.isna() | ~finite | numeric_ratings.lt(0) | numeric_ratings.gt(10)
    if invalid.any():
        rows = ", ".join(map(str, renamed.index[invalid][:5].tolist()))
        raise ValueError(
            f"Ratings data contains invalid book_rating values at rows: {rows}. "
            "Ratings must be numeric values from 0 through 10."
        )

    renamed["book_rating"] = numeric_ratings
    explicit = renamed[renamed["book_rating"] > 0]
    if explicit.empty:
        return pd.DataFrame(
            {
                "user_id": pd.Series(dtype="int64"),
                "isbn": pd.Series(dtype="string"),
                "book_rating": pd.Series(dtype="float32"),
            }
        )

    # sort=False preserves the source order for the common no-duplicate case while the
    # aggregation policy itself is independent of which duplicate appears first.
    deduplicated = (
        explicit.groupby(["user_id", "isbn"], sort=False, as_index=False, dropna=False)
        .agg(book_rating=("book_rating", "mean"))
        .reset_index(drop=True)
    )
    deduplicated["user_id"] = deduplicated["user_id"].astype("int64")
    deduplicated["isbn"] = deduplicated["isbn"].astype("string")
    deduplicated["book_rating"] = deduplicated["book_rating"].astype("float32")
    return deduplicated[list(RATING_COLUMNS)]


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    renamed = _renamed_with_required_columns(
        users,
        rename_map={
            "User-ID": "user_id",
            "Location": "location",
            "Age": "age",
        },
        required=USER_COLUMNS,
        dataset_name="Users",
    )

    renamed["user_id"] = _valid_user_ids(renamed["user_id"], dataset_name="Users")
    renamed["age"] = pd.to_numeric(renamed["age"], errors="coerce")
    renamed["location"] = renamed["location"].astype("string").str.strip()
    return renamed[["user_id", "location", "age"]].drop_duplicates("user_id").reset_index(drop=True)


def filter_sparse_interactions(
    ratings: pd.DataFrame,
    min_user_ratings: int = 1,
    min_book_ratings: int = 1,
    max_rounds: int | None = None,
) -> pd.DataFrame:
    """Compute the bipartite user/book k-core to convergence.

    Peeling one sparse user or book can make a neighbor sparse, so a fixed number of
    passes is not sufficient. ``max_rounds`` is retained as an optional safety guard for
    callers, but reaching it raises instead of returning data that violates the contract.
    """
    min_user_ratings = int(
        _validate_integer_option("min_user_ratings", min_user_ratings, minimum=1)
    )
    min_book_ratings = int(
        _validate_integer_option("min_book_ratings", min_book_ratings, minimum=1)
    )
    max_rounds = _validate_integer_option(
        "max_rounds",
        max_rounds,
        minimum=1,
        allow_none=True,
    )

    # Cleaning here makes the public function safe when called independently and prevents
    # duplicate user/ISBN rows from inflating degree counts.
    filtered = clean_ratings(ratings)
    rounds = 0
    while not filtered.empty:
        user_counts = filtered.groupby("user_id")["isbn"].transform("size")
        book_counts = filtered.groupby("isbn")["user_id"].transform("size")
        keep = (user_counts >= min_user_ratings) & (book_counts >= min_book_ratings)
        if bool(keep.all()):
            break
        if max_rounds is not None and rounds >= max_rounds:
            raise RuntimeError(
                "Sparse-interaction filtering did not converge within "
                f"max_rounds={max_rounds}. Increase the guard or leave it as None."
            )
        filtered = filtered.loc[keep].reset_index(drop=True)
        rounds += 1

    return filtered.reset_index(drop=True)


def select_most_rated_books(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    max_books: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_books = _validate_integer_option("max_books", max_books, minimum=1, allow_none=True)
    books = _renamed_with_required_columns(
        books,
        rename_map={},
        required=("isbn",),
        dataset_name="Books",
    )
    ratings = _renamed_with_required_columns(
        ratings,
        rename_map={},
        required=("isbn",),
        dataset_name="Ratings",
    )
    books["isbn"] = _normalized_isbns(books["isbn"], dataset_name="Books")
    ratings["isbn"] = _normalized_isbns(ratings["isbn"], dataset_name="Ratings")
    valid_isbns = set(books["isbn"])
    ratings = ratings[ratings["isbn"].isin(valid_isbns)].reset_index(drop=True)

    if max_books is None or books["isbn"].nunique() <= max_books:
        return books.reset_index(drop=True), ratings.reset_index(drop=True)

    # ISBN is the stable tie-breaker when books have equal interaction counts.
    ranked = (
        ratings.groupby("isbn", as_index=False, sort=False)
        .size()
        .sort_values(["size", "isbn"], ascending=[False, True], kind="mergesort")
    )
    top_isbns = ranked.head(max_books)["isbn"]
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
    (
        min_user_ratings,
        min_book_ratings,
        max_books,
        max_ratings,
        random_state,
    ) = _validate_prepare_options(
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
        max_books=max_books,
        max_ratings=max_ratings,
        random_state=random_state,
    )
    dataset = load_clean_dataset(data_dir)
    return prepare_interaction_dataset(
        books=dataset.books,
        ratings=dataset.ratings,
        users=dataset.users,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
        max_books=max_books,
        max_ratings=max_ratings,
        random_state=random_state,
    )


def prepare_hybrid_dataset(
    data_dir: str | Path = "data/raw",
    min_user_ratings: int = 1,
    min_book_ratings: int = 1,
    max_books: int | None = None,
    max_ratings: int | None = None,
    max_content_books: int | None = None,
    random_state: int = 42,
) -> BookDataset:
    """Prepare an interaction core while retaining a separate cold-item content catalog."""

    clean = load_clean_dataset(data_dir)
    interaction = prepare_interaction_dataset(
        books=clean.books,
        ratings=clean.ratings,
        users=clean.users,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
        max_books=max_books,
        max_ratings=max_ratings,
        random_state=random_state,
    )
    content_books = select_content_catalog(
        clean.books,
        required_books=interaction.books,
        max_content_books=max_content_books,
    )
    return BookDataset(
        books=content_books,
        ratings=interaction.ratings,
        users=clean.users,
    )


def select_content_catalog(
    books: pd.DataFrame,
    *,
    required_books: pd.DataFrame,
    max_content_books: int | None = None,
) -> pd.DataFrame:
    """Keep all interaction items and deterministically add cold metadata items."""

    books = clean_books(books)
    required_books = clean_books(required_books)
    max_content_books = _validate_integer_option(
        "max_content_books",
        max_content_books,
        minimum=1,
        allow_none=True,
    )
    required_isbns = set(required_books["isbn"])
    missing_required = required_isbns.difference(set(books["isbn"]))
    if missing_required:
        preview = ", ".join(sorted(missing_required)[:5])
        raise ValueError(f"Required content ISBNs are missing from books: {preview}")
    if max_content_books is not None and len(required_isbns) > max_content_books:
        raise ValueError(
            "max_content_books cannot be smaller than the interaction catalog "
            f"({len(required_isbns)} books)."
        )
    if max_content_books is None or len(books) <= max_content_books:
        return books.reset_index(drop=True)

    required = books[books["isbn"].isin(required_isbns)]
    remaining = books[~books["isbn"].isin(required_isbns)].sort_values(
        "isbn",
        kind="stable",
    )
    selected = pd.concat(
        [required, remaining.head(max_content_books - len(required))],
        ignore_index=True,
    )
    return selected.drop_duplicates("isbn").reset_index(drop=True)


def prepare_interaction_dataset(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    min_user_ratings: int = 1,
    min_book_ratings: int = 1,
    max_books: int | None = None,
    max_ratings: int | None = None,
    random_state: int = 42,
) -> BookDataset:
    (
        min_user_ratings,
        min_book_ratings,
        max_books,
        max_ratings,
        random_state,
    ) = _validate_prepare_options(
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
        max_books=max_books,
        max_ratings=max_ratings,
        random_state=random_state,
    )

    # This function is public and is also used directly by evaluation code, so it enforces
    # the same normalized schema and row-level contracts as file-based preparation.
    books = clean_books(books)
    ratings = clean_ratings(ratings)
    users = clean_users(users)

    ratings = ratings[ratings["isbn"].isin(books["isbn"])].reset_index(drop=True)
    ratings = filter_sparse_interactions(
        ratings,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
    )
    books = books[books["isbn"].isin(ratings["isbn"].unique())].reset_index(drop=True)

    books, ratings = select_most_rated_books(books, ratings, max_books=max_books)
    ratings = filter_sparse_interactions(
        ratings,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
    )
    books = books[books["isbn"].isin(ratings["isbn"].unique())].reset_index(drop=True)

    if max_ratings is not None and len(ratings) > max_ratings:
        ratings = (
            ratings.sample(n=max_ratings, random_state=random_state)
            .sort_index(kind="stable")
            .reset_index(drop=True)
        )

    # Sampling can create a new cascade of sparse users/books. Always peel again so the
    # returned ratings satisfy the requested thresholds, even when max_ratings was applied.
    ratings = filter_sparse_interactions(
        ratings,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
    )
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
        "rating_users": int(dataset.ratings["user_id"].nunique())
        if not dataset.ratings.empty
        else 0,
    }
