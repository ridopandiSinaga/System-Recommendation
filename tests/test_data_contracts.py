import pandas as pd
import pytest

from book_recommender.data import (
    clean_books,
    clean_ratings,
    clean_users,
    filter_sparse_interactions,
    prepare_interaction_dataset,
    select_content_catalog,
    select_most_rated_books,
)


def _books(*isbns: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ISBN": isbn,
                "Book-Title": f"Book {isbn}",
                "Book-Author": "Author",
                "Publisher": "Publisher",
            }
            for isbn in isbns
        ]
    )


def _users(*user_ids: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{"User-ID": user_id, "Location": "somewhere", "Age": 30} for user_id in user_ids]
    )


def _ratings(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"User-ID": user_id, "ISBN": isbn, "Book-Rating": rating}
            for user_id, isbn, rating in rows
        ]
    )


@pytest.mark.parametrize(
    ("cleaner", "frame", "missing_column"),
    [
        (
            clean_books,
            pd.DataFrame({"ISBN": ["A"], "Book-Title": ["A"], "Book-Author": ["Author"]}),
            "publisher",
        ),
        (clean_ratings, pd.DataFrame({"User-ID": [1], "ISBN": ["A"]}), "book_rating"),
        (clean_users, pd.DataFrame({"User-ID": [1], "Location": ["x"]}), "age"),
    ],
)
def test_cleaners_reject_missing_core_columns(cleaner, frame, missing_column):
    with pytest.raises(ValueError, match=missing_column):
        cleaner(frame)


def test_cleaners_reject_invalid_identifiers_and_ratings():
    with pytest.raises(ValueError, match="ISBN"):
        clean_books(_books(" "))

    with pytest.raises(ValueError, match="positive integers"):
        clean_users(pd.DataFrame({"User-ID": [1.5], "Location": ["x"], "Age": [30]}))

    with pytest.raises(ValueError, match="0 through 10"):
        clean_ratings(_ratings([(1, "A", 11)]))

    with pytest.raises(ValueError, match="ISBN"):
        clean_ratings(_ratings([(1, "", 8)]))


def test_clean_ratings_drops_implicit_zero_and_averages_duplicate_pair():
    source = _ratings([(1, "A", 2), (1, "A", 10), (1, "B", 0), (2, "A", 8)])

    cleaned = clean_ratings(source)
    reversed_cleaned = clean_ratings(source.iloc[::-1])

    expected = pd.DataFrame(
        {
            "user_id": pd.Series([1, 2], dtype="int64"),
            "isbn": pd.Series(["A", "A"], dtype="string"),
            "book_rating": pd.Series([6.0, 8.0], dtype="float32"),
        }
    )
    pd.testing.assert_frame_equal(
        cleaned.sort_values(["user_id", "isbn"]).reset_index(drop=True),
        expected,
    )
    pd.testing.assert_frame_equal(
        reversed_cleaned.sort_values(["user_id", "isbn"]).reset_index(drop=True),
        expected,
    )


def test_k_core_filter_converges_through_long_cascade_and_preserves_core():
    # Users 1 and 2 form a stable 2x2 core. The chain beginning at user 3 peels
    # backward one side at a time and requires more than the old five-pass limit.
    rows = [(1, "A", 8), (1, "B", 8), (2, "A", 8), (2, "B", 8)]
    previous_book = "B"
    for user_id, next_book in zip(range(3, 12), "CDEFGHIJK"):
        rows.extend([(user_id, previous_book, 8), (user_id, next_book, 8)])
        previous_book = next_book

    filtered = filter_sparse_interactions(
        clean_ratings(_ratings(rows)),
        min_user_ratings=2,
        min_book_ratings=2,
    )

    assert set(filtered["user_id"]) == {1, 2}
    assert set(filtered["isbn"]) == {"A", "B"}
    assert len(filtered) == 4
    assert filtered["user_id"].value_counts().ge(2).all()
    assert filtered["isbn"].value_counts().ge(2).all()


def test_prepare_reapplies_k_core_after_max_books():
    dataset = prepare_interaction_dataset(
        books=_books("A", "B", "C"),
        ratings=_ratings(
            [
                (1, "A", 8),
                (1, "B", 8),
                (2, "A", 8),
                (2, "B", 8),
                (3, "A", 8),
                (3, "C", 8),
                (4, "B", 8),
                (4, "C", 8),
            ]
        ),
        users=_users(1, 2, 3, 4),
        min_user_ratings=2,
        min_book_ratings=2,
        max_books=2,
    )

    assert set(dataset.ratings["user_id"]) == {1, 2}
    assert set(dataset.ratings["isbn"]) == {"A", "B"}
    assert set(dataset.books["isbn"]) == {"A", "B"}
    assert dataset.ratings["user_id"].value_counts().ge(2).all()
    assert dataset.ratings["isbn"].value_counts().ge(2).all()


def test_prepare_reapplies_k_core_after_max_ratings_sampling():
    rows = [(user_id, isbn, 8) for user_id in range(1, 5) for isbn in "ABCD"]
    dataset = prepare_interaction_dataset(
        books=_books(*"ABCD"),
        ratings=_ratings(rows),
        users=_users(1, 2, 3, 4),
        min_user_ratings=2,
        min_book_ratings=2,
        max_ratings=10,
        random_state=0,
    )

    assert not dataset.ratings.empty
    assert len(dataset.ratings) <= 10
    assert dataset.ratings["user_id"].value_counts().ge(2).all()
    assert dataset.ratings["isbn"].value_counts().ge(2).all()
    assert set(dataset.books["isbn"]) == set(dataset.ratings["isbn"])


def test_public_numeric_options_are_validated():
    cleaned_ratings = clean_ratings(_ratings([(1, "A", 8)]))
    cleaned_books = clean_books(_books("A"))

    with pytest.raises(ValueError, match="min_user_ratings"):
        filter_sparse_interactions(cleaned_ratings, min_user_ratings=0)
    with pytest.raises(ValueError, match="max_rounds"):
        filter_sparse_interactions(cleaned_ratings, max_rounds=0)
    with pytest.raises(ValueError, match="max_books"):
        select_most_rated_books(cleaned_books, cleaned_ratings, max_books=0)
    with pytest.raises(ValueError, match="max_ratings"):
        prepare_interaction_dataset(
            cleaned_books,
            cleaned_ratings,
            clean_users(_users(1)),
            max_ratings=0,
        )
    with pytest.raises(ValueError, match="random_state"):
        prepare_interaction_dataset(
            cleaned_books,
            cleaned_ratings,
            clean_users(_users(1)),
            random_state=-1,
        )


def test_select_most_rated_books_never_returns_ratings_outside_catalog():
    books = clean_books(_books("A"))
    ratings = clean_ratings(_ratings([(1, "B", 9), (2, "B", 8)]))

    selected_books, selected_ratings = select_most_rated_books(
        books,
        ratings,
        max_books=None,
    )

    assert set(selected_books["isbn"]) == {"A"}
    assert selected_ratings.empty


def test_content_catalog_retains_interaction_items_and_adds_cold_items():
    books = clean_books(_books("A", "B", "C", "D"))
    required = clean_books(_books("C", "D"))

    selected = select_content_catalog(
        books,
        required_books=required,
        max_content_books=3,
    )

    assert {"C", "D"}.issubset(set(selected["isbn"]))
    assert len(selected) == 3

    with pytest.raises(ValueError, match="missing from books"):
        select_content_catalog(
            books,
            required_books=clean_books(_books("missing")),
            max_content_books=3,
        )
