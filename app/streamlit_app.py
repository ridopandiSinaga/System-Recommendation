from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender
from book_recommender.data import DatasetNotFoundError, dataset_summary, prepare_dataset, validate_data_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_MAX_BOOKS = 5_000


st.set_page_config(
    page_title="Book Recommender",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("Book Recommender")

    with st.sidebar:
        st.header("Dataset")
        max_books = st.slider(
            "Books",
            min_value=1_000,
            max_value=20_000,
            value=DEFAULT_MAX_BOOKS,
            step=1_000,
        )
        min_user_ratings = st.slider("Min user ratings", 1, 10, 2)
        min_book_ratings = st.slider("Min book ratings", 1, 10, 2)
        top_n = st.slider("Recommendations", 3, 20, 8)

    try:
        validate_data_dir(DATA_DIR)
    except DatasetNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    dataset = load_dataset(max_books, min_user_ratings, min_book_ratings)
    content_model = load_content_model(max_books, min_user_ratings, min_book_ratings)
    collab_model = load_collab_model(max_books, min_user_ratings, min_book_ratings)

    render_summary(dataset)

    content_tab, collab_tab = st.tabs(["Similar books", "User recommendations"])

    with content_tab:
        render_content_tab(dataset.books, content_model, top_n)

    with collab_tab:
        render_collab_tab(dataset.books, dataset.ratings, collab_model, top_n)


@st.cache_data(show_spinner="Loading dataset")
def load_dataset(max_books: int, min_user_ratings: int, min_book_ratings: int):
    return prepare_dataset(
        DATA_DIR,
        min_user_ratings=min_user_ratings,
        min_book_ratings=min_book_ratings,
        max_books=max_books,
        random_state=42,
    )


@st.cache_resource(show_spinner="Building content model")
def load_content_model(max_books: int, min_user_ratings: int, min_book_ratings: int):
    dataset = load_dataset(max_books, min_user_ratings, min_book_ratings)
    return ContentBasedRecommender().fit(dataset.books)


@st.cache_resource(show_spinner="Building collaborative model")
def load_collab_model(max_books: int, min_user_ratings: int, min_book_ratings: int):
    dataset = load_dataset(max_books, min_user_ratings, min_book_ratings)
    return ItemBasedCollaborativeRecommender(n_neighbors=50).fit(dataset.books, dataset.ratings)


def render_summary(dataset) -> None:
    summary = dataset_summary(dataset)
    cols = st.columns(4)
    cols[0].metric("Books", f"{summary['books']:,}")
    cols[1].metric("Ratings", f"{summary['ratings']:,}")
    cols[2].metric("Rating users", f"{summary['rating_users']:,}")
    cols[3].metric("Rated books", f"{summary['rated_books']:,}")


def render_content_tab(
    books: pd.DataFrame,
    content_model: ContentBasedRecommender,
    top_n: int,
) -> None:
    book_options = books.sort_values("book_title")
    titles = book_options["book_title"].drop_duplicates().tolist()
    default_index = find_default_title_index(titles, "Adventures of Huckleberry Finn")

    selected_title = st.selectbox("Book title", titles, index=default_index)
    query_book = book_options[book_options["book_title"] == selected_title].iloc[0]

    selected_cols = st.columns([1, 4])
    render_cover(selected_cols[0], query_book)
    with selected_cols[1]:
        st.subheader(query_book["book_title"])
        st.caption(f"{query_book['book_author']} · {query_book['publisher']}")
        st.code(str(query_book["isbn"]), language=None)

    recommendations = content_model.recommend_similar(title=selected_title, top_n=top_n)
    render_recommendation_grid(recommendations)


def render_collab_tab(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    collab_model: ItemBasedCollaborativeRecommender,
    top_n: int,
) -> None:
    active_users = ratings["user_id"].value_counts()
    default_user = int(active_users.index[0])
    user_id = st.number_input("User ID", min_value=1, value=default_user, step=1)

    user_history = (
        ratings[ratings["user_id"] == int(user_id)]
        .sort_values("book_rating", ascending=False)
        .merge(books, on="isbn", how="left")
    )

    if user_history.empty:
        st.warning("User history is not available in the current sample.")
    else:
        st.subheader("Highest rated books")
        render_history(user_history.head(5))

    recommendations = collab_model.recommend_for_user(int(user_id), top_n=top_n)
    st.subheader("Recommendations")
    render_recommendation_grid(recommendations)


def render_history(history: pd.DataFrame) -> None:
    cols = st.columns(min(5, len(history)))
    for col, row in zip(cols, history.itertuples(index=False)):
        with col:
            render_cover(st, row)
            st.markdown(f"**{row.book_title}**")
            st.caption(f"{row.book_author}")
            st.write(f"Rating: {float(row.book_rating):.0f}/10")


def render_recommendation_grid(recommendations: pd.DataFrame) -> None:
    if recommendations.empty:
        st.info("No recommendations found.")
        return

    for start in range(0, len(recommendations), 4):
        cols = st.columns(4)
        for col, row in zip(cols, recommendations.iloc[start : start + 4].itertuples(index=False)):
            with col:
                render_cover(st, row)
                st.markdown(f"**{row.book_title}**")
                st.caption(f"{row.book_author}")
                if hasattr(row, "score"):
                    st.write(f"Score: {float(row.score):.3f}")
                st.caption(str(row.publisher))


def render_cover(container, row) -> None:
    image_url = value_from_row(row, "image_m_url")
    title = value_from_row(row, "book_title") or "Book cover"
    if image_url and isinstance(image_url, str) and image_url.startswith(("http://", "https://")):
        container.image(image_url, use_container_width=True)
    else:
        container.info(title)


def find_default_title_index(titles: list[str], default_title: str) -> int:
    try:
        return titles.index(default_title)
    except ValueError:
        return 0


def value_from_row(row, key: str):
    if isinstance(row, pd.Series):
        return row.get(key)
    return getattr(row, key, None)


if __name__ == "__main__":
    main()
