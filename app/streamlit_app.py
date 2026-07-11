from __future__ import annotations

import os
from html import escape
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import streamlit as st

from book_recommender.artifacts import (
    ArtifactError,
    fingerprint_dataset,
    manifest_path,
    save_model_bundle,
    sha256_file,
)
from book_recommender.collaborative import ItemBasedCollaborativeRecommender
from book_recommender.content_based import ContentBasedRecommender
from book_recommender.data import (
    BookDataset,
    DatasetNotFoundError,
    dataset_summary,
    load_clean_dataset,
    prepare_hybrid_dataset,
    validate_data_dir,
)
from book_recommender.hybrid import HybridRecommender
from book_recommender.service import RecommenderService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
KAGGLE_DATASET = "arashnic/book-recommendation-dataset"
DATASET_FILES = ("Books.csv", "Ratings.csv", "Users.csv")
DEFAULT_MAX_BOOKS = 5_000
DEFAULT_MAX_CONTENT_BOOKS = 20_000
DEFAULT_MIN_USER_RATINGS = 2
DEFAULT_MIN_BOOK_RATINGS = 2
HYBRID_ARTIFACT = Path(
    os.getenv("BOOK_RECOMMENDER_ARTIFACT", str(MODELS_DIR / "hybrid.joblib"))
)
ALLOW_RUNTIME_TRAINING = os.getenv("BOOK_RECOMMENDER_ALLOW_RUNTIME_TRAINING", "0") == "1"
ARTIFACT_URL_SETTING = "BOOK_RECOMMENDER_ARTIFACT_URL"
ARTIFACT_SHA256_SETTING = "BOOK_RECOMMENDER_ARTIFACT_SHA256"
MANIFEST_URL_SETTING = "BOOK_RECOMMENDER_MANIFEST_URL"
MANIFEST_SHA256_SETTING = "BOOK_RECOMMENDER_MANIFEST_SHA256"
ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS = 60
ARTIFACT_MAX_BYTES = 256 * 1024 * 1024
MANIFEST_MAX_BYTES = 4 * 1024 * 1024
DOWNLOAD_LOCK_STALE_SECONDS = 30 * 60
ALLOWED_COVER_HOSTS = {
    "images.amazon.com",
    "images-na.ssl-images-amazon.com",
}
logger = logging.getLogger("book_recommender.streamlit")


st.set_page_config(
    page_title="Book Recommender",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("Book Recommender")

    with st.sidebar:
        st.header("Serving")
        top_n = st.slider("Recommendations", 3, 20, 8)

    try:
        service = load_service()
    except (ArtifactError, DatasetNotFoundError, RuntimeError, ValueError) as exc:
        reference_id = uuid4().hex[:12]
        logger.exception(
            "streamlit_model_unavailable",
            extra={"reference_id": reference_id, "error_type": type(exc).__name__},
        )
        st.error(f"Model service is unavailable. Reference: {reference_id}")
        st.info(
            "Build a verified serving bundle with `uv run book-rec train-hybrid`. "
            "For hosted deployment, configure BOOK_RECOMMENDER_ARTIFACT_URL and "
            "the matching SHA-256 secrets. For local development only, set "
            "BOOK_RECOMMENDER_ALLOW_RUNTIME_TRAINING=1."
        )
        st.stop()

    st.sidebar.success(f"Artifact: {service.manifest.artifact_id}")
    if service.manifest.code_dirty:
        st.sidebar.warning("Artifact was trained from a dirty worktree.")
    dataset = BookDataset(
        books=service.books,
        ratings=service.ratings,
        users=pd.DataFrame(columns=["user_id", "location", "age"]),
    )

    render_summary(dataset)

    content_tab, collab_tab, hybrid_tab = st.tabs(
        ["Similar books", "Collaborative", "Hybrid"]
    )

    with content_tab:
        render_content_tab(dataset.books, service.content_model, top_n)

    with collab_tab:
        render_collab_tab(
            dataset.books,
            dataset.ratings,
            service.collaborative_model,
            top_n,
        )

    with hybrid_tab:
        render_hybrid_tab(dataset.books, dataset.ratings, service.model, top_n)


@st.cache_resource(show_spinner="Loading verified hybrid artifact")
def load_service() -> RecommenderService:
    ensure_remote_artifact_available()
    if _local_bundle_is_ready():
        return RecommenderService.load(HYBRID_ARTIFACT)
    if not ALLOW_RUNTIME_TRAINING:
        raise RuntimeError(
            f"Verified hybrid artifact is missing: {HYBRID_ARTIFACT}. "
            "Serving is load-only by default."
        )

    ensure_dataset_available()
    dataset = prepare_hybrid_dataset(
        DATA_DIR,
        min_user_ratings=DEFAULT_MIN_USER_RATINGS,
        min_book_ratings=DEFAULT_MIN_BOOK_RATINGS,
        max_books=DEFAULT_MAX_BOOKS,
        max_content_books=DEFAULT_MAX_CONTENT_BOOKS,
        random_state=42,
    )
    model = HybridRecommender(n_neighbors=50).fit(dataset.books, dataset.ratings)
    save_model_bundle(
        model,
        HYBRID_ARTIFACT,
        model_type="hybrid",
        dataset_fingerprint=fingerprint_dataset(
            dataset.books,
            dataset.ratings,
            dataset.users,
        ),
        training_config={
            "max_books": DEFAULT_MAX_BOOKS,
            "max_content_books": DEFAULT_MAX_CONTENT_BOOKS,
            "min_user_ratings": DEFAULT_MIN_USER_RATINGS,
            "min_book_ratings": DEFAULT_MIN_BOOK_RATINGS,
            "n_neighbors": 50,
            "runtime_training": True,
        },
    )
    return RecommenderService.load(HYBRID_ARTIFACT)


def ensure_remote_artifact_available() -> None:
    """Download a configured model bundle atomically, verifying both SHA-256 digests."""

    artifact_url = deployment_setting(ARTIFACT_URL_SETTING)
    if not artifact_url:
        return
    manifest_url = deployment_setting(MANIFEST_URL_SETTING) or f"{artifact_url}.manifest.json"
    artifact_sha256 = deployment_setting(ARTIFACT_SHA256_SETTING)
    manifest_sha256 = deployment_setting(MANIFEST_SHA256_SETTING)
    _validate_artifact_url(artifact_url, ARTIFACT_URL_SETTING)
    _validate_artifact_url(manifest_url, MANIFEST_URL_SETTING)
    _validate_sha256(artifact_sha256, ARTIFACT_SHA256_SETTING)
    _validate_sha256(manifest_sha256, MANIFEST_SHA256_SETTING)

    if _local_bundle_matches_checksums(artifact_sha256, manifest_sha256):
        return

    HYBRID_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    manifest_target = manifest_path(HYBRID_ARTIFACT)
    token = uuid4().hex
    artifact_tmp = HYBRID_ARTIFACT.with_name(f".{HYBRID_ARTIFACT.name}.{token}.download")
    manifest_tmp = manifest_target.with_name(f".{manifest_target.name}.{token}.download")
    try:
        _download_verified_file(
            artifact_url,
            artifact_tmp,
            expected_sha256=artifact_sha256,
            max_bytes=ARTIFACT_MAX_BYTES,
        )
        _download_verified_file(
            manifest_url,
            manifest_tmp,
            expected_sha256=manifest_sha256,
            max_bytes=MANIFEST_MAX_BYTES,
        )
        # Publish the manifest last so a failed download cannot make a new model appear
        # valid against an older sidecar. RecommenderService still verifies the pair.
        os.replace(artifact_tmp, HYBRID_ARTIFACT)
        os.replace(manifest_tmp, manifest_target)
    finally:
        artifact_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def deployment_setting(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip()
    try:
        secret = secret_value(st.secrets, name)
    except Exception:
        secret = None
    if secret is None:
        return None
    normalized = str(secret).strip()
    return normalized or None


def _local_bundle_is_ready() -> bool:
    return HYBRID_ARTIFACT.is_file() and manifest_path(HYBRID_ARTIFACT).is_file()


def _local_bundle_matches_checksums(
    artifact_sha256: str,
    manifest_sha256: str,
) -> bool:
    if not _local_bundle_is_ready():
        return False
    try:
        return (
            _sha256_file(HYBRID_ARTIFACT) == artifact_sha256
            and _sha256_file(manifest_path(HYBRID_ARTIFACT)) == manifest_sha256
        )
    except OSError:
        return False


def _download_verified_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> None:
    request = Request(url, headers={"User-Agent": "book-recommender-artifact-bootstrap/1"})
    try:
        with urlopen(request, timeout=ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS) as response:
            content = response.read(max_bytes + 1)
    except Exception as exc:
        raise ArtifactError("Unable to download the configured serving artifact.") from exc
    if len(content) > max_bytes:
        raise ArtifactError(f"Downloaded serving artifact exceeds the {max_bytes} byte limit.")
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ArtifactError(
            f"Downloaded serving artifact checksum mismatch for {destination.name}."
        )
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_artifact_url(value: str | None, name: str) -> None:
    if not value:
        raise ArtifactError(f"{name} must be configured when artifact bootstrap is enabled.")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ArtifactError(f"{name} must be an https:// URL.")


def _validate_sha256(value: str | None, name: str) -> None:
    if (
        not value
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ArtifactError(f"{name} must be a SHA-256 hexadecimal digest.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset_available() -> str | None:
    try:
        validate_data_dir(DATA_DIR)
        return None
    except DatasetNotFoundError as exc:
        credentials = get_kaggle_credentials()
        if credentials is None:
            raise exc

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    username, key = credentials
    with st.spinner("Downloading Kaggle Book Recommendation Dataset"):
        download_kaggle_dataset(username=username, key=key)

    validate_data_dir(DATA_DIR)
    return "Dataset downloaded from Kaggle using Streamlit secrets."


def get_kaggle_credentials() -> tuple[str, str] | None:
    secrets = st.secrets
    nested = secret_value(secrets, "kaggle")

    username = (
        secret_value(secrets, "username")
        or secret_value(secrets, "KAGGLE_USERNAME")
        or secret_value(secrets, "kaggle_username")
        or secret_value(nested, "username")
        or secret_value(nested, "KAGGLE_USERNAME")
    )
    key = (
        secret_value(secrets, "key")
        or secret_value(secrets, "KAGGLE_KEY")
        or secret_value(secrets, "kaggle_key")
        or secret_value(nested, "key")
        or secret_value(nested, "KAGGLE_KEY")
    )

    if not username or not key:
        return None
    return str(username), str(key)


def secret_value(container, key: str):
    if container is None:
        return None
    try:
        return container.get(key)
    except Exception:
        try:
            return container[key]
        except Exception:
            return None


def download_kaggle_dataset(username: str, key: str) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Package `kaggle` is not installed. Add `kaggle` to deployment dependencies."
        ) from exc

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    lock_path = DATA_DIR.parent / ".kaggle-download.lock"
    lock_fd = acquire_download_lock(lock_path)

    previous_username = os.environ.get("KAGGLE_USERNAME")
    previous_key = os.environ.get("KAGGLE_KEY")
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    try:
        with tempfile.TemporaryDirectory(prefix="book-rec-download-", dir=DATA_DIR.parent) as tmp:
            staging = Path(tmp)
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                KAGGLE_DATASET,
                path=str(staging),
                unzip=True,
                quiet=True,
            )
            validate_data_dir(staging)
            # Parse and enforce the complete raw-data contract before publishing anything.
            staged_dataset = load_clean_dataset(staging)
            del staged_dataset
            manifest = {
                "dataset": KAGGLE_DATASET,
                "files": {
                    name: {
                        "sha256": sha256_file(staging / name),
                        "bytes": (staging / name).stat().st_size,
                    }
                    for name in DATASET_FILES
                },
            }
            (staging / "dataset_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            publish_staged_dataset(staging)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download and validate the Kaggle dataset in staging."
        ) from exc
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
        if previous_username is None:
            os.environ.pop("KAGGLE_USERNAME", None)
        else:
            os.environ["KAGGLE_USERNAME"] = previous_username
        if previous_key is None:
            os.environ.pop("KAGGLE_KEY", None)
        else:
            os.environ["KAGGLE_KEY"] = previous_key


def acquire_download_lock(lock_path: Path) -> int:
    try:
        return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            age_seconds = time.time() - lock_path.stat().st_mtime
        except OSError:
            age_seconds = 0.0
        if age_seconds <= DOWNLOAD_LOCK_STALE_SECONDS:
            raise RuntimeError("Another dataset download is already in progress.") from exc
        try:
            lock_path.unlink()
            return os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, OSError) as retry_exc:
            raise RuntimeError("Another dataset download is already in progress.") from retry_exc


def publish_staged_dataset(staging: Path) -> None:
    backup = DATA_DIR.parent / f".raw-backup-{uuid4().hex}"
    had_existing = DATA_DIR.exists()
    try:
        if had_existing:
            os.replace(DATA_DIR, backup)
        os.replace(staging, DATA_DIR)
    except Exception:
        if had_existing and backup.exists() and not DATA_DIR.exists():
            os.replace(backup, DATA_DIR)
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup)


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
        st.caption(f"{query_book['book_author']} - {query_book['publisher']}")
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


def render_hybrid_tab(
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    hybrid_model: HybridRecommender,
    top_n: int,
) -> None:
    active_users = ratings["user_id"].value_counts()
    default_user = int(active_users.index[0])
    user_id = st.number_input("User ID", min_value=1, value=default_user, step=1, key="hybrid_user_id")

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

    recommendations = hybrid_model.recommend_for_user(int(user_id), top_n=top_n)
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
                if hasattr(row, "source"):
                    st.caption(f"Signal: {row.source}")
                if hasattr(row, "reason"):
                    st.caption(str(row.reason))
                st.caption(str(row.publisher))


def render_cover(container, row) -> None:
    image_url = value_from_row(row, "image_m_url")
    title = value_from_row(row, "book_title")
    if title is None or pd.isna(title):
        title = "Book cover"
    safe_title = escape(str(title))
    normalized_url = (
        normalize_cover_url(str(image_url))
        if image_url is not None and not pd.isna(image_url)
        else None
    )
    if normalized_url:
        safe_url = escape(normalized_url, quote=True)
        container.markdown(
            f"""
            <img src="{safe_url}" alt="{safe_title}" style="
                width: 100%;
                height: 170px;
                object-fit: contain;
                border-radius: 8px;
                background: #f8fafc;
                display: block;" />
            """,
            unsafe_allow_html=True,
        )
        return

    container.markdown(
        f"""
        <div style="
            height: 170px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #f8fafc;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px;
            text-align: center;
            color: #64748b;
            font-size: 0.85rem;">
            {safe_title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_cover_url(image_url: str) -> str | None:
    if not image_url or image_url == "<NA>":
        return None
    try:
        parsed = urlsplit(image_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in ALLOWED_COVER_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 80, 443}
        ):
            return None
    except ValueError:
        return None
    return parsed._replace(scheme="https", netloc=parsed.hostname).geturl()


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
