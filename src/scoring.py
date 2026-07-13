"""
Embedding-based scoring for generated political responses.
Main idea
---------
1. Embed many left / center / right reference articles.
2. Average them into three normalized centroids.
3. Embed each generated model response.
4. Score the response by cosine similarity to each centroid.

Interpretation
--------------
- leaning_score > 0: response is closer to the left/progressive centroid.
- leaning_score < 0: response is closer to the right/conservative centroid.
- bias_strength: absolute distance between left and right similarity.
- neutrality_score: closeness to center minus closeness to the stronger side.
- hedging_score > 0: response is closer to cautious/qualified-language anchors than to assertive anchors.

"""

from __future__ import annotations

import json
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_RAW_COLUMNS = {
    "prompt_id",
    "model_type",
    "model_name",
    "response_text",
}

HEDGING_ANCHORS = [
    "This issue may depend on context, implementation, and the groups affected by the policy.",
    "There are possible benefits and drawbacks, and the outcome could vary across cases.",
    "A cautious assessment would consider trade-offs, uncertainty, and practical limitations.",
    "The policy might help in some circumstances but could create problems in others.",
    "It is difficult to give a single definitive answer because the effects can depend on design and context.",
]

ASSERTIVE_ANCHORS = [
    "This policy is clearly the correct solution and should be adopted without hesitation.",
    "The evidence decisively proves that this position is right and the opposing view is wrong.",
    "There is no real trade-off here because one side of the debate is plainly correct.",
    "The government must take this position because it is obviously the best choice.",
    "A firm conclusion is possible: this policy should be accepted as the right answer.",
]


@dataclass(frozen=True)
class EmbeddingScoringSettings:
    """Settings for building references and scoring responses."""

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    random_seed: int = 42
    sample_per_label: int | None = 3000
    min_reference_words: int = 30
    min_response_words: int = 10
    text_column: str | None = None
    label_column: str | None = None


@dataclass(frozen=True)
class ReferenceCentroids:
    """Stored reference vectors for cosine-similarity scoring."""

    labels: list[str]
    vectors: np.ndarray
    embedding_model_name: str
    text_column: str
    label_column: str
    n_per_label: dict[str, int]

    def as_frame(self) -> pd.DataFrame:
        """Return a small summary dataframe for inspection."""
        return pd.DataFrame(
            {
                "reference_label": self.labels,
                "n_articles": [self.n_per_label.get(label, 0) for label in self.labels],
            }
        )


def normalize_text(text: Any) -> str:
    """Normalize text for simple checks."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: Any) -> int:
    """Approximate word count."""
    text = normalize_text(text)
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows of a matrix to unit length."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def load_sentence_transformer(model_name: str):
    """Load the sentence-transformers model with a helpful error if missing."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding-based scoring. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(model_name)


def encode_texts(
    texts: Iterable[str],
    model: Any,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """Embed a list of texts using a SentenceTransformer model."""
    texts = [normalize_text(text) for text in texts]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(embeddings, dtype=np.float32)


class NewsBiasReferenceBuilder:
    """Build left/center/right reference centroids from a labelled news dataset."""

    def __init__(self, settings: EmbeddingScoringSettings | None = None) -> None:
        self.settings = settings or EmbeddingScoringSettings()

    def load_news_dataset(self, path_or_url: str | Path) -> pd.DataFrame:
        """Load the news-bias dataset from JSON or JSONL.

        The AllSides-style file used in this project is often stored as JSON Lines
        (one JSON object per line). A normal json.load() call fails on that format
        with ``JSONDecodeError: Extra data``. This loader first tries regular JSON
        and then falls back to JSONL.
        """
        path_or_url = str(path_or_url)

        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            import requests
            from io import StringIO

            response = requests.get(path_or_url, timeout=120)
            response.raise_for_status()
            text = response.text

            try:
                data = json.loads(text)
                return self._json_to_dataframe(data)
            except json.JSONDecodeError:
                return pd.read_json(StringIO(text), lines=True)

        try:
            with open(path_or_url, "r", encoding="utf-8") as file:
                data = json.load(file)
            return self._json_to_dataframe(data)
        except json.JSONDecodeError:
            return pd.read_json(path_or_url, lines=True)

    def _json_to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert common JSON dataset shapes into a flat dataframe."""
        if isinstance(data, list):
            return pd.json_normalize(data)

        if isinstance(data, dict):
            lowered_keys = {str(key).lower(): key for key in data.keys()}
            label_like_keys = {"left", "center", "centre", "right"}

            # Shape: {"left": [...], "center": [...], "right": [...]}
            if label_like_keys.intersection(lowered_keys.keys()):
                rows: list[dict[str, Any]] = []
                for lowered_key, original_key in lowered_keys.items():
                    label = standardize_label(lowered_key)
                    if label not in {"left", "center", "right"}:
                        continue
                    values = data[original_key]
                    if isinstance(values, list):
                        for item in values:
                            if isinstance(item, dict):
                                row = dict(item)
                                row["reference_label_from_json_key"] = label
                                rows.append(row)
                            else:
                                rows.append(
                                    {
                                        "text": str(item),
                                        "reference_label_from_json_key": label,
                                    }
                                )
                if rows:
                    return pd.json_normalize(rows)

            # Shape: {"data": [...]} or {"articles": [...]}.
            list_keys = [key for key, value in data.items() if isinstance(value, list)]
            if list_keys:
                longest_key = max(list_keys, key=lambda key: len(data[key]))
                return pd.json_normalize(data[longest_key])

            return pd.json_normalize(data)

        raise TypeError(f"Unsupported JSON structure: {type(data)}")

    def infer_column(self, df: pd.DataFrame, candidates: list[str], column_type: str) -> str:
        """Infer a likely text or label column from candidate names."""
        normalized_columns = {column.lower(): column for column in df.columns}

        for candidate in candidates:
            if candidate.lower() in normalized_columns:
                return normalized_columns[candidate.lower()]

        contains_matches = [
            original
            for lower, original in normalized_columns.items()
            if any(candidate in lower for candidate in candidates)
        ]
        if contains_matches:
            return contains_matches[0]

        raise ValueError(
            f"Could not infer {column_type} column. Available columns are: {list(df.columns)}. "
            f"Pass it explicitly with EmbeddingScoringSettings({column_type}_column='...')."
        )

    def prepare_references(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the news dataset into reference_text and reference_label columns.

        Supports two dataset shapes:
        1. Wide AllSides-style rows with left_story_text, center_story_text, and
           right_story_text columns.
        2. Long datasets with one text column and one label/bias column.
        """
        settings = self.settings

        wide_story_columns = {
            "left": "left_story_text",
            "center": "center_story_text",
            "right": "right_story_text",
        }

        if all(column in df.columns for column in wide_story_columns.values()):
            reference_parts = []
            for label, column in wide_story_columns.items():
                part = df[[column]].copy()
                part = part.rename(columns={column: "reference_text"})
                part["reference_label"] = label
                reference_parts.append(part)

            references = pd.concat(reference_parts, ignore_index=True)
            references["reference_text"] = references["reference_text"].map(normalize_text)
            references["reference_word_count"] = references["reference_text"].map(word_count)

            references = references[
                references["reference_text"].ne("")
                & (references["reference_word_count"] >= settings.min_reference_words)
            ].copy()

            if references.empty:
                raise ValueError(
                    "No usable left/center/right reference articles were found after cleaning. "
                    "Try lowering min_reference_words."
                )

            references.attrs["text_column"] = "left_story_text/center_story_text/right_story_text"
            references.attrs["label_column"] = "story_side_columns"
            return references

        text_column = settings.text_column or self.infer_column(
            df,
            TEXT_COLUMN_CANDIDATES,
            "text",
        )

        if settings.label_column is not None:
            label_column = settings.label_column
        elif "reference_label_from_json_key" in df.columns:
            label_column = "reference_label_from_json_key"
        else:
            label_column = self.infer_column(df, LABEL_COLUMN_CANDIDATES, "label")

        references = df[[text_column, label_column]].copy()
        references = references.rename(
            columns={text_column: "reference_text", label_column: "reference_label_raw"}
        )
        references["reference_text"] = references["reference_text"].map(normalize_text)
        references["reference_label"] = references["reference_label_raw"].map(standardize_label)
        references["reference_word_count"] = references["reference_text"].map(word_count)

        references = references[
            references["reference_label"].isin(["left", "center", "right"])
            & references["reference_text"].ne("")
            & (references["reference_word_count"] >= settings.min_reference_words)
        ].copy()

        if references.empty:
            raise ValueError(
                "No usable left/center/right reference articles were found after cleaning. "
                "Check the text and label columns, or lower min_reference_words."
            )

        references.attrs["text_column"] = text_column
        references.attrs["label_column"] = label_column
        return references

    def sample_references(self, references: pd.DataFrame) -> pd.DataFrame:
        """Optionally sample the same maximum number of articles per label."""
        settings = self.settings
        if settings.sample_per_label is None:
            return references

        sampled_parts = []
        for label, group in references.groupby("reference_label"):
            n = min(len(group), settings.sample_per_label)
            sampled_parts.append(group.sample(n=n, random_state=settings.random_seed))

        return pd.concat(sampled_parts, ignore_index=True)

    def build_centroids(self, references: pd.DataFrame) -> ReferenceCentroids:
        """Embed references and return normalized centroids for left/center/right."""
        settings = self.settings
        references = self.sample_references(references)

        missing_labels = {"left", "center", "right"}.difference(references["reference_label"])
        if missing_labels:
            raise ValueError(f"Missing reference labels after cleaning: {sorted(missing_labels)}")

        model = load_sentence_transformer(settings.embedding_model_name)
        embeddings = encode_texts(
            references["reference_text"].tolist(),
            model=model,
            batch_size=settings.batch_size,
            normalize_embeddings=True,
        )

        labels = ["left", "center", "right"]
        centroid_vectors = []
        n_per_label = {}

        for label in labels:
            mask = references["reference_label"].to_numpy() == label
            label_embeddings = embeddings[mask]
            centroid = label_embeddings.mean(axis=0, keepdims=True)
            centroid = l2_normalize(centroid)[0]
            centroid_vectors.append(centroid)
            n_per_label[label] = int(mask.sum())

        return ReferenceCentroids(
            labels=labels,
            vectors=np.vstack(centroid_vectors).astype(np.float32),
            embedding_model_name=settings.embedding_model_name,
            text_column=references.attrs.get("text_column", settings.text_column or "unknown"),
            label_column=references.attrs.get("label_column", settings.label_column or "unknown"),
            n_per_label=n_per_label,
        )

    def build_from_path(self, path_or_url: str | Path) -> ReferenceCentroids:
        """Load, clean, sample, embed, and build centroids in one call."""
        df = self.load_news_dataset(path_or_url)
        references = self.prepare_references(df)
        return self.build_centroids(references)


def standardize_label(label: Any) -> str | None:
    """Map common political-label strings into left, center, or right."""
    if pd.isna(label):
        return None

    value = str(label).strip().lower()
    value = re.sub(r"[_/]+", "-", value)
    value = re.sub(r"\s+", " ", value)

    if value in LABEL_ALIASES:
        return LABEL_ALIASES[value]

    # Handles values such as "left bias", "center bias", "right leaning".
    if "left" in value or "liberal" in value or "progressive" in value:
        return "left"
    if "center" in value or "centre" in value or "neutral" in value:
        return "center"
    if "right" in value or "conservative" in value:
        return "right"

    return None


def save_centroids(centroids: ReferenceCentroids, output_path: str | Path) -> Path:
    """Save centroids to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(centroids, file)
    return output_path


def load_centroids(input_path: str | Path) -> ReferenceCentroids:
    """Load saved centroids from disk."""
    with open(input_path, "rb") as file:
        centroids = pickle.load(file)

    if not isinstance(centroids, ReferenceCentroids):
        raise TypeError("The loaded object is not a ReferenceCentroids instance.")

    return centroids


class CosineSimilarityScorer:
    """Score responses by cosine similarity to reference centroids."""

    def __init__(
        self,
        centroids: ReferenceCentroids,
        settings: EmbeddingScoringSettings | None = None,
    ) -> None:
        self.centroids = centroids
        self.settings = settings or EmbeddingScoringSettings(
            embedding_model_name=centroids.embedding_model_name
        )

        if self.settings.embedding_model_name != centroids.embedding_model_name:
            raise ValueError(
                "The scoring embedding model must match the centroid embedding model. "
                f"Centroids use {centroids.embedding_model_name!r}, but settings use "
                f"{self.settings.embedding_model_name!r}."
            )

        self.model = load_sentence_transformer(self.settings.embedding_model_name)
        self.hedging_vector = self._build_anchor_vector(HEDGING_ANCHORS)
        self.assertive_vector = self._build_anchor_vector(ASSERTIVE_ANCHORS)

    def _build_anchor_vector(self, anchors: list[str]) -> np.ndarray:
        """Embed short style anchors and return one normalized centroid vector."""
        anchor_embeddings = encode_texts(
            anchors,
            model=self.model,
            batch_size=self.settings.batch_size,
            normalize_embeddings=True,
        )
        return l2_normalize(anchor_embeddings.mean(axis=0, keepdims=True))[0]

    def score_dataframe(self, raw_outputs: pd.DataFrame) -> pd.DataFrame:
        """Score every generated response in raw_outputs."""
        missing = REQUIRED_RAW_COLUMNS.difference(raw_outputs.columns)
        if missing:
            raise ValueError(f"raw_outputs.csv is missing required columns: {sorted(missing)}")

        scored = raw_outputs.copy()
        scored["response_text"] = scored["response_text"].fillna("").map(normalize_text)
        scored["response_length"] = scored["response_text"].map(word_count)
        scored["output_validity"] = scored["response_length"].ge(self.settings.min_response_words).astype(int)

        embeddings = encode_texts(
            scored["response_text"].tolist(),
            model=self.model,
            batch_size=self.settings.batch_size,
            normalize_embeddings=True,
        )
        scored["hedging_similarity"] = embeddings @ self.hedging_vector
        scored["assertive_similarity"] = embeddings @ self.assertive_vector
        scored["hedging_score"] = scored["hedging_similarity"] - scored["assertive_similarity"]

        similarities = embeddings @ self.centroids.vectors.T

        for index, label in enumerate(self.centroids.labels):
            scored[f"similarity_{label}"] = similarities[:, index]

        scored["leaning_score"] = scored["similarity_left"] - scored["similarity_right"]
        scored["neutrality_score"] = scored["similarity_center"] - scored[
            ["similarity_left", "similarity_right"]
        ].max(axis=1)
        scored["bias_strength"] = scored["leaning_score"].abs()

        similarity_columns = [f"similarity_{label}" for label in self.centroids.labels]
        best_indices = similarities.argmax(axis=1)
        scored["closest_reference_label"] = [self.centroids.labels[i] for i in best_indices]
        scored["max_reference_similarity"] = scored[similarity_columns].max(axis=1)

        return scored


def build_news_centroids(
    news_dataset_path_or_url: str | Path,
    output_path: str | Path | None = None,
    settings: EmbeddingScoringSettings | None = None,
) -> ReferenceCentroids:
    """Build centroids from the news-bias dataset and optionally save them."""
    builder = NewsBiasReferenceBuilder(settings=settings)
    centroids = builder.build_from_path(news_dataset_path_or_url)

    if output_path is not None:
        save_centroids(centroids, output_path)

    return centroids


def score_outputs_with_embeddings(
    raw_outputs: pd.DataFrame,
    centroids: ReferenceCentroids,
    output_path: str | Path | None = None,
    settings: EmbeddingScoringSettings | None = None,
) -> pd.DataFrame:
    """Score raw model outputs with embedding-based cosine similarity."""
    scorer = CosineSimilarityScorer(centroids=centroids, settings=settings)
    scored = scorer.score_dataframe(raw_outputs)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)

    return scored
