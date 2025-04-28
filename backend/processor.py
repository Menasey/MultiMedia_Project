# ===============================================================
# backend/processor.py   —  feature & ML utilities (2025‑04‑23)
# ===============================================================
"""
All computationally heavy helpers used by main.py live here.

Public API (imported by main.py):
    extract_urls(zip_path)                    ➜  [str]
    scrape_urls(list[str])                    ➜  [str]
    build_features_and_vect(texts)            ➜  Xf, vect, top_terms
    train_one_class_model(classifier, X)      ➜  model, train_scores
    compute_threshold(scores, keep=0.95)      ➜  float
    score_sample(model, Xrow)                 ➜  float  (normality; higher=normal)
    get_model(name="svm")                     ➜  sklearn.Predictor

The two score helpers **always** follow the convention:
    higher score  ⇒  *more in‑distribution / normal*
so that main.py can apply a single comparison `score >= threshold`.
"""

from __future__ import annotations

import os
import zipfile
import re
import shutil
from typing import List, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup

# ── scikit‑learn & friends ──────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

# EIF (pure‑python extended isolation forest)
from eif import iForest as ExtendedIsolationForest

# Deep One‑Class variant (keras/torch) lives in same package
from .deep_one_class import DeepOneClassClassifier


# ===========================================================================
#  Basic IO helpers  (unchanged from original)
# ===========================================================================

def extract_urls(zip_path: str, extract_dir: str = "temp_urls") -> List[str]:
    """Unzip *zip_path*, read each contained file line‑by‑line, return all URLs."""
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    urls: List[str] = []
    for fname in os.listdir(extract_dir):
        path = os.path.join(extract_dir, fname)
        if os.path.isfile(path):
            with open(path) as fh:
                urls.extend([line.strip() for line in fh if line.strip()])
    return urls


def scrape_urls(urls: List[str], timeout: int = 10) -> List[str]:
    """Best‑effort HTML → plaintext scrape (returns empty string on failure)."""

    def _scrape_one(u: str) -> str:
        try:
            resp = requests.get(u, timeout=timeout)
            resp.raise_for_status()
            text = BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True)
            return re.sub(r"\s+", " ", text)
        except Exception:
            return ""

    return [_scrape_one(u) for u in urls]

# ===========================================================================
#  TF‑IDF helpers
# ===========================================================================

def get_top_terms_by_tfidf(X, vectorizer: TfidfVectorizer, n_terms: int = 300):
    """Select *n_terms* highest‑mean TF‑IDF features for dimensionality control."""
    means = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    idx   = means.argsort()[::-1][:n_terms]
    return [terms[i] for i in idx]

# ---------------------------------------------------------------------------
#  Feature pipeline builder
# ---------------------------------------------------------------------------

def build_features_and_vect(texts: List[str]) -> Tuple[np.ndarray, "Pipeline", List[str]]:
    """Return dense feature matrix, fitted *vect* pipeline, and top‑term list."""
    tfidf0 = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=8000,
    )
    X_full = tfidf0.fit_transform(texts)

    top_terms = get_top_terms_by_tfidf(X_full, tfidf0, n_terms=300)
    tfidf = TfidfVectorizer(
        vocabulary=top_terms,
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    n_components = max(1, min(128, len(top_terms) - 1))
    svd  = TruncatedSVD(n_components=n_components, random_state=42)
    norm = Normalizer(copy=False)

    vect = make_pipeline(tfidf, svd, norm)
    Xf   = vect.fit_transform(texts)
    return Xf, vect, top_terms

# ===========================================================================
#  One‑class model factory  &  training wrapper
# ===========================================================================

def get_model(model_type: str = "svm"):
    if model_type == "svm":
        return OneClassSVM(kernel="rbf", gamma="auto", nu=0.1)
    raise ValueError(f"Unknown model: {model_type}")


def train_one_class_model(classifier: str, Xf) -> Tuple[object, np.ndarray]:
    """Fit the requested *classifier* on feature matrix *Xf*.
    Returns ``model`` and the array of training *normality* scores (higher = normal).
    """
    classifier = classifier.lower()

    # Deep One‑Class (auto‑encoder‑style)
    if classifier == "deep_one_class":
        model = DeepOneClassClassifier(
            input_dim=Xf.shape[1],
            latent_dim=32,
            epochs=50,
            batch_size=16,
        )
        model.fit(Xf)
        scores = model.decision_function(Xf)
        return model, scores

    # Extended Isolation Forest
    if classifier == "eif":
        sample_size = min(128, Xf.shape[0])
        model = ExtendedIsolationForest(
            Xf, ntrees=300, sample_size=sample_size, ExtensionLevel=1,
        )
        if hasattr(model, "anomaly_score"):
            scores = -model.anomaly_score(Xf)
        elif hasattr(model, "compute_scores"):
            scores = -model.compute_scores(Xf)
        else:
            paths  = model.compute_paths(Xf)
            scores = -_paths_to_score(paths, Xf.shape[0])
        return model, scores

    # Plain Isolation Forest
    if classifier == "iforest":
        model = IsolationForest(
            n_estimators=400,
            contamination=0.1,
            random_state=42,
        ).fit(Xf)
        scores = model.score_samples(Xf)      # higher≈normal
        return model, scores

    # Default: One‑Class SVM
    model = get_model(classifier)
    model.fit(Xf)
    scores = model.decision_function(Xf)
    return model, scores

# ---------------------------------------------------------------------------
#  Score helpers (shared with main.py)
# ---------------------------------------------------------------------------

def compute_threshold(scores: np.ndarray, keep: float = 0.9) -> float:
    """Return score cut‑off that keeps *keep* fraction of *highest* scores."""
    return float(np.quantile(scores, 1.0 - keep))


# −− internal util for EIF paths → anomaly −−
_EULER = 0.5772156649

def _paths_to_score(paths: np.ndarray, n_samples: int) -> np.ndarray:
    c_n = 2 * (np.log(n_samples - 1) + _EULER) - 2 * (n_samples - 1) / n_samples
    return 2 ** (-paths / c_n)                       # higher=anomalous


def score_sample(model, X_row) -> float:
    """Return *normality* score of a **single** feature row."""
    from numpy import ndarray  # type: ignore  # (for MyPy friendliness)

    # Extended IF
    if isinstance(model, ExtendedIsolationForest):
        if hasattr(model, "anomaly_score"):
            return -model.anomaly_score(X_row)[0]
        if hasattr(model, "compute_scores"):
            return -model.compute_scores(X_row)[0]
        path  = model.compute_paths(X_row)[0]
        return -_paths_to_score(np.array([path]), model.nobjs)[0]

    # scikit IF
    if isinstance(model, IsolationForest):
        return model.score_samples(X_row)[0]        # already higher=normal

    # SVM / Deep OC etc.
    if hasattr(model, "decision_function"):
        return model.decision_function(X_row)[0]
    if hasattr(model, "score_samples"):
        return model.score_samples(X_row)[0]

    raise TypeError(f"Unsupported model type: {type(model)}")
