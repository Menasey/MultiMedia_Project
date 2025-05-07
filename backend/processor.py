# ===============================================================
# backend/processor.py   —  feature & ML utilities
# ===============================================================

from __future__ import annotations
import os, zipfile, re, shutil
from typing import List, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM

from .deep_one_class import DeepOneClassClassifier

# ========================== IO HELPERS ==========================

def extract_urls(zip_path: str, extract_dir: str = "temp_urls") -> List[str]:
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    urls = []
    for fname in os.listdir(extract_dir):
        path = os.path.join(extract_dir, fname)
        if os.path.isfile(path):
            with open(path) as fh:
                urls.extend([line.strip() for line in fh if line.strip()])
    return urls

def scrape_urls(urls: List[str], timeout: int = 10) -> List[str]:
    def _scrape_one(u: str) -> str:
        try:
            resp = requests.get(u, timeout=timeout)
            resp.raise_for_status()
            text = BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True)
            return re.sub(r"\s+", " ", text)
        except Exception:
            return ""
    return [_scrape_one(u) for u in urls]

# ====================== TF-IDF + FEATURES =======================

def get_top_terms_by_tfidf(X, vectorizer: TfidfVectorizer, n_terms: int = 300):
    means = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    idx   = means.argsort()[::-1][:n_terms]
    return [terms[i] for i in idx]

def build_features_and_vect(texts):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        lowercase=True
    )
    X = vectorizer.fit_transform(texts)
    top_terms = vectorizer.get_feature_names_out()
    return X, vectorizer, top_terms

# ====================== MODEL TRAINING ==========================

def train_one_class_model(classifier: str, Xf, **kwargs) -> Tuple[object, np.ndarray]:
    classifier = classifier.lower()

    if classifier == "deep_one_class":
        X_dense = Xf.toarray() if hasattr(Xf, "toarray") else Xf
        model = DeepOneClassClassifier(
            input_dim=X_dense.shape[1],
            latent_dim=kwargs.get("latent_dim", 32),
            epochs=kwargs.get("epochs", 50),
            batch_size=kwargs.get("batch_size", 16),
        )
        model.fit(X_dense)
        scores = model.decision_function(X_dense)
        return model, scores

    # One-Class SVM (default)
    nu = kwargs.get("nu", 0.1)
    gamma = kwargs.get("gamma", 'auto')
    model = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
    model.fit(Xf)
    scores = model.decision_function(Xf)
    return model, scores

# ===================== THRESHOLD + SCORING ======================

def compute_threshold(scores: np.ndarray, alpha: float = 0.5) -> float:
    mean = np.mean(scores)
    std = np.std(scores)
    if std < 1e-8:
        std = 1e-8  # prevent threshold = mean
    return float(mean - alpha * std)


def score_sample(model, X_sample):
    try:
        return model.decision_function(X_sample)[0]
    except AttributeError:
        try:
            return model.score_samples(X_sample)[0]
        except AttributeError:
            raise ValueError("Unsupported model type for scoring")

