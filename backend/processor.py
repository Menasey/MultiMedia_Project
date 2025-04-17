# backend/processor.py

import os, zipfile, re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

def extract_urls(zip_path, extract_dir="temp_urls"):
    os.makedirs(extract_dir, exist_ok=True)
    urls = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    for fname in os.listdir(extract_dir):
        path = os.path.join(extract_dir, fname)
        if os.path.isfile(path):
            with open(path) as f:
                urls += [l.strip() for l in f if l.strip()]
    return urls

def scrape_urls(urls):
    def scrape(u):
        try:
            r = requests.get(u, timeout=10)
            r.raise_for_status()
            text = BeautifulSoup(r.text, 'html.parser').get_text(' ', strip=True)
            return re.sub(r'\s+', ' ', text)
        except:
            return ""
    return [scrape(u) for u in urls]

def encode_texts_with_selected_terms(texts, top_terms):
    if top_terms:
        vec = TfidfVectorizer(vocabulary=top_terms, lowercase=True)
    else:
        vec = TfidfVectorizer(max_features=5000, stop_words='english', lowercase=True)
    X = vec.fit_transform(texts)
    return X, vec

def get_top_terms_by_tfidf(X, vectorizer, n_terms=300):
    import numpy as np
    means = np.array(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    idx = means.argsort()[::-1][:n_terms]
    return [terms[i] for i in idx]

def reduce_dimensionality(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X), svd

def normalize_vectors(X):
    return Normalizer(norm='l2').fit_transform(X)

def get_model(model_type='svm'):
    if model_type == 'svm':
        return OneClassSVM(kernel='rbf', gamma='scale', nu=0.3)
    elif model_type == 'iforest':
        return IsolationForest(contamination=0.3, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_type}")