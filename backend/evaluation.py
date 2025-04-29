# backend/evaluation.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(X, y_true, model, threshold, k=5):
    """K-Fold cross-validation for any one-class model."""
    from backend.processor import DeepOneClassClassifier, ExtendedIsolationForest
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]

        # CASE 1: Deep One Class (needs dense + fit)
        if isinstance(model, DeepOneClassClassifier):
            X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test
            model.fit(X_train)

        # CASE 2: Extended Isolation Forest (no refit needed)
        elif isinstance(model, ExtendedIsolationForest):
            X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test
            pass  # no fit

        # CASE 3: Normal models (SVM, IForest)
        else:
            model.fit(X_train)

        # ---- Now scoring ----
        try:
            scores = model.decision_function(X_test)
        except AttributeError:
            try:
                scores = model.score_samples(X_test)
            except AttributeError:
                from backend.processor import score_sample
                scores = np.array([score_sample(model, row.reshape(1, -1)) for row in X_test])

        preds = np.where(scores >= threshold, 1, 0)
        y_test = y_true[test_idx]

        results.append({
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        })

    return results
