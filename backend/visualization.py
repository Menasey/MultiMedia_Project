import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import numpy as np

def plot_fold_scores(results, outpath=None):
    folds = list(range(1, len(results) + 1))
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1"] for r in results]

    plt.figure(figsize=(10, 5))
    plt.plot(folds, precision, label="Precision", marker='o')
    plt.plot(folds, recall, label="Recall", marker='o')
    plt.plot(folds, f1, label="F1 Score", marker='o')
    plt.xlabel("Fold")
    plt.ylabel("Score")
    plt.title("Cross-Validation Performance per Fold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath)
        print(f"Saved {outpath}")
    else:
        plt.show()


def plot_decision_scores(model, X, outpath=None):
    try:
        scores = model.decision_function(X)
    except AttributeError:
        try:
            scores = model.score_samples(X)
        except AttributeError:
            try:
                # Extended Isolation Forest special case
                if hasattr(X, "toarray"):
                    X = X.toarray()  # 🛠 convert sparse to dense

                paths = model.compute_paths(X)
                n_samples = X.shape[0]
                euler_constant = 0.5772156649
                c_n = 2 * (np.log(n_samples - 1) + euler_constant) - (2 * (n_samples - 1) / n_samples)
                anomaly_scores = 2 ** (-paths / c_n)  # standard EIF scoring
                scores = -anomaly_scores  # NEGATE to follow normality convention
            except AttributeError:
                raise ValueError("Unsupported model type for scoring")

    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=30, edgecolor='black')
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath)
        print(f"Saved {outpath}")
    else:
        plt.show()

def get_top_terms_by_tfidf(X_tfidf, vectorizer, n_terms=100):
    tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_indices = tfidf_means.argsort()[::-1][:n_terms]
    terms = vectorizer.get_feature_names_out()
    return [terms[i] for i in top_indices]

def plot_tfidf_term_importance(X_tfidf, top_terms, n_terms=30, outpath=None):
    tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_indices = tfidf_means.argsort()[::-1][:n_terms]

    selected_terms = [top_terms[i] for i in top_indices]
    selected_scores = [tfidf_means[i] for i in top_indices]

    plt.figure(figsize=(12, 6))
    bars = plt.barh(selected_terms[::-1], selected_scores[::-1])  # reverse to get highest at top
    plt.xlabel("Average TF-IDF Score")
    plt.title(f"Top {n_terms} Most Informative Terms (TF-IDF)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        print(f"Saved {outpath}")
    else:
        plt.show()
