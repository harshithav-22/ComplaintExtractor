"""
Customer Complaint Topic Extractor
----------------------------------
Pipeline: load -> filter 1-2 star -> clean -> lemmatize -> TF-IDF (uni+bi-grams)
          -> K-Means (k chosen by silhouette) -> themes + example reviews + bigram counts.

Usage:
    python complaint_topic_extractor.py --csv "Womens Clothing E-Commerce Reviews.csv"
    python complaint_topic_extractor.py --csv reviews.csv --text-col content --rating-col score --k 5

Requires: pandas, scikit-learn, nltk   (pip install pandas scikit-learn nltk)
"""
import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score

# ---------- Lemmatizer (WordNet if available, light fallback otherwise) ----------
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    _wnl = WordNetLemmatizer()
    _wnl.lemmatize("tests")  # force-load; raises if data missing

    def lemma(w):
        # try verb form first (arrived->arrive, shipped->ship), then noun
        v = _wnl.lemmatize(w, "v")
        return v if v != w else _wnl.lemmatize(w, "n")
except Exception:  # pragma: no cover
    def lemma(w):
        for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: -len(suf)] + rep
        return w

# ---------- Stopwords ----------
# Keep negations ("not", "no", "never") out of the list: they carry complaint meaning.
KEEP = {"not", "no", "never", "nor", "cannot"}
DOMAIN_STOP = {
    "item", "product", "order", "ordered", "get", "got", "would", "could", "one",
    "really", "just", "also", "even", "like", "dress", "top", "shirt", "app",  # tweak per dataset
    "buy", "bought", "purchase", "purchased", "wear", "wore", "review",
    # sentiment filler that forms meaningless clusters ("wanted to love it")
    "love", "want", "wanted", "loved", "cute", "pretty", "beautiful", "nice", "great",
    "good", "sure", "unfortunately", "disappointed", "disappointing", "disappoint",
    "try", "tried", "way", "time", "much", "little", "bit", "didnt", "dont", "wasnt",
}
STOP = (set(ENGLISH_STOP_WORDS) - KEEP) | DOMAIN_STOP

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\uFE0F\u200d]+"
)


def clean(text: str) -> str:
    t = text.lower()
    t = EMOJI_RE.sub(" ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = t.replace("'", "")                 # don't -> dont, wasn't -> wasnt
    t = re.sub(r"[^a-z\s]", " ", t)        # strip punctuation / digits
    toks = [lemma(w) for w in t.split() if len(w) > 2]
    toks = [w for w in toks if w not in STOP]
    return " ".join(toks)


# ---------- Loading ----------
def load(path, text_col=None, rating_col=None, n_max=5000, seed=42):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = text_col or next(
        (cols[c] for c in ("review text", "review", "content", "text", "review_text") if c in cols), None)
    rating_col = rating_col or next(
        (cols[c] for c in ("rating", "score", "stars", "review_rating") if c in cols), None)
    if not text_col or not rating_col:
        raise SystemExit(f"Could not detect columns. Found: {list(df.columns)}. Use --text-col/--rating-col.")

    df = df[[text_col, rating_col]].rename(columns={text_col: "raw", rating_col: "rating"})
    df = df[df["rating"].isin([1, 2])]
    df = df.dropna(subset=["raw"])
    df = df[df["raw"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset="raw")
    if len(df) > n_max:
        df = df.sample(n_max, random_state=seed)
    print(f"[load] {len(df)} low-rated reviews (1-2 stars) after dropping blanks/duplicates")
    return df.reset_index(drop=True)


# ---------- Mining ----------
def pick_k(X, k_range=range(4, 9), seed=42):
    best, best_s = None, -1
    sample = np.random.RandomState(seed).choice(X.shape[0], min(2000, X.shape[0]), replace=False)
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)
        s = silhouette_score(X[sample], km.labels_[sample])
        print(f"[k-select] k={k}  silhouette={s:.4f}")
        if s > best_s:
            best, best_s = k, s
    return best


def top_terms(km, vec, n=8):
    terms = np.array(vec.get_feature_names_out())
    order = km.cluster_centers_.argsort()[:, ::-1]
    return [terms[order[i, :n]].tolist() for i in range(km.n_clusters)]


def bigram_counts(docs, n=15):
    c = Counter()
    for d in docs:
        w = d.split()
        c.update(zip(w, w[1:]))
    return [(" ".join(b), n_) for b, n_ in c.most_common(n)]


def run(df, k=None, seed=42):
    df["clean"] = df["raw"].map(clean)
    df = df[df["clean"].str.split().str.len() >= 3].reset_index(drop=True)  # drop near-empty after cleaning
    print(f"[clean] {len(df)} reviews kept after cleaning")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True)
    X = vec.fit_transform(df["clean"])
    print(f"[tfidf] matrix {X.shape}")

    k = k or pick_k(X, seed=seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    df["cluster"] = km.labels_

    terms = top_terms(km, vec)
    # nearest-to-centroid reviews are the most representative
    dist = km.transform(X)
    rows = []
    for c in range(k):
        members = df.index[df["cluster"] == c]
        rep = members[np.argsort(dist[members, c])[:3]]
        rows.append({
            "cluster": c,
            "n_reviews": len(members),
            "share_%": round(100 * len(members) / len(df), 1),
            "top_terms": ", ".join(terms[c]),
            "examples": [df.loc[i, "raw"][:220] for i in rep],
            "bigrams": bigram_counts(df.loc[members, "clean"], 5),
        })
    summary = pd.DataFrame(rows).sort_values("n_reviews", ascending=False).reset_index(drop=True)
    return df, summary, bigram_counts(df["clean"], 20)


def report(summary, global_bigrams):
    print("\n" + "=" * 70 + "\nTOP COMPLAINT THEMES (largest first)\n" + "=" * 70)
    for i, r in summary.iterrows():
        print(f"\n#{i+1}  Cluster {r.cluster}  -  {r.n_reviews} reviews ({r['share_%']}%)")
        print(f"   keywords : {r.top_terms}")
        print(f"   bigrams  : {r.bigrams}")
        for ex in r.examples:
            print(f"   > {ex}")
    print("\nOverall most frequent bigrams:")
    for b, n in global_bigrams:
        print(f"   {n:5d}  {b}")
    print("\nTip: label each cluster yourself from its keywords (e.g. 'sizing', 'late delivery').")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col")
    ap.add_argument("--rating-col")
    ap.add_argument("--k", type=int, help="number of themes (default: auto 4-8)")
    ap.add_argument("--max", type=int, default=5000)
    a = ap.parse_args()

    df = load(a.csv, a.text_col, a.rating_col, a.max)
    df, summary, gb = run(df, a.k)
    report(summary, gb)
    df.drop(columns=["clean"]).to_csv("reviews_with_clusters.csv", index=False)
    summary.drop(columns=["examples"]).to_csv("theme_summary.csv", index=False)
    print("\nSaved reviews_with_clusters.csv and theme_summary.csv")
