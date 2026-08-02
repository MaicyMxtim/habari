"""Classical baseline: TF-IDF features with logistic regression.

Uses both word and character n-grams. Character n-grams matter for Swahili
because its agglutinative morphology packs subject, tense, and object
markers into single words that word-level tokens treat as unrelated.
Hyperparameters are selected on the validation split only; the test split
is touched exactly once.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from habari import data, evaluate


def build_pipeline(C=1.0):
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    classifier = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("classifier", classifier)])


def main():
    train = data.load("train")
    val = data.load("validation")
    test = data.load("test")
    labels = data.label_names()

    best_c, best_f1 = None, -1.0
    for C in (0.1, 0.5, 1.0, 2.0, 5.0):
        model = build_pipeline(C)
        model.fit(train["full_text"], train["label"])
        val_f1 = f1_score(val["label"], model.predict(val["full_text"]), average="macro")
        print(f"C={C}: validation macro F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_c, best_f1 = C, val_f1

    print(f"selected C={best_c}, refitting on train+validation")
    import pandas as pd

    train_full = pd.concat([train, val], ignore_index=True)
    model = build_pipeline(best_c)
    model.fit(train_full["full_text"], train_full["label"])

    predictions = model.predict(test["full_text"])
    probabilities = model.predict_proba(test["full_text"])
    metrics = evaluate.evaluate(
        "tfidf_logreg",
        test["label"].to_numpy(),
        predictions,
        labels,
        texts=test["full_text"].tolist(),
        probabilities=probabilities,
    )
    evaluate.print_report(metrics)


if __name__ == "__main__":
    main()
