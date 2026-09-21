"""
Tweet Sentiment Classification — Model Comparison
------------------------------------------------------
Classifies tweet sentiment (positive/negative) using TF-IDF features and
compares three classifiers: Bernoulli Naive Bayes, Linear SVM, and
Logistic Regression. Covers data loading/cleaning, text preprocessing,
vectorization, training, evaluation, and sample predictions.
"""

import csv
from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    csv_path: str = "training.1600000.processed.noemoticon.csv"
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 5000
    ngram_range: tuple = (1, 2)
    sample_tweets: list = field(default_factory=lambda: [
        "I love this!", "I hate that!", "It was okay, not great."
    ])


def load_data(cfg: PipelineConfig) -> pd.DataFrame:
    """Load the raw CSV and keep only the polarity and text columns."""
    df = pd.read_csv(
        cfg.csv_path, encoding="latin-1", header=None,
        on_bad_lines="skip", quoting=csv.QUOTE_NONE,
    )
    df = df[[0, 5]]
    df.columns = ["polarity", "text"]
    print(df.head())
    return df


def clean_polarity(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the polarity column, drop neutral rows, and map to binary labels."""
    df["polarity"] = df["polarity"].astype(str).str.strip().str.replace('"', "")

    print("Polarity value counts BEFORE filtering and mapping:")
    print(df["polarity"].value_counts(dropna=False))

    df = df[df.polarity != "2"]  # drop neutral sentiment
    print("Polarity value counts AFTER filtering '2's:")
    print(df["polarity"].value_counts(dropna=False))

    df["polarity"] = df["polarity"].map({"0": 0, "4": 1})
    df.dropna(subset=["polarity"], inplace=True)
    df["polarity"] = df["polarity"].astype(int)

    print("Polarity value counts AFTER mapping and cleanup:")
    print(df["polarity"].value_counts())
    print("Final df.shape:", df.shape)
    return df


def clean_text(text: str) -> str:
    """Basic text normalization."""
    return text.lower()


def prepare_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and preview the result."""
    df["clean_text"] = df["text"].apply(clean_text)
    print(df[["text", "clean_text"]].head())
    return df


def split_data(df: pd.DataFrame, cfg: PipelineConfig):
    """Split into train/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["polarity"], test_size=cfg.test_size, random_state=cfg.random_state
    )
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test, cfg: PipelineConfig):
    """Fit a TF-IDF vectorizer on training text only, then transform both splits."""
    vectorizer = TfidfVectorizer(max_features=cfg.max_features, ngram_range=cfg.ngram_range)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print("TF-IDF shape (train):", X_train_tfidf.shape)
    print("TF-IDF shape (test):", X_test_tfidf.shape)
    return X_train_tfidf, X_test_tfidf, vectorizer


def train_and_evaluate(name: str, model, X_train_tfidf, y_train, X_test_tfidf, y_test):
    """Fit a model, predict on the test set, and print accuracy + classification report."""
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    print(f"{name} Accuracy:", accuracy_score(y_test, predictions))
    print(f"\n{name} Classification Report:\n", classification_report(y_test, predictions))
    return model


def preview_sample_predictions(models: dict, vectorizer: TfidfVectorizer, cfg: PipelineConfig) -> None:
    """Run all trained models on a few hand-written sample tweets."""
    sample_vec = vectorizer.transform(cfg.sample_tweets)
    print("\nSample Predictions:")
    for name, model in models.items():
        print(f"{name}:", model.predict(sample_vec))


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Load data, clean it, train all three models, evaluate, and preview predictions."""
    df = load_data(cfg)
    df = clean_polarity(df)
    df = prepare_text_column(df)

    X_train, X_test, y_train, y_test = split_data(df, cfg)
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test, cfg)

    model_builders = {
        "BernoulliNB": BernoulliNB(),
        "SVM": LinearSVC(),
        "Logistic Regression": LogisticRegression(),
    }

    trained_models = {
        name: train_and_evaluate(name, model, X_train_tfidf, y_train, X_test_tfidf, y_test)
        for name, model in model_builders.items()
    }

    preview_sample_predictions(trained_models, vectorizer, cfg)


if __name__ == "__main__":
    run_pipeline()