# Tweet Sentiment Classification — Model Comparison

Classifies tweet sentiment (positive/negative) using TF-IDF text features, comparing three classic ML classifiers on the same data: Bernoulli Naive Bayes, Linear SVM, and Logistic Regression.

## What it does

1. Loads the raw Sentiment140 CSV (1.6M tweets) and keeps only the polarity and text columns.
2. Cleans the polarity column: strips whitespace/quotes, drops neutral-labeled rows (`'2'`), and maps the remaining values to binary labels (`0` = negative, `4` → `1` = positive).
3. Lowercases the tweet text.
4. Splits the cleaned data into train/test sets.
5. Fits a TF-IDF vectorizer (unigrams + bigrams, capped at 5,000 features) on training text only, then transforms both splits.
6. Trains three classifiers on the same TF-IDF features: `BernoulliNB`, `LinearSVC`, and `LogisticRegression`.
7. Evaluates each on accuracy and a full classification report (precision/recall/F1 per class).
8. Runs all three trained models on a few hand-written sample tweets to sanity-check predictions.

## Concepts covered

- **TF-IDF vectorization** — weights words by how distinctive they are to a document relative to the whole corpus (common words across all tweets get down-weighted, rare-but-repeated words get up-weighted), generally outperforming raw word counts for text classification.
- **N-grams (`ngram_range=(1, 2)`)** — including both single words (unigrams) and two-word phrases (bigrams) as features lets the model pick up on short phrases ("not good") that a single-word model would miss the meaning of.
- **Fit/transform separation** — the vectorizer is `.fit_transform()`-ed on training text only and `.transform()`-ed (not re-fit) on test text, so evaluation reflects performance on genuinely unseen vocabulary distributions, not vocabulary the model was allowed to see in advance.
- **Comparing multiple classifiers on identical features** — training `BernoulliNB`, `LinearSVC`, and `LogisticRegression` on the exact same TF-IDF matrix isolates the effect of the *classifier* from the effect of the *features*, making the accuracy comparison meaningful.
- **Classification report** — accuracy alone can hide class imbalance issues; precision, recall, and F1 per class show whether a model is systematically better at detecting one sentiment than the other.

## Project structure

```
main.py      # full pipeline, organized into functions (config, data loading,
              # cleaning, vectorization, training/evaluation, sample predictions)
README.md    # this file
```

The code is organized into small, named functions driven by a `PipelineConfig` dataclass — `load_data`, `clean_polarity`, `prepare_text_column`, `vectorize_text`, `train_and_evaluate`, `preview_sample_predictions` — tied together by `run_pipeline()`. All three models are trained and evaluated through one shared `train_and_evaluate()` function via a `model_builders` dict, rather than three near-identical blocks of code.

## Requirements

```
pandas
scikit-learn
```

Install with:

```bash
pip install pandas scikit-learn
```

## How to run

Download the [Sentiment140 dataset](https://www.kaggle.com/datasets/milobele/sentiment140-dataset-1600000-tweets?resource=download) (`training.1600000.processed.noemoticon.csv`) and place it in the same directory as `main.py`, then:

```bash
python main.py
```

Note: this is a 1.6M-row dataset — TF-IDF fitting and training all three models can take a few minutes depending on your machine.

## Sample output

Running the script prints dataset diagnostics, TF-IDF shapes, accuracy and a classification report for each of the three models, and predictions on sample tweets.

## References

- [GeeksforGeeks — Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/) — used as a general reference/inspiration while working on this project.

---
*This is a personal learning project, not a production sentiment analysis system.*
