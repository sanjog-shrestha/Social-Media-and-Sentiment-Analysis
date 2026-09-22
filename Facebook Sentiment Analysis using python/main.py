"""
Text Analysis with NLTK: Tokenization, Stemming, Lemmatization, POS Tagging, Sentiment
-------------------------------------------------------------------------------------------
Runs a full classic-NLP sweep over a text file: sentence/word tokenization,
stemming, lemmatization, part-of-speech tagging, and line-by-line VADER
sentiment scoring.
"""

from dataclasses import dataclass

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    file_path: str = "fb_sentiment.csv"
    encoding: str = "ISO-8859-2"


def download_nltk_resources() -> None:
    """Download the NLTK resources this pipeline depends on."""
    nltk.download("punkt_tab")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger_eng")
    nltk.download("vader_lexicon")


def load_text(cfg: PipelineConfig) -> str:
    """Read the source file as raw text."""
    with open(cfg.file_path, encoding=cfg.encoding) as f:
        return f.read()


def tokenize_sentences(text: str) -> list:
    """Tokenize text into sentences using a Punkt tokenizer trained on this text."""
    sentences = PunktSentenceTokenizer(text).tokenize(text)
    print(sentences)
    return sentences


def tokenize_words(text: str) -> list:
    """Tokenize text into words and print the token list."""
    tokens = word_tokenize(text)
    print(tokens)
    return tokens


def stem_words(tokens: list) -> None:
    """Print each token alongside its Porter stem."""
    stemmer = PorterStemmer()
    for w in tokens:
        print("Actual: %s  Stem: %s" % (w, stemmer.stem(w)))


def lemmatize_words(tokens: list) -> None:
    """Print each token alongside its WordNet lemma."""
    lemmatizer = WordNetLemmatizer()
    for w in tokens:
        print("Actual %s Lemma: %s" % (w, lemmatizer.lemmatize(w)))


def pos_tag_tokens(tokens: list) -> None:
    """Print part-of-speech tags for each token."""
    print(nltk.pos_tag(tokens))


def analyze_line_sentiment(cfg: PipelineConfig) -> None:
    """Run VADER sentiment analysis on the file line by line."""
    sid = SentimentIntensityAnalyzer()
    with open(cfg.file_path, encoding=cfg.encoding) as f:
        for line_text in f.read().split("\n"):
            print(line_text)
            scores = sid.polarity_scores(line_text)
            for key in sorted(scores):
                print("{0}: {1}, ".format(key, scores[key]), end="")
            print()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Download resources, then run tokenization, stemming, lemmatization, POS tagging, and sentiment analysis."""
    download_nltk_resources()

    text = load_text(cfg)
    tokenize_sentences(text)
    tokens = tokenize_words(text)

    stem_words(tokens)
    lemmatize_words(tokens)
    pos_tag_tokens(tokens)

    analyze_line_sentiment(cfg)


if __name__ == "__main__":
    run_pipeline()