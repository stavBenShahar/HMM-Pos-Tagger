# pylint: disable=invalid-name
import nltk
from nltk.corpus import brown
from src import HMMTagger, BaseLine, add_1_smoothing
from src.special_types import WordT, TagT


def announce(func):
    def wrapper(*args, **kwargs):
        print(f" {func.__module__}.{func.__qualname__} started ".center(40, '='))
        return func(*args, **kwargs)

    return wrapper


def flatten(arr) -> list:
    return [elem for sublist in arr for elem in sublist]


@announce
def A(category: str = "news", split_ratio: float = 0.9) -> tuple[
    list[list[tuple[WordT, TagT]]], list[list[tuple[WordT, TagT]]]]:
    brown_corpus = brown.tagged_sents(categories=category)
    split_index = round(len(brown_corpus) * split_ratio)
    train_set = brown_corpus[:split_index]
    test_set = brown_corpus[split_index:]
    return list(train_set), list(test_set)  # casting to list is important as it is not a list and behaves differently


@announce
def B(train_set: list[list[tuple[WordT, TagT]]], test_set: list[list[tuple[WordT, TagT]]]):
    print("BASELINE")
    lm = BaseLine()
    lm.train(train_set, test_set)

    lm.loss(test_set)
    # Known words error rate: 0.0832
    # Unknown words error rate: 0.7893
    # Total error rate: 0.1638


@announce
def C(train_set: list[list[tuple[WordT, TagT]]], test_set: list[list[tuple[WordT, TagT]]]):
    print("HMM")
    lm = HMMTagger()
    lm.train(train_set, test_set)

    lm.loss(test_set)
    # Known words error rate: 0.8859
    # Unknown words error rate: 0.7893
    # Total error rate: 0.8749


@announce
def D(train_set: list[list[tuple[WordT, TagT]]], test_set: list[list[tuple[WordT, TagT]]]):
    print("HMM + smoothing")
    lm = HMMTagger(add_1_smoothing)
    lm.train(train_set, test_set)

    lm.loss(test_set)
    # Known words error rate: 0.1695
    # Unknown words error rate: 0.7893
    # Total error rate: 0.2403


@announce
def E(train_set: list[list[tuple[WordT, TagT]]], test_set: list[list[tuple[WordT, TagT]]]):
    print("HMM + smoothing + pseudo words")
    lm = HMMTagger(add_1_smoothing, use_pseudo_words=True)
    lm.train(train_set, test_set)
    y_true = flatten([[tag for _, tag in sent] for sent in test_set])
    y_pred = flatten(preds := lm.predict(test_set))

    cm = nltk.ConfusionMatrix(y_true, y_pred)
    with open("./conf.txt", "w", encoding="utf8") as f:
        print(cm.pretty_format(sort_by_count=True, show_percents=False), file=f)

    lm.loss(test_set, preds)
    # Known words error rate: 0.2113
    # Unknown words error rate: 0
    # Total error rate: 0.2113


if __name__ == '__main__':
    train_set, test_set = A()
    B(train_set, test_set)
    C(train_set, test_set)
    D(train_set, test_set)
    E(train_set, test_set)
