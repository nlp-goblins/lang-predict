import pickle
import unicodedata
import re
from markdown import markdown
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from functools import partial, reduce

MODEL_OBJ = "lrmodel.obj"
TFIDF_OBJ = "tfidf.obj"

# right to left
def compose(*fns):
    return partial(reduce, lambda x, f: f(x), reversed(fns))


# applies in the order supplied
def pipe(v, *fns):
    return reduce(lambda x, f: f(x), fns, v)


def map_exhaust(func, *iters):
    for args in zip(*iters):
        func(*args)


def normalize_text(text):
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def remove_chars(text):
    return re.sub(r"[^A-Za-z0-9 ]", "", re.sub(r"\s", " ", text))


def remove_bogus_words(text):
    no_single_words = re.sub(r"\s.{1}\s", "", text)  # remove single characters
    return re.sub(r"http.{1,}[\s\.]*", "", no_single_words)  # remove links


def basic_clean(text):
    return pipe(text, str.lower, normalize_text, remove_chars)


def tokenize(text):
    tokenizer = ToktokTokenizer()
    return tokenizer.tokenize(text, return_str=True)


def stem(text):
    ps = nltk.porter.PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])


def lemmatize(text):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    return " ".join(lemmas)


def remove_stopwords(text, include=[], exclude=[]):
    stopword_list = stopwords.words("english")

    map_exhaust(stopword_list.remove, exclude)
    map_exhaust(stopword_list.append, include)

    removed = " ".join([w for w in text.split() if w not in stopword_list])

    #     print("Removed", len(text.split()) - len(removed.split()), "words")
    return removed


def predict_lang(readme):
    # render the markdown to html
    html = markdown(readme)
    # and extract the text from the html
    soup = BeautifulSoup(html, "html.parser")
    text = soup.text

    clean = pipe(
        text, basic_clean, tokenize, remove_stopwords, remove_bogus_words
    )

    # stemmed = stem(clean)

    with open(TFIDF_OBJ, "rb") as fp:
        tfidf = pickle.load(fp)

    # I need to rerun the notebook to save the right vectorizer
    tfidf_vals = tfidf.transform([clean])

    with open(MODEL_OBJ, "rb") as fp:
        model = pickle.load(fp)

    return model.predict(tfidf_vals)[0]
