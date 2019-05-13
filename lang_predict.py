# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # README Prophesy
# By Nicole Garza & Michael P. Moran

#   ## Table of contents
# 1. [Project Planning](#project-planning)
# 1. [Acquisition](#acquisition)
# 1. [Preparation](#preparation)
# 1. [Exploration](#exploration)
# 1. [Modeling](#modeling)

# ## Project Planning <a name="project-planning"></a>

# ### Goals
#
# A model that predicts the predominant language of a GitHub repo given the repo's README file

# ### Deliverables
#
# * Jupyter notebook containing analysis
# * One or two google slides suitable for a general audience that summarize findings. Include a well-labelled visualization in your slides.
# * A function taking a README as input and outputting the language

# ### Project Conclusions

# * Acquisition
#     * Acquiring the data was challenging. We had two main options: (1) scrape the HTML-rendered search results and (2) use the JSON API. We chose to use the JSON API because it allowed dictionary-based access to the information we needed. We did not have to identify the correct HTML tag or do anything else associated with scraping. The JSON API also allowed us to download many more repos faster than scraping the HTML.
#     * We also had the choice of scraping the HTML rendered README or downloading the raw README file. Acquiring the raw README was significantly easier, so we chose this. However, GitHub's API only returns a raw README file, not the rendered HTML version (although it's available in a link).
# * Preparation
#     * We processed the raw README using a markdown module, which rendered it to HTML. We then used Beautiful Soup to extract the text. We removed single character words (which was pointless because the sklearn vectorizers do this already) and also removed links.
#     * We also removed non-English repos given that we are ASCII normalizing. Thus, our model is for English repos only. We dropped repos with no programming language, so our model has this limitation.
# * Exploration
#     * The most common languages were JavaScript, Java, Python, C++, and HTML. JavaScript and Java are heavily overrepresented. If we had more time, we would have rebalanced the dataset, so they do not predominate.
#     * The most common words look like generic programming terms ("use", "code", "file") and do not appear to be useful indicators of the language (except for JavaScript, which we hope would indicate JavaScript). Also, there is significant overlap of the most common bigrams for the languages. Thus, bigrams may not perform better than single words.
#     * We have problems with runon words. If we had more time, this is something to address.
# * Modeling
#     * Our model predicts for only the top 5 most common repos. Thus, an input repo that is not predominately programmed in one of these languages will automatically be wrong. We tried to use an "other" category but this sorely hurt our models' performance. Accuracy plummeted about 20-30 percentage points on average. It may be because "other" had such diversity of language it was pulling in repos it shouldn't. We tried using bigrams, but these did not give us better predictive power, which was not expected. The bigrams appeared to be unique overall to the individual languages.
#     * We also used lemmatized, stemmed, and clean version of the README. Clean appeared to perform on par with stemmed. Not really sure why at this time.
#     * With more time, we would add the number of words in the README as a feature.
#     * KNN and Random Forest give us our best results

# ### Data Dictionary & Domain Knowledge

# ### Hypotheses
# * The primary language may be mentioned in the README. But some repositories mention multiple languages, so this may interfere with this method.
# * The number of words may be an indication of the language. Older repositories are probably written in certain languages and because of their age, may have more documentation.

# ### Thoughts & Questions
#
# * The code in many repositories are written in multiple languages. We will go with the most predominant language.
# * Take out repos with no programming language
# * After taking out the "Other" programming language category, the accuracy of the model shot way up! I believe this category acquired so much language that was used in the top5 repos the models were having difficulty choosing the class.
#

# ### Prepare the Environment

# +
import os
import json
from pprint import pprint
import re
import unicodedata
from functools import reduce, partial
from copy import deepcopy
from markdown import markdown
import pickle

import requests
from bs4 import BeautifulSoup
import pandas as pd

import env

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2, whiten
from sklearn.cluster import KMeans
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from langdetect import detect

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# -

# ## Acquisition <a name="acquisition"></a>

# **Grab data for 1000 most forked repos on GitHub**

# **Constants**

NUM_PER_PAGE = 50
PAGES = 20
API_URL = f"https://api.github.com/search/repositories?q=stars:%3E1&sort=forks&order=desc&per_page={NUM_PER_PAGE}"
HEADERS = {"Authorization": f"token {env.oauth_token}"}
REPO_FILE_NAME = "repos.json"


# **Download the data or read from repos.json file**

# +
def github_api_req(page):
    data = requests.get(API_URL + f"&page={page}", headers=HEADERS).json()
    return data["items"]


def readme_url(contents_url):
    # find name of README file and construct a link to the raw text of the readme
    for file in requests.get(contents_url, headers=HEADERS).json():
        if file["name"].lower().startswith("readme"):
            return file["download_url"]


def load_repo_metadata(use_cache=True):
    if use_cache and os.path.exists(REPO_FILE_NAME):
        with open(REPO_FILE_NAME, "r") as f:
            contents = json.load(f)
        return contents
    else:
        response = []
        for num in range(1, PAGES + 1):
            response += github_api_req(num)
        for repo in response:
            # get link to contents of repo
            contents_url = repo["contents_url"][
                :-8
            ]  # remove last 8 characters to get working URL

            rmurl = readme_url(contents_url)

            readme_text = None  # sometimes there is no valid URL to the readme
            if rmurl:
                # download README text
                readme_text = requests.get(rmurl, headers=HEADERS).text

            repo["readme"] = readme_text
        with open(REPO_FILE_NAME, "w") as f:
            json.dump(response, f)
        return response


all_repo_data = load_repo_metadata()
# -

all_repo_data[:3]


# ## Preparation <a name="preparation"></a>

# +
def all_repo_metadata(api_data):
    return [
        repo_metadata(repo) for repo in api_data if repo["readme"] is not None
    ]


def repo_metadata(api_dict):
    # store the id, username, name of repo
    repo_id = api_dict["id"]
    user_name = api_dict["owner"]["login"]
    repo_name = api_dict["name"]

    # find the predominant programming language
    lang = api_dict["language"]

    # render the markdown to html
    html = markdown(api_dict["readme"])
    # and extract the text from the html
    soup = BeautifulSoup(html, "html.parser")
    readme_text = soup.text

    return dict(
        repo_id=repo_id,
        user_name=user_name,
        repo_name=repo_name,
        lang=lang,
        readme=readme_text,
    )


some_repo_data = all_repo_metadata(all_repo_data)
# -

some_repo_data[:3]


# **Clean, stem, lemmatize, and remove stopwords**

# +
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


def prep_readme(repo_data):
    copy = deepcopy(repo_data)

    copy["clean"] = pipe(
        copy["readme"],
        basic_clean,
        tokenize,
        remove_stopwords,
        remove_bogus_words,
    )

    copy["stemmed"] = stem(copy["clean"])

    copy["lemmatized"] = lemmatize(copy["clean"])

    return copy


def prep_readme_data(all_repo_data):
    return [prep_readme(repo) for repo in all_repo_data]


df = pd.DataFrame(prep_readme_data(some_repo_data))
# -

# any single j's?

df[df.clean.str.contains(" j ")]

# any links?

df[df.clean.str.contains(" http")]

# ### Summarize Data

df.head()

df.describe(include="all")

df.lang.value_counts(dropna=False)

# ### Remove repos that have one or fewer words

print("Before removal:", len(df))
df = df[df.clean.apply(lambda s: len(s.split()) > 1)]
print("After removal:", len(df))

# ### Remove Non-English Repos

# **What does the language spread look like?**

df.clean.apply(detect).value_counts()

print("Before removal:", len(df))
df = df[df.clean.apply(detect) == "en"]
print("After removal:", len(df))

# ### Check Missing Values

# #### Remove repos that have no programming language

len(df)

df.isna().sum()

df = df.dropna()

df.isna().sum()

len(df)

# #### Most common languages

langs = pd.concat(
    [df.lang.value_counts(dropna=False), df.lang.value_counts(dropna=False, normalize=True)], axis=1
)
langs.columns = ["n", "percent"]
langs

# **Go with top 5 languages and roll the rest into an "other" category**

# +
top_five = list(langs[:5].index)
pprint(top_five)

lang_grouped = df.lang.apply(
    lambda lang: lang if lang in top_five else "Other"
).rename("lang_grouped")

# pprint(lang_grouped)
df = pd.concat([df, lang_grouped], axis=1)
# -

df.lang_grouped.value_counts()

# ### Ensure no missing values

df.isnull().sum()

# ## Exploration  <a name="exploration"></a>

# ### Extract words from readmes for top 5 languages and "Other"

top_six = df.lang_grouped.value_counts().index
words_by_lang = {}
for lang in top_six:
    words_by_lang[lang] = " ".join(df[df.lang_grouped == lang].stemmed)

# **Series of all words and their frequencies**

words_by_freq = " ".join(df.stemmed)
words_by_freq = pd.Series(words_by_freq.split()).value_counts()
print("Top 5 most common words")
words_by_freq.head()

# **Dictionary of top 5 languages + "other" and the frequency of their words**

word_freq_by_lang = {
    lang: pd.Series(words.split()).value_counts()
    for lang, words in words_by_lang.items()
}
pprint(word_freq_by_lang)

# **Most frequent words overall and how they measure in top 5 languages + other**

# +
top_words = (
    pd.concat(
        [words_by_freq] + list(word_freq_by_lang.values()), axis=1, sort=True
    )
    .set_axis(["all"] + list(word_freq_by_lang.keys()), axis=1, inplace=False)
    .fillna(0)
    .apply(lambda s: s.astype(int))
)

top_words.sort_values(by="all", ascending=False).head(5)
# -

# **And least frequent**

top_words.sort_values(by="all", ascending=False).tail(5)

# **Most common users**

df.user_name.value_counts().head(5)

# **Top 5 words unique to top 5 languages**

unique_words_by_lang = pd.DataFrame()
for lang in top_words.drop(columns="all"):
    unique = top_words.drop(columns="all")[
        top_words[lang] == top_words.drop(columns=["all"]).sum(axis=1)
    ]
    unique_words_by_lang = pd.concat(
        [
            unique_words_by_lang,
            unique.sort_values(by=lang, ascending=False).head(5),
        ]
    )

unique_words_by_lang

# ### Visualizations

# +
lang_prob = top_words[["all"]].copy()
for lang in top_words.drop(columns="all"):
    lang_prob[f"p_{lang}"] = top_words[lang] / top_words["all"]

lang_prob.sort_values(by="all").tail(15).drop(columns="all").plot.barh(
    stacked=True, figsize=(12, 8)
)
plt.title("Probability of Language of Top 15 Most Common Words")
plt.show()
# -

# Unsurprisingly, the most common words are representatively spread out among the categories

# #### Word Cloud

for lang, words in words_by_lang.items():
    plt.figure(figsize=(12, 8))
    cloud = WordCloud(
        background_color="white", height=600, width=800
    ).generate(words)
    plt.title(lang)
    plt.axis("off")
    plt.imshow(cloud)

# **Conclusions**
#
# "project", "use' are common words among the languages, but other than these, the most common words among the languages are different.

# ### Bigrams

# **Most common bigrams and bar plot**

for lang, words in words_by_lang.items():
    bigrams = pd.Series(nltk.ngrams(words.split(), 2)).value_counts()
    print(f"{lang}\n{bigrams.head()}")

    # Bar plot the bigrams
    bigrams.sort_values().tail(10).plot.barh(
        color="pink", width=0.9, figsize=(10, 6)
    )

    plt.title(f"10 Most frequently occurring {lang} bigrams")
    plt.ylabel("Bigram")
    plt.xlabel("# Occurrences")

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = (
        bigrams.sort_values()
        .tail(10)
        .reset_index()["index"]
        .apply(lambda t: " ".join(t))
    )
    _ = plt.yticks(ticks, labels)
    plt.show()


for lang, words in words_by_lang.items():
    bigrams = pd.Series(nltk.ngrams(words.split(), 2)).value_counts()

    # word cloud
    data = {" ".join(k): v for k, v in bigrams.to_dict().items()}
    img = WordCloud(
        background_color="white", width=800, height=400
    ).generate_from_frequencies(data)
    plt.figure(figsize=(12, 8))
    plt.axis("off")
    plt.title(lang)
    plt.imshow(img)

# **Conclusions**
#
# There is overlap among the languages as to the most common bigrams. The brigrams may not be that helpful after all.

# ### Trigram

for lang, words in words_by_lang.items():
    trigrams = pd.Series(nltk.ngrams(words.split(), 3)).value_counts()
    print(f"{lang}\n{trigrams.head()}")

    # Bar plot the trigrams
    trigrams.sort_values().tail(10).plot.barh(
        color="pink", width=0.9, figsize=(10, 6)
    )

    plt.title(f"10 Most frequently occurring {lang} trigrams")
    plt.ylabel("Trigram")
    plt.xlabel("# Occurrences")

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = (
        trigrams.sort_values()
        .tail(10)
        .reset_index()["index"]
        .apply(lambda t: " ".join(t))
    )
    _ = plt.yticks(ticks, labels)
    plt.show()

for lang, words in words_by_lang.items():
    trigrams = pd.Series(nltk.ngrams(words.split(), 3)).value_counts()

    # word cloud
    data = {" ".join(k): v for k, v in trigrams.to_dict().items()}
    img = WordCloud(
        background_color="white", width=800, height=400
    ).generate_from_frequencies(data)
    plt.figure(figsize=(12, 8))
    plt.axis("off")
    plt.title(lang)
    plt.imshow(img)


# **Conclusion**
#
#
# While there is less overlap of the most common trigrams than bigrams, these appear to be mostly junk or unique to a specific repo. 

# ### Summarize Conclusions

# ## Modeling <a name="modeling"></a>

def confmatrix(y_actual, y_pred):
    df = pd.DataFrame(dict(actual=y_actual, predicted=y_pred))
    return pd.crosstab(df.predicted, df.actual)


# ### Train test split

print("before removal", len(df))
df = df[df.lang_grouped.isin(top_five)]
print("after removal", len(df))

df.lang_grouped.value_counts()

# **Clean gives better results than lemmatized or stemmed**

# +
# X_train, X_test, y_train, y_test = train_test_split(
#     df.lemmatized, df.lang_grouped, stratify=df.lang_grouped, test_size=0.2, random_state=123
# )

# X_train, X_test, y_train, y_test = train_test_split(
#     df.stemmed, df.lang_grouped, stratify=df.lang_grouped, test_size=0.2, random_state=123
# )

X_train, X_test, y_train, y_test = train_test_split(
    df.clean, df.lang_grouped, stratify=df.lang_grouped, test_size=0.2, random_state=123
)
# -

X_train.shape

X_train.head()

type(X_train)

# ### For ALL Words

# ### Calculate TF-IDF for each word

# +
tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(X_train)
df_tfidf = pd.DataFrame(train_tfidf.todense(), columns=tfidf.get_feature_names())

test_tfidf = tfidf.transform(X_test)
# -

train_tfidf.shape

with open("tfidf.obj", 'wb') as fp:
    pickle.dump(tfidf, fp)

# **Words with highest tf-idf**

df_tfidf.max().sort_values(ascending=False).head(10)

# **Words with lowest tf-idf**

df_tfidf.max().sort_values(ascending=False).tail(10)

# **What does the distribution look like?**

sns.distplot(train_tfidf.todense().flatten())


# ### KNN

def knnmodel(X_train, X_test, y_train, y_test, **kwargs):
    ks = range(1, 15)
    sse = []
    for k in ks:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_train)

        # inertia: Sum of squared distances of samples to their closest cluster center.
        sse.append(kmeans.inertia_)

    print(pd.DataFrame(dict(k=ks, sse=sse)))

    plt.plot(ks, sse, "bx-")
    plt.xlabel("k")
    plt.ylabel("SSE")
    plt.title("The Elbow Method showing the optimal k")
    plt.show()
    
    knn = KNeighborsClassifier(**kwargs)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_proba_train = knn.predict_proba(X_train)
    
    print("TRAIN")
    print()
    print(
        "Accuracy of KNN classifier on training set: {:.2f}".format(
        knn.score(X_train, y_train)
        )
    )
    print()
    confmatrix(y_train, y_pred_train)
    print()
    print(classification_report(y_train, y_pred_train))
    
    y_pred_test = knn.predict(X_test)
    y_pred_proba_test = knn.predict_proba(X_test)
    
    print("-" * 20)
    print()
    print("TEST")
    print()
    print(
        "Accuracy of KNN classifier on training set: {:.2f}".format(
        knn.score(X_test, y_test)
        )
    )
    print()
    confmatrix(y_test, y_pred_test)
    print()
    print(classification_report(y_test, y_pred_test))
    
    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.figure()
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.scatter(k_range, scores)
    # plt.xticks([0,5,10,15,20])
    
    return knn


knn = knnmodel(train_tfidf, test_tfidf, y_train, y_test,
    n_neighbors=6, weights="uniform")

with open("knnmodel.obj", 'wb') as fp:
    pickle.dump(knn, fp)


# ### Naive Bayes Model

def nbmodel(X_train, X_test, y_train, y_test, **kwargs):
    gnb = GaussianNB(**kwargs)
    gnb.fit(X_train, y_train)
    
    print("TRAIN")
    print()
    y_pred_train = gnb.predict(X_train)
    print(
        "Accuracy of GNB classifier on training set: {:.2f}".format(
            gnb.score(X_train, y_train)
        )
    )
    print()
    print(confmatrix(y_train, y_pred_train))
    print()
    print(classification_report(y_train, y_pred_train))
    
    print("-" * 20)
    print()
    print("TEST")
    print()
    y_pred_test = gnb.predict(X_test)
    print(
        "Accuracy of GNB classifier on training set: {:.2f}".format(
            gnb.score(X_test, y_test)
        )
    )
    print()
    print(confmatrix(y_test, y_pred_test))
    print()
    print(classification_report(y_test, y_pred_test))
    
    return gnb


nbmodel(train_tfidf.todense(), test_tfidf.todense(), y_train, y_test)


# ### Logistic Regression

def lrmodel(X_train, X_test, y_train, y_test, **kwargs):
    lm = LogisticRegression(**kwargs).fit(X_train, y_train)
    
    print("TRAIN")
    print()
    y_pred_train = lm.predict(X_train)
    print(
        "Accuracy of lm classifier on training set: {:.2f}".format(
            accuracy_score(y_train, y_pred_train)
        )
    )
    print()
    print(confmatrix(y_train, y_pred_train))
    print()
    print(classification_report(y_train, y_pred_train))
    
    print("-" * 20)
    print()
    print("TEST")
    print()
    y_pred_test = lm.predict(X_test)
    print(
        "Accuracy of lm classifier on training set: {:.2f}".format(
            accuracy_score(y_test, y_pred_test)
        )
    )
    print()
    print(confmatrix(y_test, y_pred_test))
    print()
    print(classification_report(y_test, y_pred_test))
    
    return lm


lrmodel(train_tfidf, test_tfidf, y_train, y_test, random_state=123,
    solver="newton-cg",
    multi_class="multinomial",
    class_weight="balanced")

# ### Decision Tree

clf = DecisionTreeClassifier(
    criterion="entropy", max_depth=20, random_state=123, class_weight="balanced"
)

clf.fit(train_tfidf, y_train)

y_pred = clf.predict(train_tfidf)
y_pred[0:5]

y_pred_proba = clf.predict_proba(train_tfidf)
# y_pred_proba

# ### Computing the accuracy of our model

print(
    "Accuracy of Decision Tree classifier on training set: {:.2f}".format(
        clf.score(train_tfidf, y_train)
    )
)

confmatrix(y_train, y_pred)

print(classification_report(y_train, y_pred))

print(
    "Accuracy of Decision Tree classifier on test set: {:.2f}".format(
        clf.score(test_tfidf, y_test)
    )
)


# ### Random Forest

def rfmodel(X_train, X_test, y_train, y_test, **kwargs):
    clf = RandomForestClassifier(**kwargs).fit(X_train, y_train)
    
#     print("Feature Importances:")
#     print(clf.feature_importances_)
    print()
    print("TRAIN")
    print()
    y_pred_train = clf.predict(X_train)
    print(
        "Accuracy of clf classifier on training set: {:.2f}".format(
            accuracy_score(y_train, y_pred_train)
        )
    )
    print()
    print(confmatrix(y_train, y_pred_train))
    print()
    print(classification_report(y_train, y_pred_train))
    
    print("-" * 20)
    print()
    print("TEST")
    print()
    y_pred_test = clf.predict(X_test)
    print(
        "Accuracy of clf classifier on training set: {:.2f}".format(
            accuracy_score(y_test, y_pred_test)
        )
    )
    print()
    print(confmatrix(y_test, y_pred_test))
    print()
    print(classification_report(y_test, y_pred_test))
    
    return clf


rfmodel(train_tfidf, test_tfidf, y_train, y_test,
    n_estimators=100,
    max_depth=10,
    random_state=123,
    class_weight="balanced",
)

# ### Excluding frequent words

# ### Calculate TF-IDF for each word

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.05)
train_tfidf = tfidf.fit_transform(X_train)
df_tfidf = pd.DataFrame(train_tfidf.todense(), columns=tfidf.get_feature_names())

test_tfidf = tfidf.transform(X_test)
# -

train_tfidf.shape

# **Words with highest tf-idf**

df_tfidf.max().sort_values(ascending=False).head(10)

# **Words with lowest tf-idf**

df_tfidf.max().sort_values(ascending=False).tail(10)

# **What does the distribution look like?**

sns.distplot(train_tfidf.todense().flatten())

lrmodel(train_tfidf, test_tfidf, y_train, y_test, random_state=123,
    solver="newton-cg",
    multi_class="multinomial",
    class_weight="balanced")

rfmodel(train_tfidf, test_tfidf, y_train, y_test,
    n_estimators=1000,
    min_samples_leaf=3,
    max_depth=20,
    random_state=123,
    class_weight="balanced",
)

# ### Excluding least frequent words

# ### Calculate TF-IDF for each word

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=0.01)
train_tfidf = tfidf.fit_transform(X_train)
df_tfidf = pd.DataFrame(train_tfidf.todense(), columns=tfidf.get_feature_names())

test_tfidf = tfidf.transform(X_test)
# -

train_tfidf.shape

# **Words with highest tf-idf**

df_tfidf.max().sort_values(ascending=False).head(10)

# **Words with lowest tf-idf**

df_tfidf.max().sort_values(ascending=False).tail(10)

# **What does the distribution look like?**

sns.distplot(train_tfidf.todense().flatten())

lm = lrmodel(train_tfidf, test_tfidf, y_train, y_test, random_state=123,
    solver="newton-cg",
    multi_class="multinomial",
    class_weight="balanced")

rfmodel(train_tfidf, test_tfidf, y_train, y_test,
    n_estimators=1000,
    min_samples_leaf=3,
    max_depth=20,
    random_state=123,
    class_weight="balanced",
)

# ### Excluding most and least frequent words

# ### Calculate TF-IDF for each word

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=0.01, max_df=0.05)
train_tfidf = tfidf.fit_transform(X_train)
df_tfidf = pd.DataFrame(train_tfidf.todense(), columns=tfidf.get_feature_names())

test_tfidf = tfidf.transform(X_test)
# -

train_tfidf.shape

# **Words with highest tf-idf**

df_tfidf.max().sort_values(ascending=False).head(10)

# **Words with lowest tf-idf**

df_tfidf.max().sort_values(ascending=False).tail(10)

# **What does the distribution look like?**

sns.distplot(train_tfidf.todense().flatten())

lrmodel(train_tfidf, test_tfidf, y_train, y_test, random_state=123,
    solver="newton-cg",
    multi_class="multinomial",
    class_weight="balanced")

rfmodel(train_tfidf, test_tfidf, y_train, y_test,
    n_estimators=100,
    max_depth=20,
    random_state=123,
    class_weight="balanced",
)

# **Conclusions**
#
# Results are very mixed when excluding the most and/or least frequent words. Sometimes the results of the model improve and sometimes they get worse.

# ### Using Bigrams as features

# ### TF-IDF

TOP_NBIGRAMS = 5_000
# top_nwords = top_words.sort_values(by="all", ascending=False).head(500)
# top_nwords.index.values

# ### Calculate TF-IDF for each word

# +
tfidf = TfidfVectorizer(
    strip_accents="unicode", max_features=TOP_NBIGRAMS, ngram_range=(2, 2)
)
# tfidf = TfidfVectorizer(
#     strip_accents="unicode", ngram_range=(2, 2)
# )
train_tfidf = tfidf.fit_transform(X_train)
df_tfidf = pd.DataFrame(train_tfidf.todense(), columns=tfidf.get_feature_names())

test_tfidf = tfidf.transform(X_test)
# -

train_tfidf.shape

test_tfidf.shape

df_tfidf.max().sort_values(ascending=False).head(10)

df_tfidf.max().sort_values(ascending=False).tail(10)

# **What does the distribution look like?**

sns.distplot(train_tfidf.todense().flatten())

# ### Naive Bayes Model

nbmodel(train_tfidf.todense(), test_tfidf.todense(), y_train, y_test)

# ### Logistic Regression

lrmodel(train_tfidf, test_tfidf, y_train, y_test,
    random_state=123,
    solver="newton-cg",
    multi_class="multinomial",
    class_weight="balanced",
)

# ### Random Forest

rfmodel(train_tfidf, test_tfidf, y_train, y_test,
    n_estimators=100, max_depth=20, random_state=123, class_weight="balanced"
)

# ### Bag of Words

# +
vectorizer = CountVectorizer(max_features=750)
train_bow = vectorizer.fit_transform(X_train)
test_bow = vectorizer.transform(X_test)

df_bow = pd.DataFrame(train_bow.todense(), columns=vectorizer.get_feature_names())
# -

train_bow.shape

# **Most common wordsf**

df_bow.sum().sort_values(ascending=False).head(10)

# **Least common words**

df_bow.sum().sort_values(ascending=False).tail(10)

nbmodel(train_bow.todense(), test_bow.todense(), y_train, y_test)

lrmodel(train_bow, test_bow, y_train, y_test, random_state=123,
    solver="newton-cg",
    multi_class="multinomial",
    class_weight="balanced")

rfmodel(train_bow, test_bow, y_train, y_test,
    n_estimators=200, max_depth=10, random_state=123, class_weight="balanced"
)

# **Conclusions**
#
# For the most part, bag of words performs worse than TF-IDF, except for the random forest model.
