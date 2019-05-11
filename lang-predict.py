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

# # Title
# By

# ## TODO
#
# - [ ] Acquisition
#     - [ ] Select what list of repos to scrape.
#     - [ ] Get requests form the site.
#     - [ ] Save responses to csv.
# - [ ] Preparation
#     - [ ] Prepare the data for analysis.
# - [ ] Exploration
#     - [ ] Answer the following prompts:
#         - [ ] What are the most common words in READMEs?
#         - [ ] What does the distribution of IDFs look like for the most common words?
#         - [ ] Does the length of the README vary by language?
#         - [ ] Do different languages use a different number of unique words?
# - [ ] Modeling
#     - [ ] Transform the data for machine learning; use language to predict.
#     - [ ] Fit several models using different text repressentations.
#     - [ ] Build a function that will take in the text of a README file, and makes a prediction of language.
# - [ ] Delivery
#     - [ ] Github repo
#         - [x] This notebook.
#         - [ ] Documentation within the notebook.
#         - [ ] README file in the repo.
#         - [ ] Python scripts if applicable.
#     - [ ] Google Slides
#         - [ ] 1-2 slides only summarizing analysis.
#         - [ ] Visualizations are labeled.
#         - [ ] Geared for the general audience.
#         - [ ] Share link @ readme file and/or classroom.

# ## Table of contents
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

# ### Data Dictionary & Domain Knowledge

# ### Hypotheses
# * I expect to see JavaScript and Python as the two most common languages used based on current popularity
# * Using symbols and characters unique to certain languages, we might be able to more accurately predict language used
# * The primary language may be mentioned in the README. But some repositories mention multiple languages, so this may interfere with this method.
# * The number of words may be an indication of the language. Older repositories are probably written in certain languages and because of their age, may have more documentation.
# * We may be able to classify some repositories based on the operators used in the README. We may want to create a mapping of operators to languages. Then we can limit the potential languages using this and look at which language has the highest probability
# * The READMEs contain sample code to download the library or use the program. This may be an indicator of the language. We can create a list of potential commands and map them to language.
# * The sentiment score of the README may be indicative of the language

# ### Thoughts & Questions
#
# * The code in many repositories are written in multiple languages.
# * Should we remove all words not in an English dictionary?
# * Try random sampling of the repos. How many would we need for a reliable result?

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
# -

# **Reload modules to capture changes**

# ## Acquisition <a name="acquisition"></a>

# **Grab data for 500 most forked repos on GitHub**

# **Constants**

NUM_PER_PAGE = 100
PAGES = 5
API_URL = f"https://api.github.com/search/repositories?q=stars:%3E1&sort=forks&order=desc&per_page={NUM_PER_PAGE}"
HEADERS = {"Authorization": f"token {env.oauth_token}"}
REPO_FILE_NAME = "repos.json"


# **Extract necessary information**

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

            # find name of README file and construct a link to the raw text of the readme
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
    return [repo_metadata(repo) for repo in api_data if repo["readme"] is not None]


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
        copy["readme"], basic_clean, tokenize, remove_stopwords, remove_bogus_words
    )

    copy["stemmed"] = stem(copy["clean"])

    copy["lemmatized"] = lemmatize(copy["clean"])

    return copy


def prep_readme_data(all_repo_data):
    return [prep_readme(repo) for repo in all_repo_data]


df = pd.DataFrame(prep_readme_data(some_repo_data))
# -

# common single letter words?

df[df.clean.str.contains(" j ")]

# any links?

df[df.clean.str.contains(" http")]

# ### Summarize Data

df.head()

df.describe(include="all")

# ### Check Missing Values

# #### Fill NaNs with "None" (i.e., they have no programming language)

df.isna().sum()

df = df.fillna("None")

df.isna().sum()

# #### Most common languages

langs = pd.concat(
    [df.lang.value_counts(), df.lang.value_counts(normalize=True)], axis=1
)
langs.columns = ["n", "percent"]
langs

# **Go with top 5 languages and roll the rest into an "other" category**

# +
top_five = list(langs[:5].index)
pprint(top_five)

lang_grouped = df.lang.apply(lambda lang: lang if lang in top_five else "Other").rename("lang_grouped")

# pprint(lang_grouped)
df = pd.concat([df, lang_grouped], axis=1)
# -

df.lang_grouped.value_counts()

# ## Exploration  <a name="exploration"></a>

# ### Extract words from readmes for top 5 languages (which includes "None") and "Other"

# +
# words_by_lang = {}
# for lang in top_five:
#     words_by_lang[lang] = " ".join(df[df.lang == lang].lemmatized)
# pprint(words_by_lang)
# -

top_six = df.lang_grouped.value_counts().index
words_by_lang = {}
for lang in top_six:
    words_by_lang[lang] = " ".join(df[df.lang_grouped == lang].lemmatized)

# **Series of all words and their frequencies**

words_by_freq = " ".join(df.lemmatized)
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
    pd.concat([words_by_freq] + list(word_freq_by_lang.values()), axis=1, sort=True)
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

# **Top 5 word unique to top 5 languages**

unique_words_by_lang = pd.DataFrame()
for lang in top_words.drop(columns="all"):
    unique = top_words.drop(columns="all")[top_words[lang] == top_words.drop(columns=["all"]).sum(axis=1)]
    unique_words_by_lang = pd.concat([unique_words_by_lang, unique.sort_values(by=lang, ascending=False).head(5)])

unique_words_by_lang

# ### Visualizations

# +
lang_prob = top_words[["all"]].copy()
for lang in top_words.drop(columns="all"):
    lang_prob[f"p_{lang}"] = top_words[lang] / top_words["all"]

lang_prob["p_other"] = 1.0 - lang_prob.drop(columns="all").sum(axis=1)

lang_prob.sort_values(by="all").tail(15).drop(columns="all").plot.barh(stacked=True, figsize=(12, 8))
plt.title("Probability of Language of Top 15 Most Common Words")
plt.show()
# -

# #### Word Cloud

for lang, words in words_by_lang.items():
    plt.figure(figsize=(12, 8))
    cloud = WordCloud(background_color='white', height=600, width=800).generate(words)
    plt.title(lang)
    plt.axis("off")
    plt.imshow(cloud)

# ### Bigrams

# **Most common bigrams and bar plot**

for lang, words in words_by_lang.items():
    bigrams = pd.Series(nltk.ngrams(words.split(), 2)).value_counts()
    print(f"{lang}\n{bigrams.head()}")
    
    # Bar plot the bigrams
    bigrams.sort_values().tail(10).plot.barh(color='pink', width=.9, figsize=(10, 6))

    plt.title(f'10 Most frequently occurring {lang} bigrams')
    plt.ylabel('Bigram')
    plt.xlabel('# Occurrences')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = bigrams.sort_values().tail(10).reset_index()['index'].apply(lambda t: " ".join(t))
    _ = plt.yticks(ticks, labels)
    plt.show()


for lang, words in words_by_lang.items():
    bigrams = pd.Series(nltk.ngrams(words.split(), 2)).value_counts()
    
    # word cloud
    data = {" ".join(k): v for k, v in bigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.title(lang)
    plt.imshow(img)

# ### Trigram

for lang, words in words_by_lang.items():
    trigrams = pd.Series(nltk.ngrams(words.split(), 3)).value_counts()
    print(f"{lang}\n{trigrams.head()}")
    
    # Bar plot the trigrams
    trigrams.sort_values().tail(10).plot.barh(color='pink', width=.9, figsize=(10, 6))

    plt.title(f'10 Most frequently occurring {lang} trigrams')
    plt.ylabel('Trigram')
    plt.xlabel('# Occurrences')

    # make the labels pretty
    ticks, _ = plt.yticks()
    labels = trigrams.sort_values().tail(10).reset_index()['index'].apply(lambda t: " ".join(t))
    _ = plt.yticks(ticks, labels)
    plt.show()

for lang, words in words_by_lang.items():
    trigrams = pd.Series(nltk.ngrams(words.split(), 3)).value_counts()
    
    # word cloud
    data = {" ".join(k): v for k, v in trigrams.to_dict().items()}
    img = WordCloud(background_color='white', width=800, height=400).generate_from_frequencies(data)
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.title(lang)
    plt.imshow(img)

# **Conclusion**
#
# Trigrams are not that helpful. They appear to be mostly junk or unique to a specific repo.

# ### Statistical Tests

# ### Summarize Conclusions

# ## Modeling <a name="modeling"></a>

# **Calculate IDF for each word**

# +
# # This is not working and is taking way too long.
# def idf(word):
#     n_occurences = sum([1 for index, row in df.iterrows() if word in row.lemmatized])
#     return len(df) / n_occurences

# # put the unique words into a data frame
# (pd.DataFrame(dict(word=words_by_freq.index))
#  # calculate the idf for each word
#  .assign(idf=lambda df: df.word.apply(idf))
#  # sort the data for presentation purposes
#  .set_index('word')
#  .sort_values(by='idf', ascending=False))
# -

# The words whose IDF equals the number of documents occur in only one document.

# ### Calculate TF-IDF for each word

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidfs = tfidf.fit_transform(df.lemmatized)

# -

# ### Train-test split

y = df.lang_grouped
X = pd.DataFrame(tfidfs.todense(), columns=tfidf.get_feature_names())
# X = tfidfs.todense()
# df_tfidfs.head()
X.shape

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.2, random_state=123)

train = pd.DataFrame(dict(actual=y_train))
test = pd.DataFrame(dict(actual=y_test))
# -

X_train.shape

type(X_train)

# **What does the distribution look like?**

sns.distplot(X_train.values.flatten())

# ### Naive Bayes Model

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_train_pred = gnb.predict(X_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

y_test_pred = gnb.predict(X_test)
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# ### Logistic Regression

# +
lm = LogisticRegression(random_state=123, solver="newton-cg", multi_class="multinomial", class_weight="balanced").fit(X_train, y_train)

train['predicted'] = lm.predict(X_train)
test['predicted'] = lm.predict(X_test)
# -

print('Accuracy: {:.2%}'.format(accuracy_score(train.actual, train.predicted)))
print('---')
print('Confusion Matrix')
print(pd.crosstab(train.predicted, train.actual))
print('---')
print(classification_report(train.actual, train.predicted))

print('Accuracy: {:.2%}'.format(accuracy_score(test.actual, test.predicted)))
print('---')
print('Confusion Matrix')
print(pd.crosstab(test.predicted, test.actual))
print('---')
print(classification_report(test.actual, test.predicted))

# ### Random Forest

from sklearn.ensemble import RandomForestClassifier

# +
clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                             random_state=123, class_weight="balanced")
clf.fit(X_train, y_train)

print(clf.feature_importances_)
# -

train['predicted'] = clf.predict(X_train)
test['predicted'] = clf.predict(X_test)

print('Accuracy: {:.2%}'.format(accuracy_score(train.actual, train.predicted)))
print('---')
print('Confusion Matrix')
print(pd.crosstab(train.predicted, train.actual))
print('---')
print(classification_report(train.actual, train.predicted))

print('Accuracy: {:.2%}'.format(accuracy_score(test.actual, test.predicted)))
print('---')
print('Confusion Matrix')
print(pd.crosstab(test.predicted, test.actual))
print('---')
print(classification_report(test.actual, test.predicted))

# ### Feature Engineering & Selection

# ### Train & Test Models

# ### Summarize Conclusions


