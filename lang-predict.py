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

# ### Prepare the Environment

# +
import os
import json
from pprint import pprint
import re
import unicodedata
from functools import reduce, partial
from copy import deepcopy

import requests
from bs4 import BeautifulSoup
import pandas as pd

import env

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

# -

# **Reload modules to capture changes**

# ## Acquisition <a name="acquisition"></a>

# **Grab data for 100 most forked repos on GitHub**

NUM_PER_PAGE = 100
API_URL = f"https://api.github.com/search/repositories?q=stars:%3E1&sort=forks&order=desc&per_page={NUM_PER_PAGE}"
HEADERS = {"Authorization": f"token {env.oauth_token}"}


def github_api_req():
    data = requests.get(API_URL, headers=HEADERS).json()
    return data["items"]


# **Extract necessary information**

# +
def readme_url(contents_url):
    # find name of README file and construct a link to the raw text of the readme
    for file in requests.get(contents_url, headers=HEADERS).json():
        if file["name"].lower().startswith("readme"):
            return file["html_url"]


REPO_FILE_NAME = "repos.json"


def load_repo_metadata(use_cache=True):
    if use_cache and os.path.exists(REPO_FILE_NAME):
        with open(REPO_FILE_NAME, "r") as f:
            contents = json.load(f)
        return contents
    else:
        response = github_api_req()
        for repo in response:
            # get link to contents of repo
            contents_url = repo["contents_url"][
                :-8
            ]  # remove last 8 characters to get working URL

            # find name of README file and construct a link to the raw text of the readme
            rmurl = readme_url(contents_url)

            # download README text
            readme_text = requests.get(rmurl, headers=HEADERS).text

            repo["readme"] = readme_text
        with open(REPO_FILE_NAME, "w") as f:
            json.dump(response, f)
        return response


all_repo_data = load_repo_metadata()
# -

all_repo_data[:3]

# ### Junk Code

# +

# URL_FMT = "https://github.com/search?o=desc&p={}&q=stars%3A%3E1&s=forks&type=Repositories"

# max_page = data["payload"]["max_page"]
# entries = []
# for i in range(1, max_page + 1):
#     url = root_endpoint + path + f"?page={i}"
#     data = requests.get(url).json()
#     entries += data["payload"][table]

# return pd.DataFrame(entries)
# example: "https://raw.githubusercontent.com/jtleek/datasharing/master/README.md"

# read links to repositories from page
# repos = []
# for page_num in range(1, 11):
#     page_url = URL_FMT.format(page_num)
#     html = requests.get(page_url)
#     soup = BeautifulSoup(html.content, "html.parser")
#     # print(soup)
#     print(type(soup))
#     repos += soup.find_all("a", class_="v-align-middle")

# print(list(repo.text for repo in repos))


# sample = api_resp[2]
# # store the id, username, name of repo
# user_id = sample["id"]
# user_name = sample["owner"]["login"]
# repo_name = sample["name"]

# print(user_id, user_name, repo_name)

# # find the predominant programming language
# lang = sample["language"]

# print(lang)

# # find the name of the README file (the capitalization may be off on some)
# contents_url = sample["contents_url"][:-8]  # remove last 8 characters to get working URL
# print(contents_url)

# # find name of README file and construct a link to the raw text of the readme
# readme_url = None
# for file in requests.get(contents_url).json():
#     if file["name"].lower().startswith("readme"):
#         readme_url = file["download_url"]

# print(readme_url)

# # download the readme
# print(requests.get(readme_url).text)

# -

# ## Preparation <a name="preparation"></a>

# +
def all_repo_metadata(api_data):
    return [repo_metadata(repo) for repo in api_data]


def repo_metadata(api_dict):
    # store the id, username, name of repo
    repo_id = api_dict["id"]
    user_name = api_dict["owner"]["login"]
    repo_name = api_dict["name"]

    # find the predominant programming language
    lang = api_dict["language"]

    # find README text
    soup = BeautifulSoup(api_dict["readme"], "html.parser")
    readme_text = soup.find("div", class_="Box mt-3 position-relative").text
    readme_text = readme_text[readme_text.find("History") + 7 :]

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

# + {"endofcell": "--"}
# # +
# right to left
def compose(*fns):
    return partial(reduce, lambda x, f: f(x), reversed(fns))


# applies in the order supplied
def pipe(v, *fns):
    return reduce(lambda x, f: f(x), fns, v)


def map_exhaust(func, *iters):
    for args in zip(*iters):
        func(*args)


# # +
def normalize_text(text):
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )


def remove_chars(text):
    return re.sub(r"[^A-Za-z0-9\s]", "", text)


def basic_clean(text):
    return pipe(text, str.lower, normalize_text, remove_chars)


# -


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
        copy["readme"], basic_clean, tokenize, remove_stopwords
    )

    copy["stemmed"] = stem(copy["clean"])

    copy["lemmatized"] = lemmatize(copy["clean"])

    return copy


def prep_readme_data(all_repo_data):
    return [prep_readme(repo) for repo in all_repo_data]


df = pd.DataFrame(prep_readme_data(some_repo_data))
# --

# ### Summarize Data

df.head()

df.describe(include="all")

# ### Fill NaNs with "None"

df.isna().sum()

df = df.fillna("None")

df.isna().sum()

# ### Check Missing Values

# ## Exploration  <a name="exploration"></a>

langs = pd.concat(
    [df.lang.value_counts(), df.lang.value_counts(normalize=True)], axis=1
)
langs.columns = ["n", "percent"]
langs

# ### Extract words from readmes for each language

# +
top_five = langs[:5].index
pprint(top_five)

langs_words = {}
for lang in top_five:
    langs_words[lang] = " ".join(df[df.lang == lang].lemmatized)
pprint(langs_words)
# -

all_words = " ".join(df.lemmatized)

lang_freqs = {
    lang: pd.Series(readme.split()).value_counts()
    for lang, readme in langs_words.items()
}
pprint(lang_freqs)

lang_list = lang_freqs.values()
lang_list

# +
# all_freqs = (
#     pd.concat([all_words].append(lang_freqs.values()), axis=1, sort=True)
#     .set_axis(["all"].append(lang_freqs.keys()), axis=1, inplace=False)
#     .fillna(0)
#     .apply(lambda s: s.astype(int))
# )

# all_freqs.head()
# -

# ### Train-Test Split
df.user_name.duplicated().sum()

df.user_name.value_counts().head(8)

df.lang.value_counts()

labels = pd.concat(
    [df.lang.value_counts(), df.lang.value_counts(normalize=True)], axis=1
)
labels.columns = ["n", "percent"]
labels

javascript_words = df[df.lang == "JavaScript"].clean
none_words = df[df.lang == "None"].clean
python_words = df[df.lang == "Python"].clean
java_words = df[df.lang == "Java"].clean
html_words = df[df.lang == "HTML"].clean
all_words = df.clean

none_words

word_counts = pd.DataFrame(lang_freqs)

word_counts.head(10)

word_counts.fillna(0, inplace=True)

word_counts.head()

word_counts.dtypes

word_counts["all"] = word_counts.sum(axis=1)

word_counts.sort_values(by="all", ascending=False).head(10)

pd.concat(
    [
        word_counts[word_counts.JavaScript == 0]
        .sort_values(by="Python")
        .tail(6),
        word_counts[word_counts.Python == 0]
        .sort_values(by="JavaScript")
        .tail(6),
    ]
)


# ### Visualizations

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# +
# figure out the percentage
(
    word_counts.assign(
        p_javascript=word_counts.JavaScript / word_counts["all"],
        p_none=word_counts["None"] / word_counts["all"],
        p_python=word_counts.Python / word_counts["all"],
        p_java=word_counts.Java / word_counts["all"],
        p_html=word_counts.HTML / word_counts["all"],
    )
    .sort_values(by="all")[
        ["p_javascript", "p_none", "p_python", "p_java", "p_html"]
    ]
    .tail(20)
    .sort_values("p_java")
    .plot.barh(stacked=True)
)

plt.title("Proportion of Spam vs Ham for the 20 most common words")
# -

(
    word_counts[
        (word_counts.JavaScript > 5)
        & (word_counts.Python > 5)
        & (word_counts["None"] > 5)
        & (word_counts.Java > 5)
        & (word_counts.HTML > 5)
    ]
    .assign(
        ratio=lambda df: df.JavaScript
        / (df.Python / df["None"] / df.Java / df.HTML + 0.01)
    )
    .sort_values(by="ratio")
    .pipe(lambda df: pd.concat([df.head(), df.tail()]))
)

# # Word Cloud!!

# +
from wordcloud import WordCloud


all_cloud = WordCloud(
    background_color="white", height=600, width=800
).generate(" ".join(all_words))
javascript_cloud = WordCloud(
    background_color="white", height=600, width=800
).generate(" ".join(javascript_words))
none_cloud = WordCloud(
    background_color="white", height=600, width=800
).generate(" ".join(none_words))
python_cloud = WordCloud(
    background_color="white", height=600, width=800
).generate(" ".join(python_words))
java_cloud = WordCloud(
    background_color="white", height=600, width=800
).generate(" ".join(java_words))
html_cloud = WordCloud(
    background_color="white", height=600, width=800
).generate(" ".join(html_words))

# plt.figure(figsize=(10, 8))
axs = [
    plt.axes([0, 0, 0.5, 1]),
    plt.axes([0.5, 0.5, 0.5, 0.5]),
    plt.axes([0.5, 0, 0.5, 0.5]),
    plt.axes([0.5, 0, 0.5, 0.5]),
    plt.axes([0.5, 0, 0.5, 0.5]),
    plt.axes([0.5, 0, 0.5, 0.5]),
]

axs[0].imshow(all_cloud)
axs[1].imshow(javascript_cloud)
axs[2].imshow(none_cloud)
axs[3].imshow(python_cloud)
axs[4].imshow(java_cloud)
axs[5].imshow(html_cloud)

axs[0].set_title("All Words")
axs[1].set_title("JavaScript")
axs[2].set_title("None")
axs[3].set_title("Python")
axs[4].set_title("Java")
axs[5].set_title("HTML")

for ax in axs:
    ax.axis("off")


# -

# # Biograms

# +
top_20_ham_bigrams = (
    pd.Series(nltk.ngrams(javascript_words, 2)).value_counts().head(20)
)

top_20_ham_bigrams.head()
# -


# ### Statistical Tests

# ### Summarize Conclusions

# ## Modeling <a name="modeling"></a>

# ### Feature Engineering & Selection

# ### Train & Test Models

# ### Summarize Conclusions
