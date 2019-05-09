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

import requests
from bs4 import BeautifulSoup
import pandas as pd

import env
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
            return file["download_url"]

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
            contents_url = repo["contents_url"][:-8]  # remove last 8 characters to get working URL

            # find name of README file and construct a link to the raw text of the readme
            rmurl = readme_url(contents_url)

            # download README text
            readme_text = requests.get(rmurl, headers=HEADERS).text
            
            repo["readme"] = readme_text
        with open(REPO_FILE_NAME, "w") as f:
            json.dump(response, f)
        return response

repo_data = load_repo_metadata()
# -

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
    return (repo_metadata(repo) for repo in api_data)


def repo_metadata(api_dict):
    # store the id, username, name of repo
    repo_id = api_dict["id"]
    user_name = api_dict["owner"]["login"]
    repo_name = api_dict["name"]
    
    # find the predominant programming language
    lang = api_dict["language"]
    
    # find README text
    readme_text = api_dict["readme"]
    
    return dict(repo_id=repo_id, user_name=user_name, repo_name=repo_name, lang=lang, readme=readme_text)

df_orig = pd.DataFrame(all_repo_metadata(repo_data))
# -

df_orig.head()

# ### Summarize Data

# ### Handle Missing Values

# ### Handle Duplicates

# ### Fix Data Types

# ### Handle Outliers

# ### Check Missing Values

# ## Exploration  <a name="exploration"></a>

# ### Train-Test Split

# ### Visualizations

# ### Statistical Tests

# ### Summarize Conclusions

# ## Modeling <a name="modeling"></a>

# ### Feature Engineering & Selection

# ### Train & Test Models

# ### Summarize Conclusions
