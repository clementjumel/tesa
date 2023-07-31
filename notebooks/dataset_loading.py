# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Setup
# ### Imports

# %%
import sys
sys.path.append('../')
del sys

# %reload_ext autoreload
# %autoreload 2

import pandas as pd
import pickle
from tesa._path import _DATA

# %% [markdown]
# ### Load the dataset

# %% [markdown]
# #### dataset.csv

# %%
df = pd.read_csv(_DATA / "results" / "publication"/ "dataset.csv")
df

# %% [markdown]
# #### dataset.pickle

# %%
data = pickle.load(open(_DATA / "results" / "publication" / "dataset.pickle", "rb"))

# %% [markdown]
# Display the first "row":

# %%
for d in data[:1]:
    for key, value in d.items():
        if isinstance(value, list):
            value = [str(v) if isinstance(v, int) else v for v in value]
            value = ", ".join(value)
        elif isinstance(value, dict):
            value = "\n".join([f"{k}: {v}" for k, v in value.items()])
            
        print(f"{key}:\n{value}\n")

# %% [markdown]
# ### Load the task

# %% [markdown]
# #### ranking_task_train.csv

# %%
df = pd.read_csv(_DATA / "results" / "publication" / "ranking_task_train.csv")
df

# %% [markdown]
# #### ranking_task_valid.csv

# %%
df = pd.read_csv(_DATA / "results" / "publication" / "ranking_task_valid.csv")
df

# %% [markdown]
# #### ranking_task_test.csv

# %%
df = pd.read_csv(_DATA / "results" / "publication" / "ranking_task_test.csv")
df

# %% [markdown]
# #### ranking_task_train.pickle

# %%
data = pickle.load(open(_DATA / "results" / "publication" / "ranking_task_train.pickle", "rb"))

# %% [markdown]
# Display the first "row":

# %%
for d in data[:1]:
    for key, value in d.items():
        if isinstance(value, list):
            value = [str(v) if isinstance(v, int) else v for v in value]
            value = ", ".join(value)
        elif isinstance(value, dict):
            value = "\n".join([f"{k}: {v}" for k, v in value.items()])
            
        print(f"{key}:\n{value}\n")

# %% [markdown]
# #### ranking_task_valid.pickle

# %%
data = pickle.load(open(_DATA / "results" / "publication" / "ranking_task_valid.pickle", "rb"))

# %% [markdown]
# Display the first "row":

# %%
for d in data[:1]:
    for key, value in d.items():
        if isinstance(value, list):
            value = [str(v) if isinstance(v, int) else v for v in value]
            value = ", ".join(value)
        elif isinstance(value, dict):
            value = "\n".join([f"{k}: {v}" for k, v in value.items()])
            
        print(f"{key}:\n{value}\n")

# %% [markdown]
# #### ranking_task_test.pickle

# %%
data = pickle.load(open(_DATA / "results" / "publication" / "ranking_task_test.pickle", "rb"))

# %% [markdown]
# Display the first "row":

# %%
for d in data[:1]:
    for key, value in d.items():
        if isinstance(value, list):
            value = [str(v) if isinstance(v, int) else v for v in value]
            value = ", ".join(value)
        elif isinstance(value, dict):
            value = "\n".join([f"{k}: {v}" for k, v in value.items()])
            
        print(f"{key}:\n{value}\n")

# %%
