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
# ## Notebook setup

# %% pycharm={"is_executing": false}
import sys
sys.path.append('../')
del sys
# %reload_ext autoreload
# %autoreload 2

# %% pycharm={"is_executing": false}
from database_creation.database import Database

# %% [markdown]
# ## Parameters

# %% pycharm={"is_executing": false}
database = Database(max_size=10000)

# %% [markdown]
# ## Preprocessing the database

# %% pycharm={"is_executing": false}
database.preprocess_database(debug=True)

# %% [markdown]
# ## Preprocessing the articles

# %% pycharm={"is_executing": true}
database.process_articles(debug=True)

# %% [markdown]
# ## Processing the wikipedia information

# %% pycharm={"is_executing": true}
database.process_wikipedia(load=False, debug=True)

# %% [markdown]
# ## Gather the wikipedia files together

# %%
database.combine_wiki()

# %% [markdown]
# ## Correct the wikipedia file

# %%
database.correct_wiki(out_name='wikipedia_global', step=1)

# %%
database.correct_wiki(out_name='wikipedia_global', step=2)

# %%
database.correct_wiki(out_name='wikipedia_global', step=3)

# %%
database.correct_wiki(out_name='wikipedia_global', step=2)

# %%
database.correct_wiki(out_name='wikipedia_global', step=3)

# %%
database.correct_wiki(out_name='wikipedia_global', step=2)

# %%
database.correct_wiki(out_name='wikipedia_global', step=4)

# %% [markdown]
# ## Processing the queries

# %% pycharm={"is_executing": true}
database.process_queries(check_changes=False, debug=True, csv_seed=1)

# %%
