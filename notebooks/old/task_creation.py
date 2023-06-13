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
from tesa.database_creation.database import Database

# %% [markdown]
# ## Parameters

# %% pycharm={"is_executing": false}
database = Database()

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
database.process_wikipedia(load=True, file_name='wikipedia_global', debug=True)

# %% [markdown]
# ## Check the correction of the wikipedia file

# %%
database.correct_wiki(out_name='wikipedia_global', step=1)

# %% [markdown]
# ## Processing the queries

# %% pycharm={"is_executing": true}
database.process_queries(debug=True, csv_size=400, csv_seed=3, exclude_seen=True)

# %% [markdown]
# ## Creation of several batches

# %%
database.process_queries(load=True)

# %%
for i in range(3, 100):
    i_str = str(i) if i > 9 else '0' + str(i)
    database.save_csv(attribute_name='queries',
                      file_name='batch_'+i_str,
                      folder_name='task_answers/v2_1/task',
                      limit=400,
                      random_seed=i+1, 
                      exclude_seen=True)

# %%
