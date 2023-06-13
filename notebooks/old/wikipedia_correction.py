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
# ## Load the wikipedia file

# %%
database.load_pkl(attribute_name='wikipedia', file_name='wikipedia_global', folder_name='wikipedia')

# %% [markdown]
# ## Correct the wikipedia file

# %% [markdown]
# Automatic correction of simple cases

# %%
database.correct_wiki(out_name='wikipedia_global', step=1)

# %% [markdown]
# Manual validation/discard of more complicated cases

# %%
database.correct_wiki(out_name='wikipedia_global', step=2)

# %% [markdown]
# Manual choice between possible answers

# %%
database.correct_wiki(out_name='wikipedia_global', step=3)

# %% [markdown]
# Discard all remaining entities as not found

# %%
database.correct_wiki(out_name='wikipedia_global', step=4)

# %%
