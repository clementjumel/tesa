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
# ## Load the data

# %% pycharm={"is_executing": false}
database.process_queries(load=True)

# %% [markdown]
# ## Observations

# %%
for query_id, query in database.queries.items():
    #if ('New York City' in [str(e) for e in query.entities] \
    #        or 'New York State' in [str(e) for e in query.entities]) \
    #        and len(query.entities) >= 2:
            
    #    print([str(e) for e in query.entities])
    if 'have not agreed on the need to impose sanctions on' in query.context:
        if [str(e) for e in query.entities] == ['Europe', 'Iran', 'Russia']:
            print(query.context)
    #if query.entities_names == 'Europe, Iran and Russia':
    #    print(query_id, query.context)

# %%
ids = ['1762937_13255_4_4']
for id_ in ids:
    for title, item in database.queries[id_].to_dict().items():
        print(title, ': ', item)

# %%
for tuple_ in database.tuples:
    print(str(tuple_))

# %%
for query_id, query in database.queries.items():
    #if "Moscow" in query.entities_names:
    #    print(query.entities_names)
    if query.entities_names == 'New York City and New York State':
        print(query_id, query.context)

# %%
ids = ['1801660_2_20_20']
for id_ in ids:
    for title, item in database.queries[id_].to_dict().items():
        print(title, ': ', item)

# %%
print(database.wikipedia['found']['New York State'])

# %%
