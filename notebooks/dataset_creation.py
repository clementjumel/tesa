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

from tesa.toolbox.parsers import standard_parser, add_annotations_arguments, add_task_arguments
from tesa.database_creation.annotation_task import AnnotationTask
from tesa.preprocess_annotations import filter_annotations
from tesa.toolbox.utils import load_task
from tesa.modeling.utils import format_context
from os.path import join as path_join
import pandas as pd
import os
import pickle

# %% [markdown]
# ### Parameters

# %%
ap = standard_parser()
add_annotations_arguments(ap)
add_task_arguments(ap)
args = ap.parse_args(["--root", ".."])

# %% [markdown]
# ### Load the annotations data (and first preprocessing step)

# %%
annotation_task = AnnotationTask(silent=args.silent,
                                     results_path=path_join(args.root, args.annotations_path),
                                     years=None,
                                     max_tuple_size=None,
                                     short=None,
                                     short_size=None,
                                     random=None,
                                     debug=None,
                                     random_seed=None,
                                     save=None,
                                     corpus_path=None)

annotation_task.process_task(exclude_pilot=args.exclude_pilot)

queries = annotation_task.queries
annotations = annotation_task.annotations

# %%
global_data = []
for id_, annotation_list in annotations.items():
    data = dict()
    
    query = queries[id_]
    data["entities_type"] = query.entities_type_
    data["entities"] = query.entities
    data["summaries"] = query.summaries
    data["urls"] = query.urls
    data["title"] = query.title
    data["date"] = query.date
    data["context"] = query.context
    data["context_type"] = query.context_type_
    
    for i, annotation in enumerate(annotation_list):
        if annotation.answers:
            data[f"answer_{i}"] = annotation.answers
            
    global_data.append(data)
    
df = pd.DataFrame(global_data)[["entities_type","entities","answer_0","answer_1","answer_2","title","date","urls","summaries","context_type","context"]]

# %%
df.to_csv("../results/publication/dataset.csv", index=False, mode="w")
pickle.dump(global_data, open("../results/publication/dataset.pickle", "wb"))

# %% [markdown]
# ### Load the modeling task

# %%
task = load_task(args)

# %%
for split in ["train", "valid", "test"]:
    loader = getattr(task, f"{split}_loader")
    global_data = []
    
    for ranking_task in loader:
        data = dict()
        
        first_input = ranking_task[0][0]
        data["entities"] = first_input["entities"]
        data["entities_type"] = first_input["entities_type"]
        data["wiki_articles"] = first_input["wiki_articles"]
        assert len(first_input["nyt_titles"]) == 1
        assert len(first_input["nyt_contexts"]) == 1
        data["nyt_title"] = first_input["nyt_titles"][0]
        data["nyt_context"] = first_input["nyt_contexts"][0]
        
        candidates = []
        labels = []
        
        for batch_inputs, batch_outputs in ranking_task:
            candidates.extend(batch_inputs["choices"])
            labels.extend(list(batch_outputs.tolist()))
            
        data["candidates"] = candidates
        data["labels"] = labels
        
        global_data.append(data)

    df = pd.DataFrame(global_data)[["entities_type","entities","wiki_articles","nyt_title","nyt_context","candidates","labels"]]
    
    df.to_csv(f"../results/publication/ranking_task_{split}.csv", index=False, mode="w")
    pickle.dump(global_data, open(f"../results/publication/ranking_task_{split}.pickle", "wb"))

# %%
