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

# %% pycharm={"is_executing": false}
import sys
sys.path.append('../')
del sys

# %reload_ext autoreload
# %autoreload 2

from tesa.toolbox.parsers import standard_parser, add_task_arguments, add_model_arguments
from tesa.toolbox.utils import load_task, get_pretrained_model, to_class_name
import tesa.modeling.models as models

# %% [markdown]
# ### Notebook functions

# %%
from numpy import argmax, mean

def run_models(model_names, word2vec, bart, args, train=False):
    args.word2vec = word2vec
    args.bart = bart
    
    pretrained_model = get_pretrained_model(args)
    
    for model_name in model_names:
        args.model = model_name
        print(model_name)

        model = getattr(models, to_class_name(args.model))(args=args, pretrained_model=pretrained_model)
        model.play(task=task, args=args)
        
        if train:
            valid_scores = model.valid_scores['average_precision']
            test_scores = model.test_scores['average_precision']

            valid_scores = [mean(epoch_scores) for epoch_scores in valid_scores]
            test_scores = [mean(epoch_scores) for epoch_scores in test_scores]

            i_max = argmax(valid_scores)
            print("max for epoch %i" % (i_max+1))
            print("valid score: %.5f" % valid_scores[i_max])
            print("test score: %.5f" % test_scores[i_max])


# %% [markdown]
# ### Parameters

# %%
ap = standard_parser()
add_task_arguments(ap)
add_model_arguments(ap)
args = ap.parse_args(["-m", "",
                      "--root", ".."])

# %% [markdown]
# ### Load the data

# %%
task = load_task(args)

# %% [markdown]
# # Basic baselines

# %%
run_models(model_names=["random",
                        "frequency"],
           word2vec=False,
           bart=False,
           args=args)

# %% [markdown]
# # Basic baselines

# %%
run_models(model_names=["summaries-count",
                        "summaries-unique-count",
                        "summaries-overlap",
                        "activated-summaries",
                        "context-count",
                        "context-unique-count",
                        "summaries-context-count",
                        "summaries-context-unique-count",
                        "summaries-context-overlap"],
           word2vec=False,
           bart=False,
           args=args)

# %% [markdown]
# # Embedding baselines

# %%
run_models(model_names=["summaries-average-embedding",
                        "summaries-overlap-average-embedding",
                        "context-average-embedding",
                        "summaries-context-average-embedding",
                        "summaries-context-overlap-average-embedding"],
           word2vec=True,
           bart=False,
           args=args)

# %% [markdown]
# ### Custom classifier

# %%
run_models(model_names=["custom-classifier"],
           word2vec=True,
           bart=False,
           args=args,
           train=True)

# %%
