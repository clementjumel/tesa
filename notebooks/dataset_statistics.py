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

from toolbox.parsers import standard_parser, add_annotations_arguments
from database_creation.annotation_task import AnnotationTask
from preprocess_annotations import filter_annotations
from os.path import join as path_join

# %% [markdown]
# ### Parameters

# %%
ap = standard_parser()
add_annotations_arguments(ap)
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

# %% [markdown]
# We discarded 23 aggregation annotations in the first step.

# %%
print(len(annotations))
for id_, annotation_list in annotations.items():
    for annotation in annotation_list:
        print(annotation)
    break

# %%
from numpy import mean
l = []
for id_, annotation_list in annotations.items():
    query = queries[id_]    
    l.append(len(query.entities))
print(min(l))
print(mean(l))
print(max(l))

# %% [markdown]
# ### Number of annotators

# %%
ids=set()
for id_, annotation_list in annotations.items():
    for annotation in annotation_list:
        ids.add(annotation.worker_id)

print(len(ids))

# %% [markdown]
# ### 2nd and 3rd preprocessing steps

# %%
annotations = filter_annotations(annotations, args=args)

# %% [markdown]
# ### Number of annotators

# %%
ids=set()
for id_, annotation_list in annotations.items():
    for annotation in annotation_list:
        ids.add(annotation.worker_id)

print(len(ids))

# %% [markdown]
# ### Remaining data

# %%
from collections import defaultdict

to_del = []
for id_, annotations_list in annotations.items():
    annotations[id_] = [annotation for annotation in annotations_list if annotation.preprocessed_answers]
    
    if not annotations[id_]:
        to_del.append(id_)
        
for id_ in to_del:
    del annotations[id_]
    
length1 = sum([len([annotation for annotation in annotation_list if annotation.preprocessed_answers])
               for _, annotation_list in annotations.items()])
length2 = sum([len([annotation for annotation in annotation_list if not annotation.preprocessed_answers])
               for _, annotation_list in annotations.items()])

detailed_aggreg = defaultdict(list)
detailed_entities = defaultdict(list)
for id_, annotation_list in annotations.items():
    type_ = queries[id_].entities_type_
    entities = ', '.join(sorted(queries[id_].entities))
    detailed_entities['all'].append(entities)
    detailed_entities[type_].append(entities)
    
    for annotation in annotation_list:
        for aggregation in annotation.preprocessed_answers:
            detailed_aggreg['all'].append(aggregation)
            detailed_aggreg[type_].append(aggregation)

# %% [markdown]
# ### Table data

# %%
initial_aggreg_instances = 2100
initial_annotations = 4993+1306
initial_na = 1306-23
first_filter_discarded_answered_annotations = 23
second_filter_discarded_answered_annotations = 4993-4963
third_filter_discarded_answered_annotations = 4963-4675
final_aggregation_annotations = 4675

assert initial_annotations  - initial_na \
                            - first_filter_discarded_answered_annotations \
                            - second_filter_discarded_answered_annotations \
                            - third_filter_discarded_answered_annotations \
            == final_aggregation_annotations

print("Initial number of aggreg. instances: %i" % initial_aggreg_instances)
print("Initial number of annotations: %i" % initial_annotations)
print()
print("Initial number of n/a annotations: %i" % initial_na)
print("First filter discarded aggregation annotations: %i" % first_filter_discarded_answered_annotations)
print("Second filter discarded aggregation annotaions: %i" % second_filter_discarded_answered_annotations)
print("Third filter discarded aggregation annotaions: %i" % third_filter_discarded_answered_annotations)
print("Final number of aggreg instances: %i" % len(annotations))
print("Final number of (aggregation) annotations: %i" % final_aggregation_annotations)
print()
for type_, l in detailed_entities.items():
    print("Entities sets (tot./unique) %s: %i/%i" % (type_, len(l), len(set(l))))
print()
for type_, l in detailed_aggreg.items():
    print("Aggregations (tot./unique) %s: %i/%i" % (type_, len(l), len(set(l))))

# %%
