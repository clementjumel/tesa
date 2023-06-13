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
# ## Setup

# %% pycharm={"is_executing": false}
import sys
sys.path.append('../')
del sys
# %reload_ext autoreload
# %autoreload 2

# %% pycharm={"is_executing": false}
from tesa.database_creation.database import Database

# %% [markdown]
# ## Load the data

# %%
database = Database()
database.process_task(assignment_threshold=5)

# %% [markdown]
# ## NAs

# %% [markdown]
# Analyze the number of valid answers (ie not bugs) per task:

# %%
valid = [0, 0, 0, 0]
valid_ids, bug_ids = [], []

for id_, annotation_list in database.annotations.items():
    cmpt = 0
    
    for annotation in annotation_list:
        if not annotation.bug:
            cmpt += 1
    
    valid[cmpt] += 1
    
    if cmpt >= 2:
        valid_ids.append(id_)
    else:
        bug_ids.append(id_)

print("Number of perfect/good examples: {}/{}".format(valid[3], valid[2]))
print("Number of bad/awful examples: {}/{}".format(valid[1], valid[0]))
print("Number of accepted tasks: {}, number of rejected: {} ({}% accepted)".format(len(valid_ids), len(bug_ids), round(100*len(valid_ids)/(len(valid_ids) + len(bug_ids)))))

# %% [markdown]
# ## Observation of the results

# %% [markdown]
# Good results

# %%
for id_ in valid_ids:
    print(database.queries[id_])
    for annotation in database.annotations[id_]:
        print(annotation)
    print()

# %% [markdown]
# Bad results

# %%
for id_ in bug_ids:
    print(database.queries[id_])
    for annotation in database.annotations[id_]:
        print(annotation)
    print()

# %% [markdown]
# Unique answers for each task

# %%
from collections import defaultdict

tuple_answers = defaultdict(list)

for id_ in valid_ids:
    a = []
    
    for annotation in database.annotations[id_]:
        if not annotation.bug:
            a.extend(annotation.preprocessed_answers)

    entities = ', '.join(sorted(database.queries[id_].entities))
    tuple_answers[entities].append(a)
            
    print(entities, ' -> ', ', '.join(set([s+' ['+str(a.count(s))+']' if a.count(s)>1 else s for s in a])))


# %% [markdown]
# Unique answers for each tuple:

# %%
sorted_answers = [(len([answer for answer_list in answers for answer in answer_list]),
                   entities,
                   answers)
                  for entities, answers in tuple_answers.items()]
sorted_answers = sorted(sorted_answers, reverse=True)

for count, entities, answers in sorted_answers:
    flattened_answers = [answer for answer_list in answers for answer in answer_list]
    flattened_unique_answers = set(flattened_answers)
    
    print(entities, ' (unique/total answers {}/{})'.format(len(flattened_unique_answers), len(flattened_answers)))
    
    answers_counts = set([(flattened_answers.count(a), a) for a in flattened_answers])
    sorted_answers_counts = sorted(answers_counts, reverse=True)
    
    print(', '.join([answer+' ['+str(count)+']' for count, answer in sorted_answers_counts]), '\n')


# %% [markdown]
# Overall most frequent answers
#

# %%
overall_answers_dict = defaultdict(int)

for _, _, answers in sorted_answers:
    flattened_answers = [answer for answer_list in answers for answer in answer_list]
    for answer in flattened_answers:
        overall_answers_dict[answer] += 1

overall_answers = [(count, answer) for answer, count in overall_answers_dict.items()]
overall_answers = sorted(overall_answers, reverse=True)
for count, answer in overall_answers:
    print(answer + ': ' + str(count))

# %% [markdown]
# Answers for the most frequent tuples accross different contexts:

# %%
for _, entities, _ in sorted_answers[:100]:
    print(entities, ':')
    
    res_res = []
    for id_ in valid_ids:
        if entities == ', '.join(sorted(database.queries[id_].entities)):
            res = []
            
            for annotation in database.annotations[id_]:
                if not annotation.bug:
                    res.extend(annotation.preprocessed_answers)
            
            res = sorted(res)
            res_res.append(res)
            
    flatten_res_res = [r for l in res_res for r in l]
    count_res_res = sorted(set([(flatten_res_res.count(a), a) for a in flatten_res_res]), reverse=True)
    
    n = int(len(count_res_res)/2)
    to_exclude = [a for _, a in count_res_res[:n]]
    
    for res in res_res:
        res_excluded = [a if a not in to_exclude else '_' for a in res]
        print(', '.join(set([r+' ['+str(res_excluded.count(r))+']' if res_excluded.count(r)>1 else r for r in res_excluded])))
    
    print()


# %% [markdown]
# ## Statistics

# %%
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ### Calcul de Kappa (Jeiss)

# %%
N = len(database.annotations)
print("Number of tasks (subjects): {}".format(N))

# %%
n = 3
print("Number of annotation per task: {}".format(n))

# %%
categories_bin = {0, 1}
k_bin = len(categories_bin)
print("Number of categories (binary case): {}".format(k_bin))

categories_gen = set()

for id_, annotation_list in database.annotations.items():
    for annotation in annotation_list:
        if annotation.preprocessed_answers == []:
            categories_gen.add('None')
        else:
            categories_gen.add(annotation.preprocessed_answers[0])

k_gen = len(categories_gen)
print("Number of catefories (genral case): {}".format(k_gen))

# %%
n_ij_bin = defaultdict(int)

for id_, annotation_list in database.annotations.items():
    for annotation in annotation_list:
        if annotation.preprocessed_answers == []:
            n_ij_bin[(id_, 0)] += 1
        else:
            n_ij_bin[(id_, 1)] += 1

n_ij_gen = defaultdict(int)

for id_, annotation_list in database.annotations.items():
    for annotation in annotation_list:
        if annotation.preprocessed_answers == []:
            n_ij_gen[(id_, 'None')] += 1
        else:
            n_ij_gen[(id_, annotation.preprocessed_answers[0])] += 1

pj_bin = dict([(category,
               sum([n_ij_bin[(id_, category)] for id_ in database.annotations])/(N*n))
               for category in categories_bin])

pj_gen = dict([(category,
               sum([n_ij_gen[(id_, category)] for id_ in database.annotations])/(N*n)
               ) for category in categories_gen])

Pi_bin = dict([(id_,
               sum([n_ij_bin[(id_, category)]*(n_ij_bin[(id_, category)]-1) for category in categories_bin])/(n*(n-1))
               ) for id_ in database.annotations])

Pi_gen = dict([(id_,
               sum([n_ij_gen[(id_, category)]*(n_ij_gen[(id_, category)]-1) for category in categories_gen])/(n*(n-1))
               ) for id_ in database.annotations])

Pm_bin = sum([Pi_bin[id_] for id_ in database.annotations])/N
Pe_bin = sum([pj_bin[category]**2 for category in categories_bin])

Pm_gen = sum([Pi_gen[id_] for id_ in database.annotations])/N
Pe_gen = sum([pj_gen[category]**2 for category in categories_gen])

kappa_bin = (Pm_bin-Pe_bin)/(1-Pe_bin)
print("Kappa (binary case): {}".format(kappa_bin))

kappa_gen = (Pm_gen-Pe_gen)/(1-Pe_gen)
print("Kappa (general case): {}".format(kappa_gen))

# %% [markdown]
# ### Number of answers per annotator

# %%
annotators = defaultdict(list)

for id_, annotation_list in database.annotations.items():
    for annotation in annotation_list:
        annotators[annotation.worker_id].extend(annotation.preprocessed_answers)
        
n_annotators = len(annotators)
mean = round(sum([len(l) for _, l in annotators.items()])/n_annotators)
maximum = max([len(l) for _, l in annotators.items()])
print("Number of annotators: {}; mean number of answers per annotators: {}; max: {}".format(n_annotators, mean, maximum))

annotators_all = sorted([(len(l), annotator) for annotator, l in annotators.items()], reverse=True)
annotators_different = sorted([(len(set(l)), annotator) for annotator, l in annotators.items()], reverse=True)
unique_answers = [answer for count, answer in overall_answers if count == 1]
annotators_unique = sorted([(
    len([answer for answer in l if answer in unique_answers]),
    annotator) for annotator, l in annotators.items()], reverse=True)

plt.figure(num=None, figsize=(16, 8))
plt.bar(range(len(annotators)), [count for count, _ in annotators_all], width=0.95, label='Total number of answers per annotator')
plt.bar(range(len(annotators)), [count for count, _ in annotators_different], width=0.95, label='Number of different answers per annotator')
plt.bar(range(len(annotators)), [count for count, _ in annotators_unique], width=0.95, label='Number of unique (overall) answers per annotator')
plt.legend()

# %% [markdown]
# ### Frequencies of the tuples

# %%
frequencies = defaultdict(int)

for id_, annotation_list in database.annotations.items():
    entities = tuple(database.queries[id_].entities)
    for annotation in annotation_list:
        frequencies[entities] += 1

frequencies = sorted([(count, entities) for entities, count in frequencies.items()], reverse = True)
n = len(frequencies)

plt.figure(num=None, figsize=(16, 4))
#plt.xscale("log")
plt.bar(range(1, n+1), [count for count, _ in frequencies], width=1)

print("Number of tuples: {}".format(len(frequencies)))

for count, entities in frequencies:
    print(', '.join(entities), ':', count)

# %%
frequencies = defaultdict(int)

for id_, annotation_list in database.annotations.items():
    entities = (database.queries[id_].entities_type_)
    for annotation in annotation_list:
        frequencies[entities] += 1

frequencies = sorted([(count, entities) for entities, count in frequencies.items()], reverse = True)
n = len(frequencies)

plt.figure(num=None, figsize=(16, 4))
#plt.xscale("log")
plt.bar(range(1, n+1), [count for count, _ in frequencies], width=1)

print("Number of tuples: {}".format(len(frequencies)))

for count, entities in frequencies:
    print(entities, ':', count)

# %% [markdown]
# ### Frequencies of the bug tuples

# %%
frequencies, frequencies_bug = defaultdict(int), defaultdict(int)

for id_, annotation_list in database.annotations.items():
    entities = tuple(database.queries[id_].entities)
    for annotation in annotation_list:
        frequencies[entities] += 1
        
        if annotation.bug:
            frequencies_bug[entities] += 1
        else:
            frequencies_bug[entities] += 0.01

frequencies_bug = sorted([(count/frequencies[entities], entities) for entities, count in frequencies_bug.items()], reverse = True)
n_bug = len(frequencies_bug)

plt.figure(num=None, figsize=(16, 4))
#plt.xscale("log")
plt.bar(range(1, n_bug+1), [count for count, _ in frequencies_bug], width=1)

for count, entities in frequencies_bug[:20]:
    print(', '.join(entities), ':', count)

# %% [markdown]
# ### Frequencies of the answers:

# %%
frequencies = defaultdict(int)

for id_, annotation_list in database.annotations.items():
    for annotation in annotation_list:
        if not annotation.bug:
            for answer in annotation.preprocessed_answers:
                frequencies[answer] += 1

frequencies = sorted([(count, answer) for answer, count in frequencies.items()], reverse = True)
n = len(frequencies)

plt.figure(num=None, figsize=(16, 4))
#plt.xscale("log")
plt.bar(range(n), [count for count, _ in frequencies], width=1, log=True)

for count, answer in frequencies[:20]:
    print(answer, ':', count)

# %% [markdown]
# ### Number of answers per task

# %%
answers = defaultdict(list)

for id_, annotation_list in database.annotations.items():
    for annotation in annotation_list:
        answers[id_].extend(annotation.preprocessed_answers)
                
answers_all = [len(a) for _, a in answers.items()]
answers_different = [len(set(a)) for _, a in answers.items()]
answers_unique = [len([answer for answer in a if answer in unique_answers]) for _, a in answers.items()]

bins = [i - 0.5 for i in range(8)]
plt.figure(num=None, figsize=(5, 8))
plt.hist(answers_all, bins, width=0.95, align='mid', label="Total number of answers per task")
plt.hist(answers_different, bins, width=.66, align='mid', label="Number of different answers per task")
plt.hist(answers_unique, bins, width=.33, align='mid', label="Number of unique (overall) answers per task")
plt.legend()
plt.xticks(np.arange(min(bins)+0.5, max(bins)+0.5, 1.0))

# %% [markdown]
# ### Number of answers per tuple of entities

# %%
answers = defaultdict(list)

for id_, annotation_list in database.annotations.items():
    entities = tuple(sorted(database.queries[id_].entities))
    for annotation in annotation_list:
        answers[entities].extend(annotation.preprocessed_answers)
                
answers_all = [len(a) for _, a in answers.items()]
answers_different = [len(set(a)) for _, a in answers.items()]
answers_unique = [len([answer for answer in a if answer in unique_answers]) for _, a in answers.items()]

bins = [i - 0.5 for i in range(20)]
plt.figure(num=None, figsize=(20, 10))
plt.hist(answers_all, bins, width=0.95, align='mid', label="Total number of answers per tuple of entities")
plt.hist(answers_different, bins, width=.66, align='mid', label="Number of different answers per tuple of entities")
plt.hist(answers_unique, bins, width=.33, align='mid', label="Number of unique (overall) answers per tuple of entities")
plt.legend()
plt.xticks(np.arange(min(bins)+0.5, max(bins)+0.5, 1.0))

# %% [markdown]
# ### Mean frequency of the answers per annotator

# %%
answers = defaultdict(list)

for id_, annotation_list in database.annotations.items():
    for annotation in annotation_list:
        answers[annotation.worker_id].extend(annotation.preprocessed_answers)
                
answers_all = [np.mean([overall_answers_dict[answer] for answer in a]) for _, a in answers.items()]
#answers_different = [len(set(a)) for _, a in answers.items()]
#answers_unique = [len([answer for answer in a if answer in unique_answers]) for _, a in answers.items()]

bins = [10*i  for i in range(13)]
plt.figure(num=None, figsize=(20, 10))
plt.hist(answers_all, bins, align='mid', label="Total number of answers per tuple of entities")
#plt.hist(answers_different, bins, width=.66, align='mid', label="Number of different answers per tuple of entities")
#plt.hist(answers_unique, bins, width=.33, align='mid', label="Number of unique (overall) answers per tuple of entities")
plt.legend()
#plt.xticks(np.arange(min(bins)+0.5, max(bins)+0.5, 1.0))

# %%
