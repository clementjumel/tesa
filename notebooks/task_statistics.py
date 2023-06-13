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
# ### Imports

# %% pycharm={"is_executing": false}
import sys
sys.path.append('../')
del sys

# %reload_ext autoreload
# %autoreload 2

from tesa.toolbox.parsers import standard_parser, add_annotations_arguments, add_task_arguments
from tesa.toolbox.utils import load_task
from tesa.modeling.utils import format_context

# %% [markdown]
# ### Parameters

# %%
ap = standard_parser()
add_annotations_arguments(ap)
add_task_arguments(ap)
args = ap.parse_args(["--root", ".."])

# %% [markdown]
# ### Load the modeling task

# %%
task = load_task(args)

# %% [markdown]
# ### Statistics

# %%
from collections import defaultdict
from numpy import mean

print("Aggregatable instances in train: %i" % len(task.train_loader))
print("Aggregatable instances in valid: %i" % len(task.valid_loader))
print("Aggregatable instances in test: %i" % len(task.test_loader))
print("Aggregatable instances (total): %i" % (len(task.train_loader)+len(task.valid_loader)+len(task.test_loader)))
print()

d1 = defaultdict(list)
for data_loader in [task.train_loader, task.valid_loader, task.test_loader]:
    for ranking_task in data_loader:
        d2 = defaultdict(int)
        for inputs, outputs in ranking_task:
            d2['all']+= outputs.sum().item()
            d2[inputs['entities_type']]+= outputs.sum().item()
        for key, value in d2.items():
            d1[key].append(value)

for key, value in d1.items():
    print("Average relevant aggregation per instance (%s): %.2f" % (key, mean(value)))

# %%
c1, c2, c3 = 0, 0, 0

for ranking_task in task.train_loader:
    for _, outputs in ranking_task:
        #c1 += outputs.sum().item()
        c1 += len(outputs)

for ranking_task in task.valid_loader:
    for _, outputs in ranking_task:
        #c2 += outputs.sum().item()
        c2 += len(outputs)
        
for ranking_task in task.test_loader:
    for _, outputs in ranking_task:
        #c3 += outputs.sum().item()
        c3 += len(outputs)
        
print(c1, c2, c3)


# %% [markdown]
# ## Display the examples

# %%
def example(task, entities, context_extract):
    for data_loader, loader_name in zip([task.train_loader, task.valid_loader, task.test_loader], ["train", "valid", "test"]):
        all_choices, all_outputs = [], []
        for ranking_task in data_loader:
            inputs, _ = ranking_task[0]
            if sorted(inputs['entities']) == sorted(entities) and context_extract in inputs['nyt_contexts'][0]:
                print(loader_name)
                for key, value in inputs.items():
                    if key != 'choices':
                        print(key, '->', value)
                print()

                context = format_context(inputs, args.context_format, args.context_max_size)
                print('context: ->', context)

                print()
                for i, o in ranking_task:
                    all_choices.extend(i['choices'])
                    all_outputs.extend(o.tolist())
                
                final_choices = []
                for choice, label in zip(all_choices, all_outputs):
                    if not label:
                        final_choices.append(choice)
                    else:
                        final_choices.append("\\textbf{" + choice + "}")
                
                print(", ".join(final_choices))
                    
                all_choices, all_outputs = [], []
                
                print('\n\n')


# %%
entities_list = [
    ["Francois Bayrou", "Nicolas Sarkozy", "Segolene Royal"],
    ["Chicago", "London"],
    ["Microsoft Corp.", "Sony Corp."],
]
context_extracts_list = [
    "", #"The Socialist candidate",
    "",
    "", # "Nintendo",
]

for entities, context_extract in zip(entities_list, context_extracts_list):
    example(task, entities, context_extract)

# %% [markdown]
# ### Check if examples are unseen

# %%
c1, c2, c3 = 0, 0, 0
for data_loader in [task.train_loader]:
    for ranking_task in data_loader:
        inputs, _ = ranking_task[0]
        if 'Francois Bayrou' in inputs['entities']:
            c1 += 1
        if 'Nicolas Sarkozy' in inputs['entities']:
            c2 += 1
        if 'Segolene Royal' in inputs['entities']:
            c3 += 1
            
print(c1, c2, c3)

# %%
s = set()
for data_loader in [task.train_loader, task.valid_loader, task.test_loader]:
    for ranking_task in data_loader:
        for inputs, _ in ranking_task:
            s.update(inputs['choices'])

for a in [
"politicians",
"american politicians",
"french politicians",
"political figures",
"French politicians",
"political leaders",
"politician",
"political candidates",
"politicans",
"politicians",

"american cities",
"cities",
"political powers",
"american regions",
"american areas",
"major cities",
"politicians",
"us cities",
"world cities",
"people",

"multinational companies",
"corporations",
"multinational corporations",
"american companies",
"textbf{technology companies",
"tech companies",
"companies",
"businesses",
"countries",
"technology firms",


    
    
    "french politicians",
        "politicians",
        "american politicians",
        "republicans",
        "french politician",
        "political figures",
        "politician",
        "political candidates",
        "nations",
        "leaders",
    
    "american cities",
        "major cities",
        "cities",
        "metropolitan cities",
        "major metropolitan cities",
        "large cities",
        "metropolitan areas",
        "populations",
        "regions",
        "political powers",
    
    'multinational corporations',
        'multinational companies',
        'corporations',
        'technology companies',
        'american corporations',
        'tech companies',
        'companies',
        'technology corporations',
        'tech corporations',
        'technology firms',
]:
    if a not in s:
        print(a)

# %% [markdown]
# ### Check number of set of unseen entities

# %%
se = set()
l = []

for data_loader in [task.train_loader, task.valid_loader, task.test_loader]:
    for ranking_task in data_loader:
        inputs, _ = ranking_task[0]
        entities = inputs['entities']
        se.update(set(entities))
        l.extend(entities)

print(len(se))
print(len(l))

# %%
se, l1, l2 = set(), [], []

data_loader = task.train_loader
for ranking_task in data_loader:
    inputs, outputs = ranking_task[0]
    entities = inputs['entities']
    se.update(set(entities))

data_loader = task.valid_loader
for ranking_task in data_loader:
    inputs, outputs = ranking_task[0]
    entities = inputs['entities']
    if all([entity not in se for entity in entities]):
        l1.append(entities)
    
data_loader = task.test_loader
for ranking_task in data_loader:
    inputs, outputs = ranking_task[0]
    entities = inputs['entities']
    if all([entity not in se for entity in entities]):
        l2.append(entities)

print(len(l1))
print(sum([len(x) for x in l1]))
print(len(l2))
print(sum([len(x) for x in l2]))

# %%
print(len({x for x in l1}))

# %% [markdown]
# ### Check if examples entities are already seen

# %%
for entities in entities_list:
    for entity in entities:
        if entity not in se:
            print(entity)

# %%
