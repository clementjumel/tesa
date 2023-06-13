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
# %load_ext tensorboard

from tesa.modeling.pipeline import Pipeline
from tesa.modeling.nn import RegressionMLP, ClassificationMLP, RegressionBilinear, ClassificationBilinear
import torch

# %% [markdown]
# ## Data pipeline parameters

# %%
batch_size = 32
drop_last = False
test_proportion = 0.25
valid_proportion = 0.25
use_k_fold = False
k_k_fold = None

# %% [markdown]
# ## Load the data

# %%
pipeline = Pipeline(use_k_fold=use_k_fold)
pipeline.process_data(batch_size=batch_size,
                      drop_last=drop_last,
                      test_proportion=test_proportion,
                      valid_proportion=valid_proportion,
                      k=k_k_fold)

# %% [markdown]
# ## Metrics

# %%
scores_names = [
    'average_precision', 
    'precision_at_k', 
    'recall_at_k', 
    'reciprocal_best_rank', 
    'reciprocal_average_rank', 
    'ndcg'
]
n_updates = 50

# %% [markdown]
# ## Initialize the embedding

# %%
from tesa.modeling.models import BaseModel
BaseModel.initialize_word2vec_embedding()
BaseModel.initialize_bert_embedding()

# %% [markdown]
# ## Half BOW

# %%
input_dim, hidden_dim1, hidden_dim2 = 5537, 1000, 100

dropout = 0.1
lr = 4e-7
milestones = [1, 2, 6]
gamma = 0.5
n_epochs = 10

is_regression = True
#is_regression = False

if is_regression:
    loss = torch.nn.MSELoss()
    net = RegressionMLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=dropout)
    
else:
    weight = torch.tensor([1, 1], dtype=torch.float)
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    net = ClassificationMLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=dropout)
    
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

# %%
from tesa.modeling.models import HalfBOWModel

model = HalfBOWModel(scores_names=scores_names, 
                     experiment_name='test_1',
                     net=net,
                     optimizer=optimizer,
                     lr_scheduler=lr_scheduler,
                     loss=loss,
                     vocab_frequency_range=[100, 10000000000])

pipeline.preview_data(model=model)

# %%
# %tensorboard --logdir logs

# %%
pipeline.train_model(model=model, 
                     n_epochs=n_epochs, 
                     n_updates=n_updates,
                     is_regression=is_regression)

# %%
pipeline.train_model(model=model, 
                     n_epochs=n_epochs, 
                     n_updates=n_updates,
                     is_regression=is_regression)

# %%
model.final_plot(align_experiments=True,
                 display_training_scores=False, 
                 scores_names=[
                     'average_precision', 
                     'precision_at_k', 
                     'recall_at_k', 
                     'reciprocal_best_rank', 
                     'reciprocal_average_rank', 
                     'ndcg'
                 ])

# %%
model.display_metrics()

# %%
pipeline.explain_model(model=model, 
                       display_explanations=True,
                       n_samples=5,
                       n_answers=10,
                       scores_names=[
                           'average_precision', 
                           'precision_at_k', 
                           'recall_at_k', 
                           'reciprocal_best_rank', 
                           'reciprocal_average_rank', 
                           'ndcg'
                       ])

# %% [markdown]
# ## Full BOW

# %%
input_dim, hidden_dim1, hidden_dim2 = 3196, 1000, 100

dropout = 0.1
lr = 4e-7
milestones = [1, 2, 5, 8]
gamma = 0.5
n_epochs = 10

is_regression = False

if is_regression:
    loss = torch.nn.MSELoss()
    net = RegressionMLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=dropout)
    
else:
    weight = torch.tensor([1, 1], dtype=torch.float)
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    net = ClassificationMLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=dropout)
    
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

# %%
from modeling.models import FullBOWModel

model = FullBOWModel(vocab_frequency_range=[100, 10000],
                     net=net,
                     optimizer=optimizer,
                     lr_scheduler=lr_scheduler,
                     loss=loss,
                     scores_names=scores_names)

pipeline.preview_data(model=model)

# %%
pipeline.train_model(model=model, 
                     n_epochs=n_epochs, 
                     n_updates=n_updates,
                     is_regression=is_regression)

# %%
model.final_plot(align_experiments=True,
                 display_training_scores=False, 
                 scores_names=[
                     'average_precision', 
                     'precision_at_k', 
                     'recall_at_k', 
                     'reciprocal_best_rank', 
                     'reciprocal_average_rank', 
                     'ndcg'
                 ])

# %%
model.display_metrics()

# %%
pipeline.explain_model(model=model, 
                       display_explanations=True,
                       n_samples=5,
                       n_answers=10,
                       scores_names=[
                           'average_precision', 
                           'precision_at_k', 
                           'recall_at_k', 
                           'reciprocal_best_rank', 
                           'reciprocal_average_rank', 
                           'ndcg'
                       ])

# %% [markdown]
# ## Embedding

# %%
input_dim, hidden_dim1, hidden_dim2 = 600, 1000, 100

dropout = 0.1
lr = 4e-7
milestones = [1, 2, 5, 8]
gamma = 0.5
n_epochs = 10

#is_regression = False
is_regression = True

if is_regression:
    loss = torch.nn.MSELoss()
    net = RegressionMLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=dropout)
    
else:
    weight = torch.tensor([1, 1], dtype=torch.float)
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    net = ClassificationMLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout=dropout)
    
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

# %%
from tesa.modeling.models import EmbeddingModel

model = EmbeddingModel(scores_names=scores_names,
                       net=net,
                       optimizer=optimizer,
                       lr_scheduler=lr_scheduler,
                       loss=loss,
                       experiment_name='test_2')

pipeline.preview_data(model=model)

# %%
# %tensorboard --logdir logs

# %%
pipeline.train_model(model=model, 
                     n_epochs=n_epochs, 
                     n_updates=n_updates,
                     is_regression=is_regression)

# %%
model.final_plot(align_experiments=True,
                 display_training_scores=False, 
                 scores_names=[
                     'average_precision', 
                     'precision_at_k', 
                     'recall_at_k', 
                     'reciprocal_best_rank', 
                     'reciprocal_average_rank', 
                     'ndcg'
                 ])

# %%
model.display_metrics()

# %%
pipeline.explain_model(model=model, 
                       display_explanations=True,
                       n_samples=5,
                       n_answers=10,
                       scores_names=[
                           'average_precision', 
                           'precision_at_k', 
                           'recall_at_k', 
                           'reciprocal_best_rank', 
                           'reciprocal_average_rank', 
                           'ndcg'
                       ])

# %% [markdown]
# ## Embedding Bilinear

# %%
input_dim1, input_dim2 = 300, 300

dropout = 0.1
lr = 8e-7
milestones = [1, 3]
gamma = 0.5
n_epochs = 2

#is_regression = False
is_regression = True

if is_regression:
    loss = torch.nn.MSELoss()
    net = RegressionBilinear(input_dim1=input_dim1, input_dim2=input_dim2, dropout=dropout)
    
else:
    weight = torch.tensor([1, 1], dtype=torch.float)
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    net = ClassificationBilinear(input_dim1=input_dim1, input_dim2=input_dim2, dropout=dropout)
    
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)

# %%
from tesa.modeling.models import EmbeddingBilinearModel

model = EmbeddingBilinearModel(scores_names=scores_names,
                               net=net,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler,
                               loss=loss,
                               experiment_name='test_3')

pipeline.preview_data(model=model)

# %%
# %tensorboard --logdir logs/test_3

# %%
pipeline.train_model(model=model, 
                     n_epochs=n_epochs, 
                     n_updates=n_updates,
                     is_regression=is_regression)

# %%
pipeline.train_model(model=model, 
                     n_epochs=n_epochs, 
                     n_updates=n_updates,
                     is_regression=is_regression)

# %%
model.final_plot(align_experiments=True,
                 display_training_scores=False, 
                 scores_names=[
                     'average_precision', 
                     'precision_at_k', 
                     'recall_at_k', 
                     'reciprocal_best_rank', 
                     'reciprocal_average_rank', 
                     'ndcg'
                 ])

# %%
model.display_metrics()

# %%
pipeline.explain_model(model=model, 
                       display_explanations=True,
                       n_samples=5,
                       n_answers=10,
                       scores_names=[
                           'average_precision', 
                           'precision_at_k', 
                           'recall_at_k', 
                           'reciprocal_best_rank', 
                           'reciprocal_average_rank', 
                           'ndcg'
                       ])

# %%
