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
# ### Setup

# %%
import sys
sys.path.append('../')
del sys

# %reload_ext autoreload
# %autoreload 2
# %load_ext tensorboard

# %% [markdown]
# ### Display the TensorBoard

# %%
logdir = '/Users/clement/Desktop/tensorboard_logs'

# %%
# %tensorboard --logdir $logdir

# %% [markdown]
# If an error occur with port 6006 (default one), in a shell, run:
#     - lsof -i:6006
#     (list the processes running on it with their PIDs)
#     - kill -9 PID
#     (to kill the corresponding process, for each process)

# %%
