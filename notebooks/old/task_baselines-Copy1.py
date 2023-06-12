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

from modeling.pipeline import Pipeline
import modeling.models as models

# %% [markdown]
# ## Data pipeline parameters

# %%
batch_size = 32
drop_last = False
test_proportion = 0.5
valid_proportion = 0.5

# %% [markdown]
# ## Load the data

# %%
pipeline = Pipeline()
pipeline.process_data(batch_size=batch_size,
                      drop_last=drop_last,
                      test_proportion=test_proportion,
                      valid_proportion=valid_proportion)

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

# %% [markdown]
# ## Initializes the embedding

# %%
from gensim.models import KeyedVectors
from transformers import BertModel, BertTokenizer

word2vec_embedding = KeyedVectors.load_word2vec_format(fname='../modeling/pretrained_models/GoogleNews-vectors-negative300.bin',
                                                       binary=True)

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# %% [markdown]
# # Simple baselines
# ## Random
# Completely random predictions

# %%
model = models.RandomBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Frequency
# The prediction is based on the frequency of the choice as a good answer.

# %%
model = models.FrequencyBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# # Summary based
# ## Summaries Counts
# The prediction is based on the total number of words (with repetition) of all the wikipedia summaries, that are in the answer.

# %%
model = models.SummariesCountBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Summaries Unique Count
# The prediction is based on the number of unique words of all the wikipedia summaries, that are in the answer.

# %%
model = models.SummariesUniqueCountBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Summaries Overlap
# The prediction is based on the number of (unique) words of the summaries' overlap that are in the answer.

# %%
model = models.SummariesOverlapBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Summaries Average Embedding
# The prediction if based on the (cosine) similarity between the average embedding of the words of the answer, and the average embedding of the words of all the wikipedia summaries.

# %%
model = models.SummariesAverageEmbeddingBaseline(scores_names=scores_names, 
                                          pretrained_model=word2vec_embedding,
                                          pretrained_model_dim=300)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Summaries Overlap Average Embedding
# The prediction is based on the (cosine) similarity of the average embedding of the words of the answer and the average embedding of the overlap between all the wikipedia summaries.

# %%
model = models.SummariesOverlapAverageEmbeddingBaseline(scores_names=scores_names,
                                                 pretrained_model=word2vec_embedding,
                                                 pretrained_model_dim=300)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Activated Summaries
# The prediction is based on the number of summaries that have words matching a word from the answer.

# %%
model = models.ActivatedSummariesBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# # Context based
# ## Context Counts
# The prediction is based on the total number of words (with repetition) of the context, that are in the answer.

# %%
model = models.ContextCountBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Context Unique Count
# The prediction is based on the number of unique words of all the context, that are in the answer.

# %%
model = models.ContextUniqueCountBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Context Average Embedding
# The prediction if based on the (cosine) similarity between the average embedding of the words of the answer, and the average embedding of the words of the context.

# %%
model = models.ContextAverageEmbeddingBaseline(scores_names=scores_names, 
                                          pretrained_model=word2vec_embedding,
                                          pretrained_model_dim=300)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# # Summary & context based
# ## Summaries Context Counts
# The prediction is based on the total number of words (with repetition) of all the wikipedia summaries and of the context, that are in the answer.

# %%
model = models.SummariesContextCountBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Summaries Context Unique Count
# The prediction is based on the number of unique words of all the wikipedia summaries and of the context, that are in the answer.

# %%
model = models.SummariesContextUniqueCountBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Summaries Overlap Context
# The prediction is based on the number of (unique) words of the summaries' overlap and of the context that are in the answer.

# %%
model = models.SummariesOverlapContextBaseline(scores_names=scores_names)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Summaries Context Average Embedding
# The prediction if based on the (cosine) similarity between the average embedding of the words of the answer, and the average embedding of the words of all the wikipedia summaries and of the context.

# %%
model = models.SummariesContextAverageEmbeddingBaseline(scores_names=scores_names, 
                                          pretrained_model=word2vec_embedding,
                                          pretrained_model_dim=300)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %% [markdown]
# ## Summaries Overlap Context Average Embedding
# The prediction is based on the (cosine) similarity of the average embedding of the words of the answer and the average embedding of the overlap between all the wikipedia summaries and of the context.

# %%
model = models.SummariesOverlapContextAverageEmbeddingBaseline(scores_names=scores_names,
                                                 pretrained_model=word2vec_embedding,
                                                 pretrained_model_dim=300)

# %%
pipeline.preview_data(model=model, include_train=False, include_valid=True)
pipeline.valid_model(model=model)
model.display_metrics(scores_names=None)
pipeline.explain_model(model=model, scores_names=None, n_samples=5, n_answers=10)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Closest Summaries Embedding
# Grade is the cosine similarity between the average embedding of the words of an answer, and the average embedding of all the words of the wikipedia summaries.

# %%
from modeling.models import ClosestSoftOverlapEmbedding

model = ClosestSoftOverlapEmbedding(scores_names=scores_names,
                                    pretrained_model=word2vec_embedding,
                                    pretrained_model_dim=300)
pipeline.preview_data(model=model, include_valid=True)

# %%
pipeline.valid_model(model=model)

# %%
model.display_metrics(scores_names=None)

# %%
pipeline.explain_model(model=model, 
                       scores_names=None,
                       n_samples=5,
                       n_answers=10)

# %% [markdown]
# ## BERT Embedding

# %%
from modeling.models import BertEmbedding

model = BertEmbedding(scores_names=scores_names,
                      pretrained_model=bert_model,
                      pretrained_model_dim=768,
                      tokenizer=bert_tokenizer)

pipeline.preview_data(model=model, include_valid=True)

# %%
pipeline.valid_model(model=model)

# %%
model.display_metrics(scores_names=None)

# %%
pipeline.explain_model(model=model, 
                       scores_names=None,
                       n_samples=5,
                       n_answers=10)

# %% [markdown]
# ## BERT for Next Sentence Prediction

# %%
from modeling.models import NSPBertEmbedding

model = NSPBertEmbedding(scores_names=scores_names)
model.initialize_nsp_bert_embedding()
pipeline.preview_data(model=model, include_valid=True)

# %%
pipeline.valid_model(model=model)

# %%
model.display_metrics(scores_names=None)

# %%
pipeline.explain_model(model=model, 
                       scores_names=None,
                       n_samples=5,
                       n_answers=10)

# %%
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# %%
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12

# %%
encoded_layers[11][:, 0, :].shape

# %%
from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
nlp = pipeline('sentiment-analysis')
nlp('We are very happy to include pipeline into the transformers repository.')

# %%
nlp = pipeline('question-answering')
nlp({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline have been included in the huggingface/transformers repository'
})

# %%

nlp = pipeline('feature-extraction')
nlp(
    'the two politicians'
)

# %%
features = pipeline.valid_loader[0][0]
outputs = pipeline.valid_loader[0][1]
sequence_0 = features['context'] + ', '.join(features['entities']) + ' '.join(features['summaries'])

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]

for i in range(len(features['choices'])):
    sequence_1 = features['choices'][i]

    paraphrase = tokenizer.encode_plus(sequence_0, sequence_1, return_tensors="pt")
    paraphrase_classification_logits = model(**paraphrase)[0]
    paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]

    if outputs[i]:
        print("Should be high")
        print(f"{classes[1]}: {round(paraphrase_results[1] * 100)}%")    
    else:
        print("Should be low")
        print(f"{classes[1]}: {round(paraphrase_results[1] * 100)}%")    

# %%
