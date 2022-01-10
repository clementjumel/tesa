""" Parameters of the various scripts. """

RANDOM_SEED = 2345

# region Annotation task parameters
YEARS = [2006, 2007]
MAX_TUPLE_SIZE = 6
RANDOM = True
EXCLUDE_PILOT = True
ANNOTATION_TASK_SHORT_SIZE = 10000

LOAD_WIKI = True
WIKIPEDIA_FILE_NAME = "wikipedia_global"
CORRECT_WIKI = True
# endregion

# region Modeling task parameters
MIN_ASSIGNMENTS = 5
MIN_ANSWERS = 2
K_CROSS_VALIDATION = 5

VALID_PROPORTION = 0.25
TEST_PROPORTION = 0.25
RANKING_SIZE = 24
BATCH_SIZE = 4
CONTEXT_FORMAT = 'v0'
TARGETS_FORMAT = 'v0'
# endregion

# region Models parameters
SCORES_NAMES = [
    'average_precision',
    'precision_at_10',
    'recall_at_10',
    'ndcg_at_10',
    'reciprocal_best_rank',
    'reciprocal_average_rank',
]

TASK_NAME = "context-dependent-same-type"
CONTEXT_MAX_SIZE = 750
SHOW_RANKINGS = 5
SHOW_CHOICES = 10

BART_BEAM = 10
BART_LENPEN = 1.0
BART_MAX_LEN_B = 100
BART_MIN_LEN = 1
BART_NO_REPEAT_NGRAM_SIZE = 2
# endregion
