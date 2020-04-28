""" Parameters of the various scripts. """

# region Annotation task parameters
YEARS = [2006, 2007]
MAX_TUPLE_SIZE = 6
RANDOM = True
EXCLUDE_PILOT = False
ANNOTATION_TASK_SHORT_SIZE = 10000
ANNOTATION_TASK_SEED = 0

LOAD_WIKI = True
WIKIPEDIA_FILE_NAME = "wikipedia_global"
CORRECT_WIKI = True
# endregion

# region Modeling task parameters
MIN_ASSIGNMENTS = 5
MIN_ANSWERS = 2
K_CROSS_VALIDATION = 5
MODELING_TASK_SEED = 1

VALID_PROPORTION = 0.25
TEST_PROPORTION = 0.25
RANKING_SIZE = 24
BATCH_SIZE = 4
CONTEXT_FORMAT = 'v0'
TARGETS_FORMAT = None
# endregion

# region Baselines parameters
SHOW_RANKINGS = 10
SHOW_CHOICES = 10
MODELS_RANDOM_SEED = 2

SCORES_NAMES = [
    'average_precision',
    # 'precision_at_10',
    # 'precision_at_100',
    'recall_at_10',
    # 'recall_at_100',
    'reciprocal_best_rank',
    'reciprocal_average_rank',
    'ndcg_at_10',
    # 'ndcg_at_100',
]
# endregion
