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
DROP_LAST = False
K_CROSS_VALIDATION = 5
MODELING_TASK_SHORT_SIZE = 1
MODELING_TASK_SEED = 1
# endregion

# region Baselines parameters
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
SHOW_RANKINGS = 5
SHOW_CHOICES = 10
BASELINES_RANDOM_SEED = 2
# endregion
