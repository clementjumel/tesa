""" Parameters of the various scripts. """

# region Miscellaneous
SILENT = False
SAVE = True
# endregion

# region Annotation task parameters
YEARS = [2006, 2007]
MAX_TUPLE_SIZE = 6
SHORT = False
RANDOM = None
WRITE_DEBUG = True
ANNOTATION_TASK_SEED = 0

LOAD_QUERIES = False
LOAD_WIKI = True
WIKIPEDIA_FILE_NAME = "wikipedia_global"
CORRECT_WIKI = True

TASK_BATCHES = 1
TASK_BATCH_SIZE = 400
# endregion

# region Modeling task parameters
MODELING_TASK_NAMES = [
    'ContextFree',
    # 'ContextFreeSameType',
    # 'ContextDependent',
    # 'ContextDependentSameType',
    # 'FullHybrid',
    # 'Hybrid',
    # 'HybridSameType'
]

MIN_ASSIGNMENTS = 5
MIN_ANSWERS = 2
EXCLUDE_PILOT = False

BATCH_SIZE = 64
DROP_LAST = False
CROSS_VALIDATION = None

MODELING_TASK_SEED = 1

BASELINES_SPLIT_VALID_PROPORTION = 0.5
BASELINES_SPLIT_TEST_PROPORTION = 0.5

MODELS_SPLIT_VALID_PROPORTION = 0.25
MODELS_SPLIT_TEST_PROPORTION = 0.25
# endregion

# region Modeling parameters
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

EXPLAIN_EXAMPLES = 5
EXPLAIN_CHOICES = 10

EPOCHS = 1
UPDATE_EVERY = 100
REGRESSION = False
# endregion
