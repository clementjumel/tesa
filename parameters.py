# region Miscellaneous
VERBOSE = True
SAVE = True
# endregion

# region Paths
MODELING_TASK_FOR_BASELINES_PATH = "results/modeling_task/baselines_split/"
MODELING_TASK_FOR_MODELS_PATH = "results/modeling_task/models_split/"

BASELINES_RESULTS_PATH = 'results/baselines/'
MODELS_RESULTS_PATH = 'results/models/'
# endregion

# region Annotation task parameters
# TODO
# endregion

# region Modeling task parameters
MODELING_TASK_NAMES = [
    'ContextFreeTask',
    'ContextFreeSameTypeTask',
    'ContextDependentTask',
    'ContextDependentSameTypeTask',
    'FullHybridTask',
    'HybridTask',
    'HybridSameTypeTask'
]

MIN_ASSIGNMENTS = 5
MIN_ANSWERS = 2
BATCH_SIZE = 64
DROP_LAST = False
K_CROSS_VALIDATION = 0
MODELING_TASK_SEED = 1

BASELINES_SPLIT_VALID_PROPORTION = 0.5
BASELINES_SPLIT_TEST_PROPORTION = 0.5

MODELS_SPLIT_VALID_PROPORTION = 0.25
MODELS_SPLIT_TEST_PROPORTION = 0.25
# endregion

# region Modeling parameters
SCORES_NAMES = [
    'average_precision',
    'precision_at_10',
    'precision_at_100',
    'recall_at_10',
    'recall_at_100',
    'reciprocal_best_rank',
    'reciprocal_average_rank',
    'ndcg_at_10',
    'ndcg_at_100',
]

# region Baselines parameters
# TODO
# endregion

# region Models parameters
# TODO
# endregion
# endregion
