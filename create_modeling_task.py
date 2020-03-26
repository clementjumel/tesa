import database_creation.modeling_task as modeling_task

from parameters import \
    MODELING_TASK_NAMES, MIN_ASSIGNMENTS, MIN_ANSWERS, MODELING_TASK_SEED, \
    BATCH_SIZE, DROP_LAST, K_CROSS_VALIDATION, \
    BASELINES_SPLIT_VALID_PROPORTION, BASELINES_SPLIT_TEST_PROPORTION, \
    MODELS_SPLIT_VALID_PROPORTION, MODELS_SPLIT_TEST_PROPORTION, \
    MODELING_TASK_FOR_BASELINES_PATH, MODELING_TASK_FOR_MODELS_PATH


def main():
    """ Creates and saves the modeling tasks. """

    for class_name in MODELING_TASK_NAMES:
        # Saves with only validation and test split (for baseline evaluations)
        task = getattr(modeling_task, class_name)(min_assignments=MIN_ASSIGNMENTS,
                                                  min_answers=MIN_ANSWERS,
                                                  test_proportion=BASELINES_SPLIT_TEST_PROPORTION,
                                                  valid_proportion=BASELINES_SPLIT_VALID_PROPORTION,
                                                  random_seed=MODELING_TASK_SEED)

        task.save_pkl(folder_path=MODELING_TASK_FOR_BASELINES_PATH)

        # Saves with train, validation and test split (for model training)
        task = getattr(modeling_task, class_name)(min_assignments=MIN_ASSIGNMENTS,
                                                  min_answers=MIN_ANSWERS,
                                                  test_proportion=MODELS_SPLIT_TEST_PROPORTION,
                                                  valid_proportion=MODELS_SPLIT_VALID_PROPORTION,
                                                  random_seed=MODELING_TASK_SEED,
                                                  batch_size=BATCH_SIZE,
                                                  drop_last=DROP_LAST,
                                                  k_cross_validation=K_CROSS_VALIDATION)

        task.save_pkl(folder_path=MODELING_TASK_FOR_MODELS_PATH)


if __name__ == '__main__':
    main()
