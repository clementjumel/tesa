import database_creation.modeling_task as modeling_task
import parameters as p

for class_name in p.modeling_task_names:
    # Saves with only validation and test split (for baseline evaluations)
    task = getattr(modeling_task, class_name)(min_assignments=p.min_assignments,
                                              min_answers=p.min_answers,
                                              test_proportion=p.evaluation_test_proportion,
                                              valid_proportion=p.evaluation_valid_proportion,
                                              batch_size=p.batch_size,
                                              drop_last=p.drop_last,
                                              k_cross_validation=p.k_cross_validation,
                                              random_seed=p.modeling_task_random_seed)

    task.save_pkl(folder_path=p.evaluation_folder_path)

    # Saves with train, validation and test split (for model training)
    task = getattr(modeling_task, class_name)(min_assignments=p.min_assignments,
                                              min_answers=p.min_answers,
                                              test_proportion=p.training_test_proportion,
                                              valid_proportion=p.training_valid_proportion,
                                              batch_size=p.batch_size,
                                              drop_last=p.drop_last,
                                              k_cross_validation=p.k_cross_validation,
                                              random_seed=p.modeling_task_random_seed)

    task.save_pkl(folder_path=p.training_folder_path)
