import database_creation.modeling_task as modeling_task

# Names of the subclasses of modeling_task.Task
class_names = ['ContextFreeTask',
               'ContextFreeSameTypeTask',
               'ContextDependentTask',
               'ContextDependentSameTypeTask',
               'FullHybridTask',
               'HybridTask',
               'HybridSameTypeTask']

# Global parameters of the task
min_assignments = 5
min_answers = 2
batch_size = 32
drop_last = False
k_cross_validation = 0
random_seed = 1

# Parameters dependent on the training set-up
test_proportion = 0.25
valid_proportion = 0.25
folder_path = 'results/modeling_task/training_split/'

for class_name in class_names:
    task = getattr(modeling_task, class_name)(min_assignments=min_assignments,
                                              min_answers=min_answers,
                                              test_proportion=test_proportion,
                                              valid_proportion=valid_proportion,
                                              batch_size=batch_size,
                                              drop_last=drop_last,
                                              k_cross_validation=k_cross_validation,
                                              random_seed=random_seed)

    task.save_pkl(folder_path=folder_path)

# Parameters dependent on the evaluation set-up
test_proportion = 0.5
valid_proportion = 0.5
folder_path = 'results/modeling_task/evaluation_split/'

for class_name in class_names:
    task = getattr(modeling_task, class_name)(min_assignments=min_assignments,
                                              min_answers=min_answers,
                                              test_proportion=test_proportion,
                                              valid_proportion=valid_proportion,
                                              batch_size=batch_size,
                                              drop_last=drop_last,
                                              k_cross_validation=k_cross_validation,
                                              random_seed=random_seed)

    task.save_pkl(folder_path=folder_path)
