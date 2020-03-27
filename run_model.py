from database_creation.modeling_task import ModelingTask
import modeling.models as models

from toolbox.utils import parse_arguments, get_experiment_name, get_pretrained_model

# TODO: complete
from toolbox.parameters import SAVE, SILENT, \
    SCORES_NAMES, EXPLAIN_EXAMPLES, EXPLAIN_CHOICES
###

from toolbox.paths import PRETRAINED_MODELS_PATH, MODELING_TASK_FOR_MODELS_PATH


def play_model(task, model, silent, explain_examples, explain_choices):
    """
    Performs the preview of the data, the training and the validation of a model.

    Args:
        task: modeling_task.ModelingTask, task to evaluate the model upon.
        model: models.Model, model to evaluate.
        silent: bool, silence option.
        explain_examples: int, number of examples to show in details.
        explain_choices: int, number of choices to show for each explained example.
    """

    task.preview_data(model=model, include_train=True, include_valid=False)
    # TODO: complete
    task.train_model(model=model)
    task.valid_model(model=model)
    ###

    if not silent:
        model.display_metrics()
        task.explain_model(model=model, explain_examples=explain_examples, explain_choices=explain_choices)


def main():
    """ Makes a model run on a task. """

    args = parse_arguments()

    task_name = args['task_name']
    model_name = args['model_name']
    experiment_name = get_experiment_name(args=args, save=SAVE)
    pretrained_model, pretrained_model_dim = get_pretrained_model(args=args,
                                                                  silent=SILENT,
                                                                  folder_path=PRETRAINED_MODELS_PATH)

    task = ModelingTask.load_pkl(task_name=task_name, folder_path=MODELING_TASK_FOR_MODELS_PATH)

    # TODO: complete parameters
    model = getattr(models, model_name)(scores_names=SCORES_NAMES,
                                        relevance_level=task.relevance_level,
                                        experiment_name=experiment_name,
                                        pretrained_model=pretrained_model,
                                        pretrained_model_dim=pretrained_model_dim)

    # TODO: complete parameters
    play_model(task=task,
               model=model,
               silent=SILENT,
               explain_examples=EXPLAIN_EXAMPLES,
               explain_choices=EXPLAIN_CHOICES)


if __name__ == '__main__':
    main()
