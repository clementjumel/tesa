from database_creation.modeling_task import ModelingTask
import modeling.models as models
from utils import parse_arguments, get_pretrained_model

from parameters import \
    SAVE, VERBOSE, MODELING_TASK_FOR_BASELINES_PATH, \
    SCORES_NAMES, EXPLAIN_SAMPLES, EXPLAIN_CHOICES


def play_baseline(task, model, verbose):
    """
    Performs the preview of the data and the evaluation of a baseline.

    Args:
        task: modeling_task.ModelingTask, task to evaluate the baseline upon.
        model: models.Baseline, baseline to evaluate.
        verbose: bool, verbose option.
    """

    task.preview_data(model=model, include_train=False, include_valid=True)
    task.valid_model(model=model)

    if verbose:
        model.display_metrics()
        task.explain_model(model=model, explain_samples=EXPLAIN_SAMPLES, explain_choices=EXPLAIN_CHOICES)


def main():
    """ Makes a baseline run on a task. """

    args = parse_arguments()

    task_name = args['task_name']
    model_name = args['model_name']

    if SAVE and "experiment_name" in args:
        experiment_name = args['experiment_name']
    elif SAVE and "experiment_name" not in args:
        raise Exception("No experiment_name specified, cannot save.")
    else:
        experiment_name = None

    if "pretrained_model_name" in args:
        pretrained_model, pretrained_model_dim = \
            get_pretrained_model(pretrained_model_name=args['pretrained_model_name'])
    else:
        pretrained_model, pretrained_model_dim = None, None

    task = ModelingTask.load_pkl(task_name=task_name, folder_path=MODELING_TASK_FOR_BASELINES_PATH)

    model = getattr(models, model_name)(scores_names=SCORES_NAMES,
                                        relevance_level=task.relevance_level,
                                        experiment_name=experiment_name,
                                        pretrained_model=pretrained_model,
                                        pretrained_model_dim=pretrained_model_dim)

    play_baseline(task=task, model=model, verbose=VERBOSE)


if __name__ == '__main__':
    main()
