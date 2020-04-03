import modeling.models as models
from toolbox.utils import to_class_name, load_task, get_pretrained_model

from argparse import ArgumentParser

from toolbox.parameters import SCORES_NAMES, MODELS_RANDOM_SEED

from toolbox.paths import PRETRAINED_MODELS_PATH, MODELING_TASK_FOR_BASELINES_PATH, TENSORBOARD_LOGS_BASELINES_PATH


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("-t", "--task", required=True, type=str, help="Name of the modeling task version.")
    ap.add_argument("-bs", "--batch_size", default=64, type=int, help="Size of the batches of the task.")
    ap.add_argument("--cross_validation", action='store_true', help="Cross validation option.")
    ap.add_argument("--short", action='store_true', help="Shorten modeling task option.")
    ap.add_argument("-m", "--model", required=True, type=str, help="Name of the model.")
    ap.add_argument("-p", "--pretrained", type=str, help="Name of the pretrained model, if any.")
    ap.add_argument("-e", "--experiment", required=True, type=str, help="Name of the experiment.")

    args = vars(ap.parse_args())

    return args


def play_baseline(task, model):
    """
    Performs the preview of the data and the evaluation of a baseline.

    Args:
        task: modeling_task.ModelingTask, task to evaluate the baseline upon.
        model: models.Baseline, baseline to evaluate.
    """

    task.preview_data(model=model, include_train=False, include_valid=True)
    task.valid_model(model=model)


def main():
    """ Makes a baseline run on a task. """

    args = parse_arguments()

    task_name = args['task']
    batch_size = args['batch_size']
    cross_validation = args['cross_validation']
    short = args['short']
    model_name = to_class_name(args['model'])
    pretrained_model_name = args['pretrained']
    experiment_name = args['experiment']

    task = load_task(task_name=task_name,
                     batch_size=batch_size,
                     cross_validation=cross_validation,
                     short=short,
                     folder_path=MODELING_TASK_FOR_BASELINES_PATH)

    pretrained_model, pretrained_model_dim = get_pretrained_model(pretrained_model_name=pretrained_model_name,
                                                                  folder_path=PRETRAINED_MODELS_PATH)

    model = getattr(models, model_name)(scores_names=SCORES_NAMES,
                                        relevance_level=task.relevance_level,
                                        pretrained_model=pretrained_model,
                                        pretrained_model_dim=pretrained_model_dim,
                                        tensorboard_logs_path=TENSORBOARD_LOGS_BASELINES_PATH,
                                        experiment_name=experiment_name,
                                        random_seed=MODELS_RANDOM_SEED)

    play_baseline(task=task,
                  model=model)


if __name__ == '__main__':
    main()
