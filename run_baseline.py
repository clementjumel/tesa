import modeling.models as models
from toolbox.utils import to_class_name, load_task, get_pretrained_model

from argparse import ArgumentParser

from toolbox.parameters import SCORES_NAMES

from toolbox.paths import PRETRAINED_MODELS_PATH, MODELING_TASK_FOR_BASELINES_PATH


def parse_arguments():
    """
    Use arparse to parse the input arguments.

    Returns:
        dict, arguments passed to the script.
    """

    ap = ArgumentParser()

    ap.add_argument("-t", "--task", required=True, type=str, help="Name of the modeling task version.")
    ap.add_argument("-m", "--model", required=True, type=str, help="Name of the model.")
    ap.add_argument("-p", "--pretrained", type=str, help="Name of the pretrained model, if any.")
    ap.add_argument("-e", "--experiment", type=str, help="Name of the experiment; if None, don't save.")

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
    model_name = to_class_name(args['model'])
    pretrained_model_name = args['pretrained']
    experiment_name = args['experiment']

    task = load_task(task_name=task_name, folder_path=MODELING_TASK_FOR_BASELINES_PATH)

    pretrained_model, pretrained_model_dim = get_pretrained_model(pretrained_model_name=pretrained_model_name,
                                                                  folder_path=PRETRAINED_MODELS_PATH)

    model = getattr(models, model_name)(scores_names=SCORES_NAMES,
                                        relevance_level=task.relevance_level,
                                        experiment_name=experiment_name,
                                        pretrained_model=pretrained_model,
                                        pretrained_model_dim=pretrained_model_dim)

    play_baseline(task=task,
                  model=model)


if __name__ == '__main__':
    main()
