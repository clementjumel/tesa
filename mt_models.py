"""
modeling_task script to run Models on a task.

Usages:
    python mt_models.py -t context_free_same_type -rs 32 -bs 8 -cf v0 -m frequency
    python mt_models.py -t context_free_same_type -rs 32 -bs 8 -cf v0 -m summaries_average_embedding --word2vec
"""

import modeling.models as models
from toolbox.utils import to_class_name, standard_parser, add_task_arguments, load_task, get_trained_model
from toolbox.parameters import SCORES_NAMES, MODELS_RANDOM_SEED
from toolbox.paths import PRETRAINED_MODELS_PATH, MODELING_TASK_RESULTS_PATH, TENSORBOARD_LOGS_PATH


def parse_arguments():
    """ Use arparse to parse the input arguments and return it as a argparse.ArgumentParser. """

    ap = standard_parser()
    add_task_arguments(ap)

    ap.add_argument("-m", "--model", required=True, type=str, help="Name of the model.")
    ap.add_argument("-w", "--word2vec", action="store_true", help="Load Word2Vec embedding.")
    ap.add_argument("-b", "--bart", action="store_true", help="Load a BART model.")
    ap.add_argument("-e", "--experiment", default=None, type=str, help="Name of the experiment.")
    ap.add_argument("--trained_path", default=None, type=str, help="Path to the trained model folder.")
    ap.add_argument("--checkpoint_file", default=None, type=str, help="Name of BART's checkpoint file.")
    ap.add_argument("--show", action="store_true", help="Option to show some results.")

    return ap.parse_args()


def main():
    """ Makes a model run on a task. """

    args = parse_arguments()

    task = load_task(args=args, folder_path=MODELING_TASK_RESULTS_PATH)
    trained_model = get_trained_model(args=args, folder_path=PRETRAINED_MODELS_PATH)

    model_name = to_class_name(args.model)
    task_name = args.full_task_name.split('/')[-1] if args.full_task_name is not None else ''

    model = getattr(models, model_name)(scores_names=SCORES_NAMES,
                                        relevance_level=task.relevance_level,
                                        trained_model=trained_model,
                                        task_name=task_name,
                                        tensorboard_logs_path=TENSORBOARD_LOGS_PATH,
                                        experiment_name=args.experiment,
                                        random_seed=MODELS_RANDOM_SEED)

    model.play(task=task, show=args.show)


if __name__ == '__main__':
    main()
