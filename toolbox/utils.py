from pickle import load


def to_class_name(name):
    """
    For a name of the format 'abc_efg', returns the corresponding Class name, of the format 'AbcEfg'.

    Args:
        name: str, name of the class in format 'abc_efg'.

    Returns:
        str, name of the class in format 'AbcEfg'.
    """

    return "".join([word.capitalize() for word in name.split("_")])


def load_task(task_name, valid_proportion, test_proportion, ranking_size, batch_size, cross_validation, short,
              folder_path):
    """
    Load a Task using pickle from [folder_path][task_name].pkl

    Args:
        task_name: str, name of the Task to load (eg 'context_free').
        valid_proportion: float, proportion of the validation set.
        test_proportion: float, proportion of the validation set.
        batch_size: int, size of the batches of the modeling task.
        ranking_size: int, size of the rankings.
        cross_validation: bool, cross-validation option.
        short: bool, whether to load the shorten task or not.
        folder_path: str, path of the folder to load from.

    Returns:
        database_creation.modeling_task.Task, loaded object.
    """

    train_proportion = ("%.2f" % (1 - valid_proportion - test_proportion)).split(".")[1]
    valid_proportion = ("%.2f" % valid_proportion).split(".")[1]
    test_proportion = ("%.2f" % test_proportion).split(".")[1]
    suffix = "_" + "-".join([train_proportion, valid_proportion, test_proportion])

    suffix += "_rs" + str(ranking_size)
    suffix += "_bs" + str(batch_size)
    suffix += "_cv" if cross_validation else ""
    suffix += "_short" if short else ""

    task_name = "".join(task_name.split("_"))

    file_name = folder_path + task_name + suffix + '.pkl'

    with open(file_name, 'rb') as file:
        task = load(file)

    print("Task loaded from %s.\n" % file_name)

    return task


def add_task_argument(ap):
    """
    Add to the argument parser the parsing arguments relative to the modeling task.

    Args:
        ap: argparse.ArgumentParser, argument parser to update with the modeling task relative arguments.
    """

    ap.add_argument("-t", "--task", required=True, type=str, help="Name of the modeling task version.")
    ap.add_argument("-vp", "--valid_proportion", default=0.25, type=float, help="Proportion of the validation set.")
    ap.add_argument("-tp", "--test_proportion", default=0.25, type=float, help="Proportion of the test set.")
    ap.add_argument("-rs", "--ranking_size", default=None, type=int, help="Size of the ranking tasks.")
    ap.add_argument("-bs", "--batch_size", default=64, type=int, help="Size of the batches of the task.")
    ap.add_argument("--cross_validation", action='store_true', help="Cross validation option.")
    ap.add_argument("--short", action='store_true', help="Shorten modeling task option.")
