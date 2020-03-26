from database_creation.modeling_task import ModelingTask
import modeling.models as models
from utils import parse_arguments, play_baseline
import parameters as p

task_name, model_name = parse_arguments()

task = ModelingTask.load_pkl(task_name=task_name, path=p.modeling_task_for_baselines_path)
model = getattr(models, model_name)

play_baseline(task=task, model=model, save=p.save, verbose=p.verbose)
