#!/bin/bash
#SBATCH --job-name=rank
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=12:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/rank-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/rank-%j.out

# Parameters
TASK_TYPE=$1
CONTEXT_FORMAT=$2
EXPERIMENT=$3
TASK=context-dependent-same-type
TRAIN_PROPORTION=50
VALID_PROPORTION=25
TEST_PROPORTION=25
RANKING_SIZE=24
BATCH_SIZE=4
TARGETS_FORMAT=v0
BART=bart.large.cnn

# Paths
MASTER_THESIS_PATH=/network/home/jumelcle/master_thesis
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data
TASKS_PATH=/network/tmp1/jumelcle/results/modeling_task
CHECKPOINTS_PATH=/network/tmp1/jumelcle/results/checkpoints

# Recover full paths/names
FULL_TASK="$TASK"_"$TRAIN_PROPORTION"-"$VALID_PROPORTION"-"$TEST_PROPORTION"_rs"$RANKING_SIZE"_bs"$BATCH_SIZE"_cf-"$CONTEXT_FORMAT"_tf-"$TARGETS_FORMAT"
RESULTS_PATH="$CHECKPOINTS_PATH/$TASK_TYPE/$FULL_TASK/$EXPERIMENT"

# Print the parameters
echo "Parameters:"; echo $TASK_TYPE $CONTEXT_FORMAT $EXPERIMENT; echo
echo "Results path:"; echo $RESULTS_PATH; echo

# Load miniconda
module load miniconda
source activate base
source activate nlp

# Load pretrained BART
tar -xf "$PRETRAINED_MODELS_PATH/$BART.tar.gz" -C $SLURM_TMPDIR

# Load the task
cp "$TASKS_PATH/$FULL_TASK.pkl" $SLURM_TMPDIR

# Load the preprocessed data if necessary
if [ $TASK_TYPE == "classification" ]
then
  cp -r "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin/input0" \
        "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin/input1" \
        "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin/label" \
        $SLURM_TMPDIR/$BART
fi

# Move to SLURM temporary directory
cd $SLURM_TMPDIR

# Makes sure we don't compute *.pt if there is no checkpoint file
shopt -s nullglob

for FULL_CHECKPOINT in $RESULTS_PATH/*.pt
do
  # Load the checkpoint
  cp $FULL_CHECKPOINT $BART

  # Recover the name of the checkpoint
  HALF_CHECKPOINT=${FULL_CHECKPOINT%.*}
  CHECKPOINT=${HALF_CHECKPOINT##*/}

  # Print the checkpoint
  echo; echo "Evaluating $CHECKPOINT"

  if [ $TASK_TYPE == "classification" ]
  then
    # Run the ranking
    python $MASTER_THESIS_PATH/run_model.py \
          --task $TASK \
          --context_format $CONTEXT_FORMAT \
          --targets_format $TARGETS_FORMAT \
          --task_path "" \
          --model classifier-bart \
          --bart \
          --pretrained_path $BART \
          --checkpoint_file $CHECKPOINT.pt;

  elif [ $TASK_TYPE == "generation" ]
  then
    # Run the ranking
    python $MASTER_THESIS_PATH/run_model.py \
          -t $TASK \
          --context_format $CONTEXT_FORMAT \
          --targets_format $TARGETS_FORMAT \
          --task_path "" \
          --model generator-bart \
          --bart \
          --pretrained_path $BART \
          --checkpoint_file $CHECKPOINT.pt \
          --results_path $RESULTS_PATH/$CHECKPOINT;
  fi
done

echo "Done."; echo
