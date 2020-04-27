#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/evaluate-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/evaluate-%j.out

TASK_TYPE=$1
MODELING_TASK=$2
BART=$3
EXPERIMENT=$4
CHECKPOINT_START=$5
CHEKCPOINT_STEP=$6
CHECKPOINT_END=$7

MASTER_THESIS_PATH=/network/home/jumelcle/master_thesis
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data
MODELING_TASKS_PATH=/network/tmp1/jumelcle/results/modeling_task
MODELS_PATH=/network/tmp1/jumelcle/results/models
RESULTS_PATH="$MODELS_PATH/$TASK_TYPE/$MODELING_TASK/$BART/$EXPERIMENT"

echo "Parameters:"
echo $TASK_TYPE
echo $MODELING_TASK
echo $BART
echo $EXPERIMENT
echo $CHECKPOINT_START
echo $CHEKCPOINT_STEP
echo $CHECKPOINT_END
echo ""

module load miniconda
source activate base
source activate nlp

tar -xf "$PRETRAINED_MODELS_PATH/$BART.tar.gz" -C $SLURM_TMPDIR
cp "$MODELING_TASKS_PATH/$MODELING_TASK.pkl" $SLURM_TMPDIR

cd $SLURM_TMPDIR

for i in $(eval echo "{$CHECKPOINT_START..$CHEKCPOINT_STEP..$CHECKPOINT_END}")
do
  CHECKPOINT=checkpoint$i

  cp "$RESULTS_PATH/$CHECKPOINT.pt" $BART

  echo ""
  echo "Evaluating checkpoint:"
  echo $CHECKPOINT
  echo ""

  if [ $TASK_TYPE == "classification" ]
  then
    cp -r "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$MODELING_TASK-bin/input0" \
          "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$MODELING_TASK-bin/input1" \
          "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$MODELING_TASK-bin/label" \
          $BART

    python $MASTER_THESIS_PATH/mt_models.py \
          --full_task_name $MODELING_TASK.pkl \
          --bart \
          --trained_path $BART/ \
          --checkpoint_file $CHECKPOINT.pt \
          --model classifier_bart;

  elif [ $TASK_TYPE == "generation" ]
  then
    python $MASTER_THESIS_PATH/mt_models.py \
          --full_task_name $MODELING_TASK.pkl \
          --bart \
          --trained_path $BART/ \
          --checkpoint_file $CHECKPOINT.pt \
          --model generator_bart;

    rm -rf $RESULTS_PATH/$CHECKPOINT
    mkdir $RESULTS_PATH/$CHECKPOINT

    mv train.source $RESULTS_PATH/$CHECKPOINT
    mv train.targets $RESULTS_PATH/$CHECKPOINT
    mv train.entities $RESULTS_PATH/$CHECKPOINT
    mv train.hypotheses $RESULTS_PATH/$CHECKPOINT

    mv valid.source $RESULTS_PATH/$CHECKPOINT
    mv valid.targets $RESULTS_PATH/$CHECKPOINT
    mv valid.entities $RESULTS_PATH/$CHECKPOINT
    mv valid.hypotheses $RESULTS_PATH/$CHECKPOINT
  fi

  echo "$CHECKPOINT evaluated"
done

echo "Done."
