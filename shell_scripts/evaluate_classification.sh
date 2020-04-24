#!/bin/bash
#SBATCH --job-name=eval_class
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=3:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/evaluate_classification-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/evaluate_classification-%j.out

MODELING_TASK=$1
BART=$2
EXPERIMENT=$3
CHECKPOINT=$4

MODELING_TASKS_PATH=/network/tmp1/jumelcle/results/modeling_task
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
MODELS_PATH=/network/tmp1/jumelcle/results/models
MASTER_THESIS_PATH=/network/home/jumelcle/master_thesis
TASK=classification

RESULTS_PATH="$MODELS_PATH/$TASK/$MODELING_TASK/$BART/$EXPERIMENT"

module load miniconda
source activate base
source activate nlp

cp "$MODELING_TASKS_PATH/$MODELING_TASK.pkl" $SLURM_TMPDIR
cp "$PRETRAINED_MODELS_PATH/$BART.tar.gz" $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xvf $BART.tar.gz
cp "$RESULTS_PATH/$CHECKPOINT.pt" $BART

cp -r "$PREPROCESSED_DATA_PATH/$TASK/$MODELING_TASK-bin" $BART
mv "$BART/$MODELING_TASK-bin/input0" $BART
mv "$BART/$MODELING_TASK-bin/input1" $BART
mv "$BART/$MODELING_TASK-bin/label" $BART

python $MASTER_THESIS_PATH/mt_models.py \
          --full_task_name $MODELING_TASK.pkl \
          --bart \
          --trained_path $BART/ \
          --checkpoint_file $CHECKPOINT.pt \
          --model classifier_bart;
