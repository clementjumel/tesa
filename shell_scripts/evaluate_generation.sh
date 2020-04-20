#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=5:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/evaluate_generation-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/evaluate_generation-%j.out

MODELING_TASKS_PATH=/network/tmp1/jumelcle/results/tasks
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
MODELS_PATH=/network/tmp1/jumelcle/results/models
MASTER_THESIS_PATH=/network/home/jumelcle/master_thesis
TASK=generation

RESULTS_PATH="$MODELS_PATH/$TASK/$1/$2/$3"

module load miniconda
source activate nlp

cp "$MODELING_TASKS_PATH/$1.pkl" $SLURM_TMPDIR
cp $PRETRAINED_MODELS_PATH/$2.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xvf $2.tar.gz
cp -r "$RESULTS_PATH" $2

python $MASTER_THESIS_PATH/mt_models.py \
          --full_task_name $1.pkl \
          --bart \
          --trained_path "$2/" \
          --checkpoint_file $4 \
          --model generator_bart;

rm $RESULTS_PATH/train.hypo
rm $RESULTS_PATH/train.goal
rm $RESULTS_PATH/valid.hypo
rm $RESULTS_PATH/valid.goal

cp train.hypo $RESULTS_PATH
cp train.goal $RESULTS_PATH
cp valid.hypo $RESULTS_PATH
cp valid.goal $RESULTS_PATH
