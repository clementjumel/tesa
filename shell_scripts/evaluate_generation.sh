#!/bin/bash
#SBATCH --job-name=eval_gen
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=16G
#SBATCH --time=2:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/evaluate_generation-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/evaluate_generation-%j.out

module load miniconda
source activate base
source activate nlp

MODELING_TASK=$1
BART=$2
EXPERIMENT=$3

TASK=generation

MASTER_THESIS_PATH=/network/home/jumelcle/master_thesis
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
MODELING_TASKS_PATH=/network/tmp1/jumelcle/results/modeling_task
MODELS_PATH=/network/tmp1/jumelcle/results/models
RESULTS_PATH="$MODELS_PATH/$TASK/$MODELING_TASK/$BART/$EXPERIMENT"

cp "$MODELING_TASKS_PATH/$MODELING_TASK.pkl" $SLURM_TMPDIR
cp "$PRETRAINED_MODELS_PATH/$BART.tar.gz" $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xvf $BART.tar.gz

MAX_EPOCHS=10
SAVE_INTERVAL=1

for i in $(eval echo "{$SAVE_INTERVAL..$MAX_EPOCHS..$SAVE_INTERVAL}")
do
   CHECKPOINT=checkpoint$i
   cp "$RESULTS_PATH/$CHECKPOINT.pt" $BART

   python $MASTER_THESIS_PATH/mt_models.py \
             --full_task_name $MODELING_TASK.pkl \
             --bart \
             --trained_path $BART/ \
             --checkpoint_file $CHECKPOINT.pt \
             --model generator_bart;

   rm -r $RESULTS_PATH/$CHECKPOINT
   mkdir $RESULTS_PATH/$CHECKPOINT

   mv train.source $RESULTS_PATH/$CHECKPOINT
   mv train.targets $RESULTS_PATH/$CHECKPOINT
   mv train.entities $RESULTS_PATH/$CHECKPOINT
   mv train.hypotheses $RESULTS_PATH/$CHECKPOINT

   mv valid.source $RESULTS_PATH/$CHECKPOINT
   mv valid.targets $RESULTS_PATH/$CHECKPOINT
   mv valid.entities $RESULTS_PATH/$CHECKPOINT
   mv valid.hypotheses $RESULTS_PATH/$CHECKPOINT

   echo "$CHECKPOINT evaluated"
done
