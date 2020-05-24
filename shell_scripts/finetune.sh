#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --gres=gpu:v100:32gb:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=8:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/finetune-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/finetune-%j.out

# Parameters
TASK_TYPE=$1
CONTEXT_FORMAT=$2
TASK=context-dependent-same-type
TRAIN_PROPORTION=50
VALID_PROPORTION=25
TEST_PROPORTION=25
RANKING_SIZE=24
BATCH_SIZE=4
TARGETS_FORMAT=v0
BART=bart.large.cnn

# Finetuning parameters
if [ $TASK_TYPE == "classification" ]
then
  MAX_EPOCHS=5
  MAX_SENTENCES=8
  UPDATE_FREQ=4
  LR=1e-05
  WARMUP_UPDATES_PERCENT=6
elif [ $TASK_TYPE == "generation" ]
then
  MAX_EPOCHS=10
  MAX_SENTENCES=8
  UPDATE_FREQ=1
  LR=3e-05
  WARMUP_UPDATES_PERCENT=6
fi

EXPERIMENT=ep"$MAX_EPOCHS"_sent"$MAX_SENTENCES"_freq"$UPDATE_FREQ"_lr"$LR"_warm"$WARMUP_UPDATES_PERCENT"

# Paths
MASTER_THESIS_PATH=/network/home/jumelcle/master_thesis
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
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

# Load the preprocessed_data
cp -r "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin" $SLURM_TMPDIR

# Re-initialize the results folder
rm -rf $RESULTS_PATH
mkdir -p $RESULTS_PATH/tensorboard_logs

# Move to SLURM temporary directory
cd $SLURM_TMPDIR

if [ $TASK_TYPE == "classification" ]
then
  # Compute the number of updates
  if [ $CONTEXT_FORMAT == "v0" ]
  then
    NUM_UPDATES_PER_EPOCH=9365
  elif [ $CONTEXT_FORMAT == "v1" ]
  then
    NUM_UPDATES_PER_EPOCH=9435
  elif [ $CONTEXT_FORMAT == "v2" ]
  then
    NUM_UPDATES_PER_EPOCH=9493
  elif [ $CONTEXT_FORMAT == "v3" ]
  then
    NUM_UPDATES_PER_EPOCH=9618
  elif [ $CONTEXT_FORMAT == "v4" ]
  then
    NUM_UPDATES_PER_EPOCH=9435
  elif [ $CONTEXT_FORMAT == "va" ]
  then
    NUM_UPDATES_PER_EPOCH=4076
  elif [ $CONTEXT_FORMAT == "vb" ]
  then
    NUM_UPDATES_PER_EPOCH=6690
  elif [ $CONTEXT_FORMAT == "vc" ]
  then
    NUM_UPDATES_PER_EPOCH=2574
  else
    NUM_UPDATES_PER_EPOCH=9365
  fi

  TOTAL_NUM_UPDATES=$(($NUM_UPDATES_PER_EPOCH * $MAX_EPOCHS / $UPDATE_FREQ))
  WARMUP_UPDATES=$(($WARMUP_UPDATES_PERCENT * $TOTAL_NUM_UPDATES / 100))

  # Print the parameters
  echo "Finetuning parameters:"; echo $MAX_EPOCHS; echo $MAX_SENTENCES; echo $UPDATE_FREQ; echo $LR;
  echo $WARMUP_UPDATES_PERCENT; echo $NUM_UPDATES_PER_EPOCH; echo $TOTAL_NUM_UPDATES; echo $WARMUP_UPDATES; echo

  CUDA_VISIBLE_DEVICES=0,1 python $MASTER_THESIS_PATH/fairseq/train.py "$FULL_TASK-bin" \
      --max-epoch $MAX_EPOCHS \
      --max-sentences $MAX_SENTENCES \
      --max-tokens 1024 \
      --update-freq $UPDATE_FREQ \
      --lr-scheduler polynomial_decay \
      --lr $LR \
      --total-num-update $TOTAL_NUM_UPDATES \
      --warmup-updates $WARMUP_UPDATES \
      --restore-file $BART/model.pt \
      --save-dir $RESULTS_PATH \
      --tensorboard-logdir $RESULTS_PATH/tensorboard_logs \
      --task sentence_prediction \
      --add-prev-output-tokens \
      --layernorm-embedding \
      --share-all-embeddings \
      --share-decoder-input-output-embed \
      --reset-optimizer \
      --reset-dataloader \
      --reset-meters \
      --required-batch-size-multiple 1 \
      --init-token 0 \
      --arch bart_large \
      --criterion sentence_prediction \
      --num-classes 2 \
      --dropout 0.1 \
      --attention-dropout 0.1 \
      --weight-decay 0.01 \
      --optimizer adam \
      --adam-betas "(0.9, 0.98)" \
      --adam-eps 1e-08 \
      --clip-norm 0.0 \
      --best-checkpoint-metric accuracy \
      --maximize-best-checkpoint-metric \
      --memory-efficient-fp16 \
      --no-last-checkpoints \
      --skip-invalid-size-inputs-valid-test \
      --find-unused-parameters;

elif [ $TASK_TYPE == "generation" ]
then
  # Compute the number of updates
  if [ $CONTEXT_FORMAT == "v0" ]
  then
    NUM_UPDATES_PER_EPOCH=831
  elif [ $CONTEXT_FORMAT == "v1" ]
  then
    NUM_UPDATES_PER_EPOCH=838
  elif [ $CONTEXT_FORMAT == "v2" ]
  then
    NUM_UPDATES_PER_EPOCH=852
  elif [ $CONTEXT_FORMAT == "v3" ]
  then
    NUM_UPDATES_PER_EPOCH=866
  elif [ $CONTEXT_FORMAT == "v4" ]
  then
    NUM_UPDATES_PER_EPOCH=838
  elif [ $CONTEXT_FORMAT == "va" ]
  then
    NUM_UPDATES_PER_EPOCH=342
  elif [ $CONTEXT_FORMAT == "vb" ]
  then
    NUM_UPDATES_PER_EPOCH=538
  elif [ $CONTEXT_FORMAT == "vc" ]
  then
    NUM_UPDATES_PER_EPOCH=288
  else
    NUM_UPDATES_PER_EPOCH=831
  fi

  # Compute the number of updates
  TOTAL_NUM_UPDATES=$(($NUM_UPDATES_PER_EPOCH * $MAX_EPOCHS / $UPDATE_FREQ))
  WARMUP_UPDATES=$(($WARMUP_UPDATES_PERCENT * $TOTAL_NUM_UPDATES / 100))

  # Print the parameters
  echo "Finetuning parameters:"; echo $MAX_EPOCHS; echo $MAX_SENTENCES; echo $UPDATE_FREQ; echo $LR;
  echo $WARMUP_UPDATES_PERCENT; echo $NUM_UPDATES_PER_EPOCH; echo $TOTAL_NUM_UPDATES; echo $WARMUP_UPDATES; echo

  # Run the finetuning
  CUDA_VISIBLE_DEVICES=0,1 python $MASTER_THESIS_PATH/fairseq/train.py "$FULL_TASK-bin" \
      --max-epoch $MAX_EPOCHS \
      --max-sentences $MAX_SENTENCES \
      --max-tokens 1024 \
      --update-freq $UPDATE_FREQ \
      --lr-scheduler polynomial_decay \
      --lr $LR \
      --total-num-update $TOTAL_NUM_UPDATES \
      --warmup-updates $WARMUP_UPDATES \
      --restore-file $BART/model.pt \
      --save-dir $RESULTS_PATH \
      --tensorboard-logdir $RESULTS_PATH/tensorboard_logs \
      --task translation \
      --source-lang source \
      --target-lang target \
      --truncate-source \
      --layernorm-embedding \
      --share-all-embeddings \
      --share-decoder-input-output-embed \
      --reset-optimizer \
      --reset-dataloader \
      --reset-meters \
      --required-batch-size-multiple 1 \
      --arch bart_large \
      --criterion label_smoothed_cross_entropy \
      --label-smoothing 0.1 \
      --dropout 0.1 \
      --attention-dropout 0.1 \
      --weight-decay 0.01 \
      --optimizer adam \
      --adam-betas "(0.9, 0.999)" \
      --adam-eps 1e-08 \
      --clip-norm 0.1 \
      --fp16 \
      --no-last-checkpoints \
      --find-unused-parameters;
fi

# Remove checkpoint_best.pt
rm $RESULTS_PATH/checkpoint_best.pt

echo "Done."; echo

sbatch --partition=unkillable /network/home/jumelcle/master_thesis/shell_scripts/rank.sh $1 $2 $EXPERIMENT
