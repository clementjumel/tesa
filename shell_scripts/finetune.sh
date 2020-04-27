#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=10:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/finetune-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/finetune-%j.out

TASK_TYPE=$1
MODELING_TASK=$2
BART=$3
EXPERIMENT=$4

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
echo ""
echo "Results path:"
echo $RESULTS_PATH
echo ""

module load miniconda
source activate base
source activate nlp

tar -xf "$PRETRAINED_MODELS_PATH/$BART.tar.gz" -C $SLURM_TMPDIR
cp -r "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$MODELING_TASK-bin" $SLURM_TMPDIR
cp "$MODELING_TASKS_PATH/$MODELING_TASK.pkl" $SLURM_TMPDIR

rm -rf $RESULTS_PATH
mkdir -p $RESULTS_PATH/tensorboard_logs

cd $SLURM_TMPDIR

if [ $TASK_TYPE == "classification" ]
then
  MAX_EPOCHS=5  # Defautl: 10
  MAX_SENTENCES=32  # Default: 32
  MAX_TOKENS=512  # Default: 1024; works: 512
  UPDATE_FREQ=1  # Default: 1
  LR=1e-05
  TOTAL_NUM_UPDATES=80410  # Default: 1018
  WARMUP_UPDATES=4824  # Default: 61
  SAVE_INTERVAL=1
  VALID_INTERVAL=1

  CUDA_VISIBLE_DEVICES=0,1 python $MASTER_THESIS_PATH/fairseq/train.py "$MODELING_TASK-bin" \
      --max-epoch $MAX_EPOCHS \
      --validate-interval $VALID_INTERVAL \
      --save-interval $SAVE_INTERVAL \
      --max-sentences $MAX_SENTENCES \
      --max-tokens $MAX_TOKENS \
      --update-freq $UPDATE_FREQ \
      --lr $LR \
      --total-num-update $TOTAL_NUM_UPDATES \
      --warmup-updates $WARMUP_UPDATES \
      --restore-file $BART/model.pt \
      --save-dir $RESULTS_PATH \
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
      --lr-scheduler polynomial_decay \
      --best-checkpoint-metric accuracy \
      --maximize-best-checkpoint-metric \
      --skip-invalid-size-inputs-valid-test \
      --disable-validation \
      --find-unused-parameters;

elif [ $TASK_TYPE == "generation" ]
then
  TOTAL_NUM_UPDATES=2190
  WARMUP_UPDATES=100
  LR=3e-05  # Default: 3e-05
  MAX_TOKENS=1024  # Default: 2048
  UPDATE_FREQ=4
  MAX_EPOCHS=10  # Defautl: None
  VALID_INTERVAL=1
  SAVE_INTERVAL=1
  #MAX_SENTENCES=1  # Default: None, works: 4

  CUDA_VISIBLE_DEVICES=0,1 python $MASTER_THESIS_PATH/fairseq/train.py "$MODELING_TASK-bin" \
      --restore-file $BART/model.pt \
      --validate-interval $VALID_INTERVAL \
      --save-interval $SAVE_INTERVAL \
      --no-last-checkpoints \
      --save-dir $RESULTS_PATH \
      --tensorboard-logdir $RESULTS_PATH/tensorboard_logs \
      --max-epoch $MAX_EPOCHS \
      --max-tokens $MAX_TOKENS \
      --update-freq $UPDATE_FREQ \
      --lr-scheduler polynomial_decay \
      --lr $LR \
      --total-num-update $TOTAL_NUM_UPDATES \
      --warmup-updates $WARMUP_UPDATES \
      --task translation \
      --source-lang source \
      --target-lang target \
      --truncate-source \
      --layernorm-embedding \
      --share-all-embeddings \
      --share-decoder-input-output-embed \
      --reset-optimizer --reset-dataloader --reset-meters \
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
      --skip-invalid-size-inputs-valid-test \
      --find-unused-parameters;
      #--max-sentences $MAX_SENTENCES \
      #--reset-lr-scheduler \
fi

echo "Done. Evaluating the checkpoints..."

rm -r *

sh $MASTER_THESIS_PATH/shell_scripts/evaluate.sh \
      $TASK_TYPE \
      $MODELING_TASK \
      $BART \
      $EXPERIMENT \
      $SAVE_INTERVAL \
      $MAX_EPOCHS \
      $SAVE_INTERVAL;
