#!/bin/bash
#SBATCH --job-name=tune-gen
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=5:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/finetune_generation-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/finetune_generation-%j.out

PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data
MODELS_PATH=/network/tmp1/jumelcle/results/models
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
FAIRSEQ_PATH=/network/home/jumelcle/master_thesis/fairseq
TASK=generation

RESULTS_PATH="$MODELS_PATH/$TASK/$1/$2/$3"

module load miniconda
source activate nlp

cp -r "$PREPROCESSED_DATA_PATH/$TASK/$1-bin" $SLURM_TMPDIR
cp $PRETRAINED_MODELS_PATH/$2.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xvf $2.tar.gz

rm -r $RESULTS_PATH
mkdir -p $RESULTS_PATH/tensorboard_logs

MAX_EPOCHS=5  # Defautl: 10
MAX_SENTENCES=4  # Default: None
MAX_TOKENS=1024  # Default: 2048; max length: 946
UPDATE_FREQ=1
LR=3e-05
TOTAL_NUM_UPDATES=4470 # Default: 20000
WARMUP_UPDATES=112  # Default: 500

CUDA_VISIBLE_DEVICES=0,1 python $FAIRSEQ_PATH/train.py "$1-bin" \
    --max-epoch $MAX_EPOCHS \
    --max-sentences $MAX_SENTENCES \
    --max-tokens $MAX_TOKENS \
    --update-freq $UPDATE_FREQ \
    --lr $LR \
    --total-num-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --restore-file $2/model.pt \
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
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --optimizer adam \
    --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
    #--fp16  \
