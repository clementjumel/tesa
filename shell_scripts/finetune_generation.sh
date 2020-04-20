#!/bin/bash
#SBATCH --job-name=tune-gen
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=5:00:00
#SBATCH --error=/network/tmp1/jumelcle/errors/finetune_generation-%j.err
#SBATCH --output=/network/tmp1/jumelcle/outputs/finetune_generation-%j.out

PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data
MODELS_PATH=/network/tmp1/jumelcle/results/models
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
FAIRSEQ_PATH=/network/home/jumelcle/master_thesis/fairseq
TASK=generation

module load miniconda
source activate nlp

cp -r "$PREPROCESSED_DATA_PATH/$TASK/$1" $SLURM_TMPDIR
cp $PRETRAINED_MODELS_PATH/$2.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xvf $2.tar.gz

rm -r "$MODELS_PATH/$1/$2/$3"
mkdir -p "$MODELS_PATH/$1/$2/$3/tensorboard_logs"

TOTAL_NUM_UPDATES=200 # Default: 20000
WARMUP_UPDATES=5  # Default: 500
LR=3e-05
MAX_TOKENS=1024  # Default: 2048
UPDATE_FREQ=4
MAX_EPOCHS=10

CUDA_VISIBLE_DEVICES=0,1 python $FAIRSEQ_PATH/train.py $1 \
    --restore-file $2/model.pt \
    --max-tokens $MAX_TOKENS \
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
    --lr $LR \
    --total-num-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --max-epoch $MAX_EPOCHS \
    --skip-invalid-size-inputs-valid-test \
    --save-dir "$MODELS_PATH/$1/$3" \
    --tensorboard-logdir "$MODELS_PATH/$1/$3/tensorboard_logs" \
    --find-unused-parameters;
    # --fp16 --update-freq $UPDATE_FREQ \
