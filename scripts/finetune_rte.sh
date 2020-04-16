#!/bin/bash
#SBATCH --job-name=finetune_rte
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00
#SBATCH --error=/network/tmp1/jumelcle/errors/finetune_rte-%j.err
#SBATCH --output=/network/tmp1/jumelcle/outputs/finetune_rte-%j.out

module load miniconda
source activate nlp

cp -r /network/tmp1/jumelcle/results/finetuning_preprocessed_data/RTE-bin_$1 $SLURM_TMPDIR
cp /network/tmp1/jumelcle/pretrained_models/$2.tar.gz $SLURM_TMPDIR

cd $SLURM_TMPDIR

tar -xvf $2.tar.gz
rm $2.tar.gz

rm -r /network/tmp1/jumelcle/results/models/$3
mkdir /network/tmp1/jumelcle/results/models/$3

FAIRSEQ_PATH=/network/home/jumelcle/master_thesis/fairseq
BART_PATH=$2/model.pt

MAX_EPOCHS=5  # Defautl: 10
TOTAL_NUM_UPDATES=10180  # Default: 1018
WARMUP_UPDATES=610
LR=1e-05
NUM_CLASSES=2
MAX_SENTENCES=32  # Default: 32
MAX_TOKENS=512  # Default: 1024; works: 512

CUDA_VISIBLE_DEVICES=0,1 python $FAIRSEQ_PATH/train.py RTE-bin_$1/ \
    --restore-file $BART_PATH \
    --max-sentences $MAX_SENTENCES \
    --max-tokens $MAX_TOKENS \
    --task sentence_prediction \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --arch bart_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch $MAX_EPOCHS \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --save-dir "/network/tmp1/jumelcle/results/models/$3" \
    --skip-invalid-size-inputs-valid-test \
    --disable-validation

    #--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
