#!/bin/bash
#SBATCH --job-name=finetune_cnndm
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=5:00:00
#SBATCH --error=/network/tmp1/jumelcle/errors/finetune_cnndm-%j.err
#SBATCH --output=/network/tmp1/jumelcle/outputs/finetune_cnndm-%j.out

module load miniconda
source activate nlp

cp -r /network/tmp1/jumelcle/results/finetuning_preprocessed_data/cnndm-bin_$1 $SLURM_TMPDIR
cp /network/tmp1/jumelcle/pretrained_models/$2.tar.gz $SLURM_TMPDIR

cd $SLURM_TMPDIR

tar -xvf $2.tar.gz
rm $2.tar.gz

FAIRSEQ_PATH=/network/home/jumelcle/master_thesis/fairseq
BART_PATH=$2/model.pt

TOTAL_NUM_UPDATES=200 # Default: 20000
WARMUP_UPDATES=5  # Default: 500
LR=3e-05
MAX_TOKENS=1024  # Default: 2048
UPDATE_FREQ=4
MAX_EPOCHS=10

CUDA_VISIBLE_DEVICES=0,1 python $FAIRSEQ_PATH/train.py cnndm-bin_$1 \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --skip-invalid-size-inputs-valid-test \
    --max-epoch $MAX_EPOCHS \
    --find-unused-parameters;

    # --fp16 --update-freq $UPDATE_FREQ \

rm -r /network/tmp1/jumelcle/results/models/$3
mkdir /network/tmp1/jumelcle/results/models/$3
cp -r checkpoints /network/tmp1/jumelcle/results/models/$3
