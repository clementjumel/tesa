#!/bin/bash
#SBATCH --job-name=vanilla                            # Name of the job
#SBATCH --partition=unkillable                        # Ask for unkillable job
#SBATCH --gres=gpu:1                                  # Ask for 1 GPU
#SBATCH --mem-per-gpu=32G                             # Memory per GPU
#SBATCH --mem=10G                                     # Ask for 10 GB of RAM
#SBATCH --time=1:00:00                                # The job will run for 1 hour
#SBATCH --error=/network/home/jumelcle/master_thesis/rte_vanilla/errors/slurm-%j.err  # Write the error on home
#SBATCH --output=/network/home/jumelcle/master_thesis/rte_vanilla/outputs/slurm-%j.out  # Write the log on home

# 1. Load the environment
source activate nlp

# 2. Copy the data on the compute node
cp -r /network/tmp1/jumelcle/preprocessed_data/RTE-bin_vanilla $SLURM_TMPDIR
cp /network/tmp1/jumelcle/pretrained_models/bart.large.tar.gz $SLURM_TMPDIR
tar -xvf $SLURM_TMPDIR/bart.large.tar.gz -C $SLURM_TMPDIR
rm $SLURM_TMPDIR/bart.large.tar.gz

FAIRSEQ_PATH=/network/home/jumelcle/master_thesis/fairseq
BART_PATH=$SLURM_TMPDIR/bart.large/model.pt

# 3. Define the parameters
TOTAL_NUM_UPDATES=1018            # 10 epochs through RTE for bsz 16 (default: 1018)
WARMUP_UPDATES=61                 # 6 percent of the number of updates (default: 61)
LR=1e-05                          # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2                     # Number of classes.
MAX_SENTENCES=8                   # Batch size (default: 32)
CUDA_VISIBLE_DEVICES=0            # (default: 0,1)

# 4. Launch the job
python $FAIRSEQ_PATH/train.py $SLURM_TMPDIR/RTE-bin_vanilla/ \
    --restore-file $BART_PATH \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 400 \
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
    --max-epoch 10 \
    --find-unused-parameters \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric
    #--max-tokens 4400 \ (default)
