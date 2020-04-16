#!/bin/bash
#SBATCH --job-name=infer_cnndm
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=5:00:00
#SBATCH --error=/network/tmp1/jumelcle/errors/infer_cnndm-%j.err
#SBATCH --output=/network/tmp1/jumelcle/outputs/infer_cnndm-%j.out

module load miniconda
source activate nlp

cd $SLURM_TMPDIR

cp /network/tmp1/jumelcle/pretrained_models/$1.tar.gz $SLURM_TMPDIR
tar -xvf $1.tar.gz
rm $1.tar.gz
cp /network/tmp1/jumelcle/results/models/$2/checkpoints/$3 $1

cp -r /network/tmp1/jumelcle/results/modeling_task $SLURM_TMPDIR

python /network/home/jumelcle/master_thesis/run_models.py \
          -t context_free_same_type \
          --task_path modeling_task/ \
          -m generator_bart \
          --bart \
          --trained_path "$1/" \
          --checkpoint_file $3

rm /network/tmp1/jumelcle/results/models/$2/train.hypo
rm /network/tmp1/jumelcle/results/models/$2/valid.hypo
cp train.hypo /network/tmp1/jumelcle/results/models/$2
cp valid.hypo /network/tmp1/jumelcle/results/models/$2
