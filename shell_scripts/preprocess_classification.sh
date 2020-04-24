#! /bin/bash
#SBATCH --job-name=prepro_class
#SBATCH --partition=main
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/preprocess_classification-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/preprocess_classification-%j.out

TASK=classification

FINETUNING_DATA_PATH=/network/tmp1/jumelcle/results/finetuning_data
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data

module load miniconda
source activate base
source activate nlp

cp -r "$FINETUNING_DATA_PATH/$TASK/$1" $SLURM_TMPDIR
cd $SLURM_TMPDIR
mv $1 RTE

/network/home/jumelcle/master_thesis/fairseq/examples/roberta/preprocess_GLUE_tasks.sh . RTE
mv RTE-bin "$1-bin"

rm -r "$PREPROCESSED_DATA_PATH/$TASK/$1-bin"
mv "$1-bin" "$PREPROCESSED_DATA_PATH/$TASK"
