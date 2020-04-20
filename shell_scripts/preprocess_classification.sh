#! /bin/bash
#SBATCH --job-name=prepro_class
#SBATCH --partition=main
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH --error=/network/tmp1/jumelcle/errors/preprocess_classification-%j.err
#SBATCH --output=/network/tmp1/jumelcle/outputs/preprocess_classification-%j.out

module load miniconda
source activate nlp

cp -r /network/tmp1/jumelcle/results/finetuning_data/classification/$1 $SLURM_TMPDIR
cd $SLURM_TMPDIR
mv $1 RTE

/network/home/jumelcle/master_thesis/fairseq/examples/roberta/preprocess_GLUE_tasks.sh . RTE
mv RTE-bin "$1-bin"

rm -r /network/tmp1/jumelcle/results/preprocessed_data/classification/"$1-bin"
mv "$1-bin" /network/tmp1/jumelcle/results/preprocessed_data/classification/
