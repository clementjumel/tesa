#! /bin/bash
#SBATCH --job-name=preprocess_rte
#SBATCH --partition=main
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH --error=/network/tmp1/jumelcle/errors/preprocess_rte-%j.err
#SBATCH --output=/network/tmp1/jumelcle/outputs/preprocess_rte-%j.out

module load miniconda
source activate nlp

echo "preprocessing glue_data_$1"

cp -r /network/tmp1/jumelcle/results/finetuning_raw_data/glue_data_$1 \
    $SLURM_TMPDIR

cd $SLURM_TMPDIR

/network/home/jumelcle/master_thesis/fairseq/examples/roberta/preprocess_GLUE_tasks.sh \
    glue_data_$1 RTE

rm -r /network/tmp1/jumelcle/results/finetuning_preprocessed_data/RTE-bin_$1

mv RTE-bin RTE-bin_$1
mv RTE-bin_$1 /network/tmp1/jumelcle/results/finetuning_preprocessed_data/
