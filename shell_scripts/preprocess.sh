#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=main
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/preprocess-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/preprocess-%j.out

# Recover the scripts arguments
TASK_TYPE=$1
TASK=$2

# Parameters
TRAIN_PROPORTION=50
VALID_PROPORTION=25
TEST_PROPORTION=25
RANKING_SIZE=24
BATCH_SIZE=4
CONTEXT_FORMAT=v1
TARGETS_FORMAT=v0

# Paths
MASTER_THESIS_PATH=/network/home/jumelcle/master_thesis
FINETUNING_DATA_PATH=/network/tmp1/jumelcle/results/finetuning_data
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data

# Recover full paths/names
FULL_TASK="$TASK"_"$TRAIN_PROPORTION"-"$VALID_PROPORTION"-"$TEST_PROPORTION"_rs"$RANKING_SIZE"_bs"$BATCH_SIZE"_cf-"$CONTEXT_FORMAT"_tf-"$TARGETS_FORMAT"

# Print the parameters
echo "Parameters:"; echo $TASK_TYPE $TASK; echo
echo "Results path:"; echo "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin"; echo

# Load miniconda
module load miniconda
source activate base
source activate nlp

# Load the finetuning data
cp -r "$FINETUNING_DATA_PATH/$TASK_TYPE/$FULL_TASK" $SLURM_TMPDIR

# Move to SLURM temporary directory
cd $SLURM_TMPDIR

if [ $TASK_TYPE == "classification" ]
then
  # Rename the input
  mv $FULL_TASK RTE

  # Run the preprocessing
  $MASTER_THESIS_PATH/fairseq/examples/roberta/preprocess_GLUE_tasks.sh . RTE

  # Rename the output
  mv RTE-bin "$FULL_TASK-bin"

elif [ $TASK_TYPE == "generation" ]
then
  # Load the encoding files
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

  # Run the preprocessing
  for SPLIT in train val
  do
    for LANG in source target
    do
      python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "$FULL_TASK/$SPLIT.$LANG" \
        --outputs "$FULL_TASK/$SPLIT.bpe.$LANG" \
        --workers 60 \
        --keep-empty;
    done
  done

  fairseq-preprocess \
      --source-lang "source" \
      --target-lang "target" \
      --trainpref "$FULL_TASK/train.bpe" \
      --validpref "$FULL_TASK/val.bpe" \
      --destdir "$FULL_TASK-bin" \
      --workers 60 \
      --srcdict dict.txt \
      --tgtdict dict.txt;
fi

# Re-initialize the results folder
rm -rf "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$FULL_TASK-bin"
mkdir -p "$PREPROCESSED_DATA_PATH/$TASK_TYPE"

# Move the data to the server
mv "$FULL_TASK-bin" "$PREPROCESSED_DATA_PATH/$TASK_TYPE"

echo "Done."; echo
