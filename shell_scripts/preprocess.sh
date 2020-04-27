#! /bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=main
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/preprocess-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/preprocess-%j.out

TASK_TYPE=$1
MODELING_TASK=$2

FINETUNING_DATA_PATH=/network/tmp1/jumelcle/results/finetuning_data
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data
FAIRSEQ_PATH=/network/home/jumelcle/master_thesis/fairseq

echo "Parameters:"
echo $TASK_TYPE
echo $MODELING_TASK
echo ""

module load miniconda
source activate base
source activate nlp

cp -r "$FINETUNING_DATA_PATH/$TASK_TYPE/$MODELING_TASK" $SLURM_TMPDIR

cd $SLURM_TMPDIR

if [ $TASK_TYPE == "classification" ]
then
  mv $MODELING_TASK RTE
  $FAIRSEQ_PATH/examples/roberta/preprocess_GLUE_tasks.sh . RTE
  mv RTE-bin "$MODELING_TASK-bin"

elif [ $TASK_TYPE == "generation" ]
then
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
  wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

  for SPLIT in train val
  do
    for LANG in source target
    do
      python -m examples.roberta.multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "$MODELING_TASK/$SPLIT.$LANG" \
        --outputs "$MODELING_TASK/$SPLIT.bpe.$LANG" \
        --workers 60 \
        --keep-empty;
    done
  done

  fairseq-preprocess \
      --source-lang "source" \
      --target-lang "target" \
      --trainpref "$MODELING_TASK/train.bpe" \
      --validpref "$MODELING_TASK/val.bpe" \
      --destdir "$MODELING_TASK-bin" \
      --workers 60 \
      --srcdict dict.txt \
      --tgtdict dict.txt;
fi

rm -rf "$PREPROCESSED_DATA_PATH/$TASK_TYPE/$MODELING_TASK-bin"
mkdir -p "$PREPROCESSED_DATA_PATH/$TASK_TYPE"
mv "$MODELING_TASK-bin" "$PREPROCESSED_DATA_PATH/$TASK_TYPE"

echo "Done."
