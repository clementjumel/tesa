#! /bin/bash
#SBATCH --job-name=prepro_gen
#SBATCH --partition=main
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/preprocess_generation-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/preprocess_generation-%j.out

TASK=generation

FINETUNING_DATA_PATH=/network/tmp1/jumelcle/results/finetuning_data
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data

module load miniconda
source activate base
source activate nlp

cp -r "$FINETUNING_DATA_PATH/$TASK/$1" $SLURM_TMPDIR
cd $SLURM_TMPDIR

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
      --inputs "$1/$SPLIT.$LANG" \
      --outputs "$1/$SPLIT.bpe.$LANG" \
      --workers 60 \
      --keep-empty;
  done
done

fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "$1/train.bpe" \
    --validpref "$1/val.bpe" \
    --destdir "$1-bin" \
    --workers 60 \
    --srcdict dict.txt \
    --tgtdict dict.txt;

rm -r "$PREPROCESSED_DATA_PATH/$TASK/$1-bin"
mv "$1-bin" "$PREPROCESSED_DATA_PATH/$TASK"
