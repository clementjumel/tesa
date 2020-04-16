#! /bin/bash
#SBATCH --job-name=preprocess_cnndm
#SBATCH --partition=main
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH --error=/network/tmp1/jumelcle/errors/preprocess_cnndm-%j.err
#SBATCH --output=/network/tmp1/jumelcle/outputs/preprocess_cnndm-%j.out

module load miniconda
source activate nlp

echo "preprocess cnndm_$1"

cp -r /network/tmp1/jumelcle/results/finetuning_raw_data/cnndm_$1 \
    $SLURM_TMPDIR

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
      --inputs "cnndm_$1/$SPLIT.$LANG" \
      --outputs "cnndm_$1/$SPLIT.bpe.$LANG" \
      --workers 60 \
      --keep-empty;
  done
done

fairseq-preprocess \
    --source-lang "source" \
    --target-lang "target" \
    --trainpref "cnndm_$1/train.bpe" \
    --validpref "cnndm_$1/val.bpe" \
    --destdir "cnndm-bin_$1/" \
    --workers 60 \
    --srcdict dict.txt \
    --tgtdict dict.txt;

rm -r /network/tmp1/jumelcle/results/finetuning_preprocessed_data/cnndm-bin_$1
mv cnndm-bin_$1 /network/tmp1/jumelcle/results/finetuning_preprocessed_data/
