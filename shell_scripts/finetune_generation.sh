#!/bin/bash
#SBATCH --job-name=tune-gen
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=3:00:00
#SBATCH --error=/network/tmp1/jumelcle/logs/finetune_generation-%j.err
#SBATCH --output=/network/tmp1/jumelcle/logs/finetune_generation-%j.out

module load miniconda
source activate base
source activate nlp

MODELING_TASK=$1
BART=$2
EXPERIMENT=$3

TASK=generation

MASTER_THESIS_PATH=/network/home/jumelcle/master_thesis
PRETRAINED_MODELS_PATH=/network/tmp1/jumelcle/pretrained_models
PREPROCESSED_DATA_PATH=/network/tmp1/jumelcle/results/preprocessed_data
MODELING_TASKS_PATH=/network/tmp1/jumelcle/results/modeling_task
MODELS_PATH=/network/tmp1/jumelcle/results/models
RESULTS_PATH="$MODELS_PATH/$TASK/$MODELING_TASK/$BART/$EXPERIMENT"

cp -r "$PREPROCESSED_DATA_PATH/$TASK/$MODELING_TASK-bin" $SLURM_TMPDIR
cp "$MODELING_TASKS_PATH/$MODELING_TASK.pkl" $SLURM_TMPDIR
cp "$PRETRAINED_MODELS_PATH/$BART.tar.gz" $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xvf $BART.tar.gz

rm -r $RESULTS_PATH
mkdir -p $RESULTS_PATH/tensorboard_logs

#MAX_EPOCHS=5  # Defautl: 10
#MAX_SENTENCES=1  # Default: None, works: 4
#MAX_TOKENS=1024  # Default: 2048; works: 1024, max length: 946
#UPDATE_FREQ=1
#LR=1e-05  # Default: 3e-05

#TOTAL_NUM_UPDATES=2038 # Default: 20000
#WARMUP_UPDATES=12  # Default: 500
#VALID_INTERVAL=1
#SAVE_INTERVAL=1

#CUDA_VISIBLE_DEVICES=0,1 python $MASTER_THESIS_PATH/fairseq/train.py "$MODELING_TASK-bin" \
#    --max-epoch $MAX_EPOCHS \
#    --max-sentences $MAX_SENTENCES \
#    --max-tokens $MAX_TOKENS \
#    --update-freq $UPDATE_FREQ \
#    --lr $LR \
#    --total-num-update $TOTAL_NUM_UPDATES \
#    --warmup-updates $WARMUP_UPDATES \
#    --validate-interval $VALID_INTERVAL \
#    --save-interval $SAVE_INTERVAL \
#    --restore-file $BART/model.pt \
#    --save-dir $RESULTS_PATH \
#    --tensorboard-logdir $RESULTS_PATH/tensorboard_logs \
#    --task translation \
#    --source-lang source \
#    --target-lang target \
#    --truncate-source \
#    --layernorm-embedding \
#    --share-all-embeddings \
#    --share-decoder-input-output-embed \
#    --reset-optimizer \
#    --reset-dataloader \
#    --reset-meters \
#    --required-batch-size-multiple 1 \
#    --arch bart_large \
#    --criterion label_smoothed_cross_entropy \
#    --label-smoothing 0.1 \
#    --dropout 0.1 --attention-dropout 0.1 \
#    --weight-decay 0.01 \
#    --optimizer adam \
#    --adam-betas "(0.9, 0.999)" \
#    --adam-eps 1e-08 \
#    --clip-norm 0.1 \
#    --lr-scheduler polynomial_decay \
#    --skip-invalid-size-inputs-valid-test \
#    --keep-best-checkpoints 0 \
#    --no-last-checkpoints \
#    --find-unused-parameters;
#    #--reset-lr-scheduler \

###
TOTAL_NUM_UPDATES=438
WARMUP_UPDATES=10
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=4
MAX_EPOCHS=2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $MASTER_THESIS_PATH/fairseq/train.py "$MODELING_TASK-bin" \
    --max-epoch $MAX_EPOCHS \
    --restore-file $BART/model.pt \
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
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-dir $RESULTS_PATH \
    --find-unused-parameters;
###

SAVE_INTERVAL=1

for i in $(eval echo "{$SAVE_INTERVAL..$MAX_EPOCHS..$SAVE_INTERVAL}")
do
   CHECKPOINT=checkpoint$i
   cp "$RESULTS_PATH/$CHECKPOINT.pt" $BART

   python $MASTER_THESIS_PATH/mt_models.py \
             --full_task_name $MODELING_TASK.pkl \
             --bart \
             --trained_path $BART/ \
             --checkpoint_file $CHECKPOINT.pt \
             --model generator_bart;

   rm -r $RESULTS_PATH/$CHECKPOINT
   mkdir $RESULTS_PATH/$CHECKPOINT

   mv train.source $RESULTS_PATH/$CHECKPOINT
   mv train.targets $RESULTS_PATH/$CHECKPOINT
   mv train.entities $RESULTS_PATH/$CHECKPOINT
   mv train.hypotheses $RESULTS_PATH/$CHECKPOINT

   mv valid.source $RESULTS_PATH/$CHECKPOINT
   mv valid.targets $RESULTS_PATH/$CHECKPOINT
   mv valid.entities $RESULTS_PATH/$CHECKPOINT
   mv valid.hypotheses $RESULTS_PATH/$CHECKPOINT

   echo "$CHECKPOINT evaluated"
done
