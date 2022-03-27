export PYTHONPATH="$PYTHONPATH:/home/zchen/encyclopedia-text-style-transfer/models/"
export CUDA_VISIBLE_DEVICES=0

NEG_PATH="/home/zchen/encyclopedia-text-style-transfer/data/ChnSentiCorp/TFrecords/negative/"
POS_PATH="/home/zchen/encyclopedia-text-style-transfer/data/ChnSentiCorp/TFrecords/positive/"
OUT_DIR="/home/zchen/encyclopedia-text-style-transfer/CycleGAN-sentiment-transfer/experiments/ChnSentiCorp/8"
VOCAB_FILE="/home/zchen/encyclopedia-text-style-transfer/data/vocab"
PRETRAINED_MODEL="/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/experiments/ETST_pretrain/1/checkpoints/best.ckpt"
CKPT_PATH="/home/zchen/encyclopedia-text-style-transfer/CycleGAN-sentiment-transfer/experiments/ETST/1/checkpoints/last/ckpt-4"

python main.py \
    --out_dir $OUT_DIR \
    --wiki_dir $NEG_PATH --baidu_dir $POS_PATH --vocab_file $VOCAB_FILE \
    --learning_rate 0.00001 \
    --n_tokens 1024 --static_batch \
    --epoch_size 1000 --early_stopping 20 \
    --log_interval 100 \
    --max_length 256 \
    --transformer_path $PRETRAINED_MODEL \
    --dis_iter 1 \
\
    `# validation
    # --mode test \
    # --ckpt_path $CKPT_PATH \
    --not_computing_metrics` \
\
    `# debug` \
    # --n_valid_steps 3 \
    # --param_set tiny \