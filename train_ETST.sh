export PYTHONPATH="$PYTHONPATH:/home/zchen/encyclopedia-text-style-transfer/models/"
export CUDA_VISIBLE_DEVICES=0

WIKIPATH="/home/zchen/encyclopedia-text-style-transfer/data/ETST/TFrecords/wiki/cleaned/"
BAIDUPATH="/home/zchen/encyclopedia-text-style-transfer/data/ETST/TFrecords/baidu/cleaned/"
OUT_DIR="/home/zchen/encyclopedia-text-style-transfer/CycleGAN-sentiment-transfer/experiments/ETST/5"
VOCAB_FILE="/home/zchen/encyclopedia-text-style-transfer/data/vocab"
PRETRAINED_MODEL="/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/experiments/ETST_pretrain/1/checkpoints/best.ckpt"
CKPT_PATH="/home/zchen/encyclopedia-text-style-transfer/CycleGAN-sentiment-transfer/experiments/ETST/5/checkpoints/best/ckpt-1000"

python main.py \
    --out_dir $OUT_DIR \
    --wiki_dir $WIKIPATH --baidu_dir $BAIDUPATH --vocab_file $VOCAB_FILE \
    --learning_rate 0.00001 \
    --n_tokens 1024 --static_batch \
    --epoch_size 1000 --early_stopping 20 \
    --log_interval 100 \
    --max_length 256 \
    --transformer_path $PRETRAINED_MODEL \
    --dis_iter 1 \
\
    `# validation` \
    --mode test \
    --ckpt_path $CKPT_PATH \
\
    `# debug` \
    --n_valid_steps 3 \
    # --param_set tiny \