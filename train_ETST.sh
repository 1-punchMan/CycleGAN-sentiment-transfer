export PYTHONPATH="$PYTHONPATH:/home/zchen/encyclopedia-text-style-transfer/models/"
export CUDA_VISIBLE_DEVICES=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

WIKIPATH="/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/data/ETST/wiki/"
BAIDUPATH="/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/data/ETST/baidu/"
OUT_DIR="/home/zchen/encyclopedia-text-style-transfer/CycleGAN-sentiment-transfer/experiments/ETST/1"
VOCAB_FILE="/home/zchen/encyclopedia-text-style-transfer/data/vocab"
PRETRAINED_MODEL="/home/zchen/encyclopedia-text-style-transfer/models/official/nlp/transformer/experiments/ETST_pretrain/1/checkpoints/best.ckpt"
CKPT_PATH="/home/zchen/encyclopedia-text-style-transfer/CycleGAN-sentiment-transfer/experiments/ETST/1/checkpoints/last/ckpt-4"

python main.py \
    --out_dir $OUT_DIR \
    --wiki_dir $WIKIPATH --baidu_dir $BAIDUPATH --vocab_file $VOCAB_FILE \
    --learning_rate 0.00001 \
    --n_tokens 1024 \
    --epoch_size 6000 --early_stopping 20 \
    --log_interval 100 \
    --max_length 256 \
    --transformer_path $PRETRAINED_MODEL \
\
    `# validation
    # --mode test \
    # --ckpt_path $CKPT_PATH` \
\
    `# debug` \
    # --n_valid_steps 3 \
    # --param_set tiny \
    # --dis_iter 1 \