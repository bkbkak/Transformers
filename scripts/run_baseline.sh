echo  "--- 启动 BASELINE 实验 (L=6, D=512, H=8, Positional Encoding) ---"
python ./src/train.py --num_layers 6 --d_model 512 --num_heads 8 --use_pos_enc true
