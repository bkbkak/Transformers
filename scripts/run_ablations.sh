#!/bin/bash

# Transformer 消融实验并行启动脚本
echo "--- 启动 A1: 减半层数 (3 层) ---"
# 使用 GPU 1: L=3, D=512, H=8
CUDA_VISIBLE_DEVICES=1 python ./src/train.py --num_layers 3 &

echo "--- 启动 A2: 减半宽度 (D=256, H=4) ---"
# 使用 GPU 2: L=6, D=256, H=4
CUDA_VISIBLE_DEVICES=2 python ./src/train.py --d_model 256 --num_heads 4 &

echo "--- 启动 A3: 无位置编码 (No Positional Encoding) ---"
# 使用 GPU 3: L=6, D=512, H=8, UsePosEnc=False
CUDA_VISIBLE_DEVICES=3 python ./src/train.py --use_pos_enc false &

echo "--- 启动 A4: 更深模型 (12 层) ---"
# 使用 GPU 4: L=12, D=512, H=8
CUDA_VISIBLE_DEVICES=4 python ./src/train.py --num_layers 12 &

echo "--- 启动 A5: 更宽模型 (D=768, H=12) ---"
# 使用 GPU 5: L=6, D=768, H=12
CUDA_VISIBLE_DEVICES=5 python ./src/train.py --d_model 768 --num_heads 12 &

echo "--- 所有 5 个消融实验已并行启动 ---"
echo "您可以使用 'jobs -l' 命令查看所有后台任务的状态。"


