echo "--- 批量 BLEU 评估开始 ---"
echo "正在使用 CUDA 设备 0 进行评估 (您可以根据需要更改 CUDA_VISIBLE_DEVICES=0)"
# 使用 GPU 0 运行所有评估
export CUDA_VISIBLE_DEVICES=0

# 检查 eval.py 是否存在
if [ ! -f ../src/eval.py ]; then
    echo "错误: 找不到 eval.py 文件。请确认文件已创建。"
    exit 1
fi

# 检查 Python 依赖
if ! python3 -c "import nltk" &> /dev/null; then
    echo "警告: 缺少 NLTK 库。请先安装: pip install nltk"
fi

# 定义检查点文件夹
CHECKPOINT_DIR="../results/checkpoints"

# 定义要评估的最佳模型（基于提供的最低 Loss 文件）
BEST_CHECKPOINTS=(
    # A4: 更深模型 (L12 D512 H8 pos) - Loss: 1.7807 (Epoch 9)
    "transformer_L12_D512_H8_pos_seed42_20251030_105135_epoch009_checkloss1.7807.pt"
    
    # A1 (Implicit): 更浅模型 (L3 D512 H8 pos) - Loss: 1.8577 (Epoch 9)
    "transformer_L3_D512_H8_pos_seed42_20251030_105135_epoch009_checkloss1.8577.pt"
    
    # A2: 减半宽度 (L6 D256 H4 pos) - Loss: 1.7382 (Epoch 15)
    "transformer_L6_D256_H4_pos_seed42_20251030_105135_epoch015_checkloss1.7382.pt"
    
    # A3: 无位置编码 (L6 D512 H8 no-pos) - Loss: 2.6934 (Epoch 9)
    "transformer_L6_D512_H8_nopos_seed42_20251030_105135_epoch009_checkloss2.6934.pt"
    
    # BASELINE: 标准配置 (L6 D512 H8 pos) - Loss: 1.7348 (Epoch 8)
    # 注意: 此处我选择了一个时间戳稍早、Loss 最低的 L6 D512 H8
    "transformer_L6_D512_H8_pos_seed42_20251030_105110_epoch008_checkloss1.7348.pt"
    
    # A5: 更宽模型 (L6 D768 H12 pos) - Loss: 1.7774 (Epoch 9)
    "transformer_L6_D768_H12_pos_seed42_20251030_105135_epoch009_checkloss1.7774.pt"
)

# 循环遍历所有检查点并运行评估
for CHECKPOINT in "${BEST_CHECKPOINTS[@]}"; do
    # 完整路径
    FULL_PATH="${CHECKPOINT_DIR}/${CHECKPOINT}"
    
    # 提取模型参数 L/D/H/pos
    # L(12)
    NUM_LAYERS=$(echo "$CHECKPOINT" | sed -E 's/.*_L([0-9]+)_.*/\1/')
    # D(512)
    D_MODEL=$(echo "$CHECKPOINT" | sed -E 's/.*_D([0-9]+)_.*/\1/')
    # H(8)
    NUM_HEADS=$(echo "$CHECKPOINT" | sed -E 's/.*_H([0-9]+)_.*/\1/')
    # pos/nopos
    POS_ENC=$(echo "$CHECKPOINT" | sed -E 's/.*_H[0-9]+_(pos|nopos)_.*/\1/')
    # 转换为 Python bool
    USE_POS_ENC=$( [ "$POS_ENC" = "pos" ] && echo "true" || echo "false" )
    
    echo "--------------------------------------------------"
    echo "-> 正在评估: ${CHECKPOINT}"
    echo "   配置: L=${NUM_LAYERS}, D=${D_MODEL}, H=${NUM_HEADS}, PosEnc=${POS_ENC}"
    
    # 运行评估脚本
    python3 ../src/eval.py \
        --checkpoint_path "$FULL_PATH" \
        --num_layers "$NUM_LAYERS" \
        --d_model "$D_MODEL" \
        --num_heads "$NUM_HEADS" \
        --use_pos_enc "$USE_POS_ENC" 
    
    echo "评估完成: ${CHECKPOINT}"
    echo "--------------------------------------------------"
done

echo "--- 所有 BLEU 评估任务完成 ---"
