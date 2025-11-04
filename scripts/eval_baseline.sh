#!/bin/bash


echo "--- 动态选择 BASELINE 最佳模型进行评估 ---"
echo "正在使用 CUDA 设备 0 进行评估"
export CUDA_VISIBLE_DEVICES=0
CHECKPOINT_DIR="./results/checkpoints"

# 检查 eval.py 是否存在
if [ ! -f ./src/eval.py ]; then
    echo "错误: 找不到 eval.py 文件。请确认文件已创建。"
    exit 1
fi

# 检查 Python 依赖（例如 NLTK）
if ! python3 -c "import nltk" &> /dev/null; then
    echo "警告: 缺少 NLTK 库。请先安装: pip install nltk"
fi


# ==========================================================
# 核心函数：查找最小 Loss 的检查点
# ==========================================================
function find_best_checkpoint() {
    local pattern="$1"
    local best_loss="9999.0"
    local best_checkpoint=""
    
    # 查找所有匹配该模式的检查点文件
    for filename in ${CHECKPOINT_DIR}/transformer_${pattern}_*.pt; do
        if [ ! -f "$filename" ]; then
            continue
        fi

        # 从文件名中提取 loss 值
        loss_str=$(echo "$filename" | grep -oE 'checkloss[0-9]+\.[0-9]+' | sed -E 's/checkloss//')

        if [ -z "$loss_str" ]; then
            continue
        fi

        # 使用 awk 进行浮点数比较
        is_smaller=$(echo "$loss_str $best_loss" | awk '{ if ($1 < $2) print "yes"; else print "no"; }')

        if [ "$is_smaller" == "yes" ]; then
            best_loss="$loss_str"
            best_checkpoint=$(basename "$filename")
        fi
    done
    
    echo "$best_checkpoint"
}
# ==========================================================


# 定义 BASELINE 配置模式 (L6 D512 H8 pos)
BASELINE_PATTERN="L6_D512_H8_pos"

echo "正在扫描配置模式 ${BASELINE_PATTERN} 下的最佳检查点..."

# 动态找到当前配置模式下的最佳检查点文件名
CHECKPOINT=$(find_best_checkpoint "$BASELINE_PATTERN")

if [ -z "$CHECKPOINT" ]; then
    echo "错误: 配置模式 ${BASELINE_PATTERN} 下未找到任何检查点文件，无法进行评估。"
    exit 1
fi

FULL_PATH="${CHECKPOINT_DIR}/${CHECKPOINT}"

# --------------------------------------------------
# 解析模型参数 (从文件名中提取 L/D/H/pos)
# --------------------------------------------------
NUM_LAYERS=$(echo "$CHECKPOINT" | sed -E 's/.*_L([0-9]+)_.*/\1/')
D_MODEL=$(echo "$CHECKPOINT" | sed -E 's/.*_D([0-9]+)_.*/\1/')
NUM_HEADS=$(echo "$CHECKPOINT" | sed -E 's/.*_H([0-9]+)_.*/\1/')
POS_ENC=$(echo "$CHECKPOINT" | sed -E 's/.*_H[0-9]+_(pos|nopos)_.*/\1/')
USE_POS_ENC=$( [ "$POS_ENC" = "pos" ] && echo "true" || echo "false" )

# 提取 Loss 值用于显示
LOSS_VAL=$(echo "$CHECKPOINT" | grep -oE 'checkloss[0-9]+\.[0-9]+' | sed -E 's/checkloss//')

echo "--------------------------------------------------"
echo "-> 正在评估 BASELINE 最佳模型:"
echo "   检查点: ${CHECKPOINT}"
echo "   Loss值: ${LOSS_VAL} (该模式下最小 Loss)"
echo "   配置: L=${NUM_LAYERS}, D=${D_MODEL}, H=${NUM_HEADS}, PosEnc=${POS_ENC}"
echo "--------------------------------------------------"

# 运行评估脚本
python3 ./src/eval.py \
    --checkpoint_path "$FULL_PATH" \
    --num_layers "$NUM_LAYERS" \
    --d_model "$D_MODEL" \
    --num_heads "$NUM_HEADS" \
    --use_pos_enc "$USE_POS_ENC"  
    
echo "--- BASELINE 评估任务完成 ---"
