#!/bin/bash

echo "--- 批量 BLEU 评估开始 (动态选择最佳 Loss) ---"
echo "正在使用 CUDA 设备 0 进行评估 (您可以根据需要更改 CUDA_VISIBLE_DEVICES=0)"
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

# ==========================================================
# 动态选择最佳检查点列表的函数
# 参数: 1. 检查点模式（用于分组，例如 'L6_D512_H8_pos'）
# 返回: 最小 Loss 对应的文件名
# ==========================================================
function find_best_checkpoint() {
    local pattern="$1"
    local best_loss="9999.0"
    local best_checkpoint=""
    
    # 使用 find 查找匹配特定模式的所有检查点文件
    # 注意：我们这里使用简单的 ls 结合 grep，因为 find 在不同系统上行为略有差异
    # 查找所有包含该模式的 .pt 文件
    # 例如：查找所有 L6_D512_H8_pos 模式的模型
    
    for filename in ${CHECKPOINT_DIR}/transformer_${pattern}_*.pt; do
        # 检查文件是否存在，防止 pattern 不匹配时 Bash 返回字面量
        if [ ! -f "$filename" ]; then
            continue
        fi

        # 从文件名中提取 loss 值
        # 模式：...checkloss(1.7348).pt
        # 使用 grep -oE 提取括号中的数字
        loss_str=$(echo "$filename" | grep -oE 'checkloss[0-9]+\.[0-9]+' | sed -E 's/checkloss//')

        # 检查是否成功提取到 Loss
        if [ -z "$loss_str" ]; then
            continue
        fi

        # 使用 awk 进行浮点数比较
        # 目标：if (loss_str < best_loss) then update
        is_smaller=$(echo "$loss_str $best_loss" | awk '{ if ($1 < $2) print "yes"; else print "no"; }')

        if [ "$is_smaller" == "yes" ]; then
            best_loss="$loss_str"
            # 使用 basename 仅保留文件名
            best_checkpoint=$(basename "$filename")
        fi
    done
    
    if [ -z "$best_checkpoint" ]; then
        echo ""
    else
        echo "$best_checkpoint"
    fi
}
# ==========================================================


# 定义要进行评估的【模型配置分组】
# 脚本将自动从每组中找到 Loss 最小的模型文件
EVAL_PATTERNS=(
    # A4: 更深模型 (L12 D512 H8 pos)
    "L12_D512_H8_pos"
    # A1 (Implicit): 更浅模型 (L3 D512 H8 pos)
    "L3_D512_H8_pos"
    # A2: 减半宽度 (L6 D256 H4 pos)
    "L6_D256_H4_pos"
    # A3: 无位置编码 (L6 D512 H8 no-pos)
    "L6_D512_H8_nopos"
    # BASELINE: 标准配置 (L6 D512 H8 pos)
    "L6_D512_H8_pos" 
    # A5: 更宽模型 (L6 D768 H12 pos)
    "L6_D768_H12_pos"
)

# 循环遍历所有配置分组并找到最佳模型进行评估
for PATTERN in "${EVAL_PATTERNS[@]}"; do
    # 动态找到当前配置模式下的最佳检查点文件名
    CHECKPOINT=$(find_best_checkpoint "$PATTERN")

    if [ -z "$CHECKPOINT" ]; then
        echo "警告: 配置模式 ${PATTERN} 下未找到任何检查点文件，跳过评估。"
        continue
    fi
    
    # 完整路径
    FULL_PATH="${CHECKPOINT_DIR}/${CHECKPOINT}"
    
    # --------------------------------------------------
    # 解析模型参数 (保持你的原有解析逻辑)
    # --------------------------------------------------
    NUM_LAYERS=$(echo "$CHECKPOINT" | sed -E 's/.*_L([0-9]+)_.*/\1/')
    D_MODEL=$(echo "$CHECKPOINT" | sed -E 's/.*_D([0-9]+)_.*/\1/')
    NUM_HEADS=$(echo "$CHECKPOINT" | sed -E 's/.*_H([0-9]+)_.*/\1/')
    POS_ENC=$(echo "$CHECKPOINT" | sed -E 's/.*_H[0-9]+_(pos|nopos)_.*/\1/')
    USE_POS_ENC=$( [ "$POS_ENC" = "pos" ] && echo "true" || echo "false" )
    
    # 提取 Loss 值用于显示
    LOSS_VAL=$(echo "$CHECKPOINT" | grep -oE 'checkloss[0-9]+\.[0-9]+' | sed -E 's/checkloss//')
    
    echo "--------------------------------------------------"
    echo "-> 正在评估: ${CHECKPOINT}"
    echo "   Loss值: ${LOSS_VAL} (该模式下最小 Loss)"
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
