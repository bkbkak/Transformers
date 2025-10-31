import os
import json
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import sentencepiece as spm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import DataLoader

from model import Transformer

# ================== 1. 参数解析 ==================
def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Evaluation and Generation.")
    
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the model checkpoint (.pt file) to be loaded.')
    parser.add_argument('--data_dir', type=str, default="../preprocessed", 
                        help='Directory for preprocessed data.')
    parser.add_argument('--save_dir', type=str, default="../results", 
                        help='Directory to save generated samples and results.')
    parser.add_argument('--max_len', type=int, default=100, 
                        help='Maximum sequence length for generation.')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for evaluation.')
    parser.add_argument('--debug_print_limit', type=int, default=5, # 增加调试参数
                        help='Number of samples for which to print raw IDs and tokens for debugging.')

    # --- 模型架构参数 (必须与训练时保持一致) ---
    parser.add_argument('--d_model', type=int, default=512, help='Embedding dimension.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of encoder/decoder layers.')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward network dimension.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (will be ignored during eval).')
    parser.add_argument('--use_pos_enc', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='Whether the model was trained with positional encoding.')

    args = parser.parse_args()
    return args

# ================== 2. 文本生成函数 (贪婪搜索) ==================
@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, sp, device):
    """
    使用贪婪搜索生成目标序列。
    """
    # 编码器输出
    memory = model.encode(src, src_mask)
    
    # 初始化目标序列 (只包含 <bos> 标记)
    batch_size = src.size(0)
    # 使用 sp.bos_id() 获取 <bos> ID
    ys = torch.full((batch_size, 1), sp.bos_id(), dtype=torch.long, device=device)

    for i in range(max_len - 1): # -1 是因为已经有 <bos> 了
        
        tgt_in = ys # 解码器输入始终是当前已生成的所有 ID
        
        # 目标序列掩码 (nopeak mask)
        tgt_mask = model.make_tgt_mask(tgt_in)
        out = model.decode(memory, src_mask, tgt_in, tgt_mask) 
        
        # 预测下一个词：只关注最后一个时间步的输出
        # prob: (batch_size, tgt_vocab)
        prob = model.generator(out[:, -1])
        
        # 贪婪选择概率最高的词
        _, next_word = torch.max(prob, dim=1) # next_word: (batch_size)
        
        # 拼接新的词汇
        ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        
        # 如果所有序列都生成了 <eos>，可以提前停止
        # 确保 next_word 是 Long 类型，以与 sp.eos_id() 比较
        if (next_word == sp.eos_id()).all():
            break
            
    return ys[:, 1:] # 移除初始的 <bos> 标记

# ================== 3. 主评估逻辑 ==================
def main():
    args = parse_args()
    
    # 设置保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # 1. 加载数据（只加载测试集）
    try:
        # 确保数据文件存在
        test_data_path = os.path.join(args.data_dir, "test_data.pt")
        vocab_info_path = os.path.join(args.data_dir, "vocab_info.json")

        if not os.path.exists(test_data_path) or not os.path.exists(vocab_info_path):
            print(f"❌ Cannot find preprocessed data in {args.data_dir}. Please run data.py first.")
            return

        test_data = torch.load(test_data_path)
        with open(vocab_info_path, "r") as f:
            vocab_info = json.load(f)
        
        # 创建一个简单的 TestDataset 类
        class TestDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.src = data["src"]
                self.tgt_out = data["tgt_out"] # TGT_OUT 是真实的参考翻译
            def __len__(self): return len(self.src)
            # 返回源序列和目标参考序列
            def __getitem__(self, idx): return self.src[idx], self.tgt_out[idx]
        
        test_set = TestDataset(test_data)
        # 确保 DataLoader 能够处理序列
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        
        print(f"✅ Loaded test set with {len(test_set)} samples.")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. 加载 SentencePiece 模型和词汇信息
    sp_path = os.path.join(args.data_dir, "spm_bpe_8k.model")
    if not os.path.exists(sp_path):
        print(f"❌ Cannot find SentencePiece model at {sp_path}.")
        return

    sp = spm.SentencePieceProcessor(model_file=sp_path)
    
    VOCAB_SIZE = vocab_info["vocab_size"]
    
    # === ID Information ===
    # 使用 .get() 方法尝试从 vocab_info 中获取 ID。
    # 为了鲁棒性，我们使用 spm 库自身的方法获取特殊 ID。
    PAD_ID = vocab_info.get("pad_id", sp.pad_id())
    BOS_ID = vocab_info.get("bos_id", sp.bos_id())
    EOS_ID = vocab_info.get("eos_id", sp.eos_id())
    UNK_ID = vocab_info.get("unk_id", sp.unk_id())

    print(f"Special IDs used (from vocab_info or default): PAD={PAD_ID}, BOS={BOS_ID}, EOS={EOS_ID}, UNK={UNK_ID}")
    # ========================================

    # 3. 初始化模型并加载权重
    print("🚀 Initializing model and loading checkpoint...")
    model = Transformer(
        src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE, 
        d_model=args.d_model, num_heads=args.num_heads, 
        num_layers=args.num_layers, d_ff=args.d_ff, 
        dropout=args.dropout, max_len=args.max_len,
        use_pos_enc=args.use_pos_enc 
    ).to(DEVICE)
    
    model.eval()
    
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=DEVICE))
        print(f"✅ Successfully loaded model from {args.checkpoint_path}")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    # 4. 评估循环：生成文本并计算 BLEU
    total_bleu_score = 0
    reference_count = 0
    generated_samples = []
    
    # 用于平滑 BLEU 分数计算
    chencherry = SmoothingFunction() 
    
    print("\n🧠 Starting Generation and BLEU Score Calculation...")
    
    # 统一文件名命名规则
    checkpoint_name = os.path.basename(args.checkpoint_path).replace(".pt", "")
    sample_filename = f"generated_samples_{checkpoint_name}.txt"
    bleu_filename = f"bleu_score_{checkpoint_name}.json"

    sample_filepath = os.path.join(args.save_dir, sample_filename)

    with open(sample_filepath, "w", encoding="utf-8") as f:
        
        for src, tgt_out in tqdm(test_loader, desc="Generating"):
            src = src.to(DEVICE)
            
            # 1. 编码器掩码
            src_mask = model.make_src_mask(src) 
            
            # 2. 生成目标序列 ID
            predicted_ids = greedy_decode(model, src, src_mask, args.max_len, sp, DEVICE)
            
            # 3. 统计和解码
            # predicted_ids 已经移除了 <bos>
            for i, (pred_id_seq, ref_id_seq) in enumerate(zip(predicted_ids.cpu().tolist(), tgt_out.cpu().tolist())):
                
                # a. ID 序列转 tokens (移除 PAD_ID)
                predicted_tokens = [sp.id_to_piece(idx) for idx in pred_id_seq if idx != PAD_ID]
                
                # b. 截断 EOS 之后的部分
                # 使用 EOS token 字符串进行查找
                eos_token_str = sp.id_to_piece(EOS_ID)
                try:
                    eos_index = predicted_tokens.index(eos_token_str)
                    predicted_tokens = predicted_tokens[:eos_index]
                except ValueError:
                    pass # 没有找到 EOS
                
                # c. 预测文本
                predicted_text = sp.decode(predicted_tokens)

                # d. 处理参考序列 (reference_tokens): 移除所有特殊 token
                special_ids = {PAD_ID, BOS_ID, EOS_ID, UNK_ID}
                reference_tokens = [sp.id_to_piece(idx) for idx in ref_id_seq if idx not in special_ids]
                reference_text = sp.decode(reference_tokens)
                
                # e. BLEU 分数计算 (使用 tokenized 序列)
                if reference_tokens:
                    # sentence_bleu 期望参考（reference）是一个列表的列表，但对于单个参考，
                    # 也可以接受一个包含 token 列表的列表，或者直接是 token 列表。
                    # 为了兼容性，我们使用 [reference_tokens]
                    bleu = sentence_bleu([reference_tokens], predicted_tokens, 
                                         smoothing_function=chencherry.method1)
                    total_bleu_score += bleu
                    reference_count += 1
                else:
                    bleu = 0.0
                
                # f. 写入样本
                sample_entry = {
                    "reference": reference_text,
                    "prediction": predicted_text,
                    "bleu": f"{bleu:.4f}"
                }
                generated_samples.append(sample_entry)

                # g. *** DEBUGGING OUTPUT (打印原始 ID) ***
                if len(generated_samples) <= args.debug_print_limit:
                    
                    # 仅移除 PAD ID 的原始生成 ID 序列
                    raw_gen_ids = [idx for idx in pred_id_seq if idx != PAD_ID]
                    raw_gen_tokens = [sp.id_to_piece(idx) for idx in raw_gen_ids]
                    
                    print(f"\n--- DEBUG SAMPLE {len(generated_samples)} ---")
                    print(f"Source ID Sequence (First 10): {src[i].cpu().tolist()[:10]}...")
                    print(f"Reference Text: {reference_text}")
                    print(f"Generated Text: {predicted_text}")
                    print(f"RAW Generated IDs (First 20, no PAD): {raw_gen_ids[:20]}...")
                    print(f"RAW Generated Tokens (First 20, no PAD): {raw_gen_tokens[:20]}...")
                    
                
                if len(generated_samples) <= 20:
                    f.write("--- Sample ---\n")
                    f.write(f"Ref: {reference_text}\n")
                    f.write(f"Gen: {predicted_text}\n")
                    f.write(f"BLEU: {bleu:.4f}\n\n")
        
        print(f"\n💾 Generated first 20 samples saved to {sample_filepath}")


    # 5. 结果总结
    avg_bleu = total_bleu_score / reference_count if reference_count > 0 else 0.0
    print(f"\n==========================================")
    print(f"  Evaluation Complete. Total Samples: {reference_count}")
    print(f"  Average BLEU Score (Smoothed): {avg_bleu:.4f}")
    print(f"==========================================")
    
    # 6. 保存最终BLEU分数（用于结果表格）
    result_data = {
        "model_name": checkpoint_name,
        "average_bleu": avg_bleu, 
        "total_samples": reference_count
    }
    bleu_filepath = os.path.join(args.save_dir, bleu_filename)
    with open(bleu_filepath, "w") as f:
        json.dump(result_data, f, indent=4)
    print(f"💾 BLEU score saved to {bleu_filepath}")


if __name__ == '__main__':
    # 确保 results 目录存在，统一管理结果
    os.makedirs("../results", exist_ok=True) 
    main()
