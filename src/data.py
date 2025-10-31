import os
import re
import torch
import torch.nn as nn
import sentencepiece as spm
import json
from torch.utils.data import Dataset, DataLoader

# ================== 配置 ==================
DATA_DIR = "./data/en-de-data"
LANG_PAIR = ("en", "de")
VOCAB_SIZE = 8000
SAVE_DIR = "../preprocessed"
os.makedirs(SAVE_DIR, exist_ok=True)


# ================== 1. 读取文本 ==================
def load_ted_file(path):
    """读取 train.tags.* 或 IWSLT17.TED.*.xml 文件"""
    lines = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 跳过非内容标签
            if line.startswith('<') and not line.startswith('<seg'):
                continue
            # 提取 <seg id="x">...</seg>
            if '<seg' in line:
                m = re.search(r'<seg id="\d+">(.*?)</seg>', line)
                if m:
                    lines.append(m.group(1))
            elif not line.startswith('<'):
                lines.append(line)
    # print(f"📄 Loaded {len(lines)} lines from {os.path.basename(path)}") # 训练时可以注释掉，避免干扰日志
    return lines


def read_iwslt_split(split_name):
    """读取 train/dev/test 文件"""
    if split_name == "train":
        src_file = f"train.tags.de-en.{LANG_PAIR[0]}"
        tgt_file = f"train.tags.de-en.{LANG_PAIR[1]}"
    else:
        src_file = f"IWSLT17.TED.{split_name}.de-en.{LANG_PAIR[0]}.xml"
        tgt_file = f"IWSLT17.TED.{split_name}.de-en.{LANG_PAIR[1]}.xml"
    src_lines = load_ted_file(os.path.join(DATA_DIR, src_file))
    tgt_lines = load_ted_file(os.path.join(DATA_DIR, tgt_file))
    assert len(src_lines) == len(tgt_lines), f"{split_name}: 源/目标句子数不匹配!"
    return src_lines, tgt_lines


# ================== 2. SentencePiece BPE ==================
def train_and_load_spm(sp_model_prefix, train_src, train_tgt):
    if not os.path.exists(sp_model_prefix + ".model"):
        print("🚀 Training SentencePiece model...")
        temp_file = os.path.join(SAVE_DIR, "train_all.txt")
        with open(temp_file, "w", encoding="utf-8") as f:
            for line in train_src + train_tgt:
                f.write(line + "\n")
        
        # 考虑到某些环境中网络问题，这里使用try-except来处理可能存在的os.remove失败
        try:
            spm.SentencePieceTrainer.Train(
                input=temp_file,
                model_prefix=sp_model_prefix,
                vocab_size=VOCAB_SIZE,
                model_type="bpe",
                pad_id=1, unk_id=0, bos_id=2, eos_id=3,
                character_coverage=1.0
            )
            os.remove(temp_file) # 训练完后删除临时文件
        except Exception as e:
            print(f"Error during SPM training: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return None

    sp = spm.SentencePieceProcessor(model_file=sp_model_prefix + ".model")
    print(f"✅ Loaded SentencePiece model (vocab={sp.get_piece_size()})")
    return sp

# ================== 3. 编码为张量 ==================
def encode_split(src_lines, tgt_lines, sp, max_len=100):
    """把文本转成张量 (src, tgt_in, tgt_out)，自动 pad 并过滤空句"""
    src_ids, tgt_in_ids, tgt_out_ids = [], [], []

    for s, t in zip(src_lines, tgt_lines):
        if not s.strip() or not t.strip():
            continue  # 跳过空句
        
        # 确保序列长度不会超过 max_len
        src_tensor = torch.tensor(sp.encode(s, out_type=int))
        tgt_tensor = torch.tensor(sp.encode(t, out_type=int))

        # 目标添加 <sos>, <eos>
        # 长度检查：确保 SOS/EOS + 句子内容不会超过 max_len
        actual_max_len = max_len - 1 # 目标序列需要多一个 EOS/SOS 位

        # 截断和添加特殊标记
        src_tensor = src_tensor[:actual_max_len] 
        tgt_tensor = tgt_tensor[:actual_max_len] 
        
        tgt_in = torch.cat([torch.tensor([sp.bos_id()]), tgt_tensor]) # tgt_in: <bos>, w1, w2, ..., wn
        tgt_out = torch.cat([tgt_tensor, torch.tensor([sp.eos_id()])]) # tgt_out: w1, w2, ..., wn, <eos>

        # pad
        def pad(t, target_len):
            if len(t) < target_len:
                return torch.cat([t, torch.full((target_len - len(t),), sp.pad_id())])
            else:
                return t[:target_len] 

        src_ids.append(pad(src_tensor, max_len))
        # 目标序列统一填充到 max_len 长度
        tgt_in_ids.append(pad(tgt_in, max_len))
        tgt_out_ids.append(pad(tgt_out, max_len))

    # print(f"  Encoded {len(src_ids)} samples")
    return torch.stack(src_ids), torch.stack(tgt_in_ids), torch.stack(tgt_out_ids)

# ================== 4. Dataset 和 DataLoader ==================
class TranslationDataset(Dataset):
    """自定义 PyTorch Dataset 类，用于加载预处理后的张量"""
    def __init__(self, src, tgt_in, tgt_out):
        self.src = src
        self.tgt_in = tgt_in
        self.tgt_out = tgt_out
        assert len(src) == len(tgt_in) == len(tgt_out)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt_in[idx], self.tgt_out[idx]

def load_data_and_get_dataloaders(batch_size, data_dir=SAVE_DIR):
    """加载已保存的张量文件，并返回 DataLoader"""
    try:
        # 1. 加载数据张量
        train_data = torch.load(os.path.join(data_dir, "train_data.pt"))
        dev_data = torch.load(os.path.join(data_dir, "dev_data.pt"))
        
        # 2. 创建 Dataset 实例
        train_dataset = TranslationDataset(train_data['src'], train_data['tgt_in'], train_data['tgt_out'])
        dev_dataset = TranslationDataset(dev_data['src'], dev_data['tgt_in'], dev_data['tgt_out'])
        
        # 3. 创建 DataLoader 实例
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        # dev 评估时不需要打乱顺序
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        
        # 4. 加载词汇表信息
        with open(os.path.join(data_dir, "vocab_info.json"), "r") as f:
            vocab_info = json.load(f)
            
        print(f"✅ DataLoaders ready. Train size: {len(train_dataset)}, Dev size: {len(dev_dataset)}")
        return train_loader, dev_loader, vocab_info

    except FileNotFoundError as e:
        print(f"❌ 找不到数据文件，请先运行预处理部分: {e}")
        return None, None, None

# ================== 5. 主执行逻辑：预处理和保存 ==================

if __name__ == '__main__':
    # 定义要设置的环境变量名和值
    variable_name = "HF_ENDPOINT"
    variable_value = "https://hf-mirror.com"
    os.environ[variable_name] = variable_value
    
    # 1. 读取原始数据
    train_src, train_tgt = read_iwslt_split("train")
    dev_src, dev_tgt = read_iwslt_split("dev2010")
    test_src, test_tgt = read_iwslt_split("tst2010")

    print(f"\n✅ train: {len(train_src)} / {len(train_tgt)}")
    print(f"✅ dev:   {len(dev_src)} / {len(dev_tgt)}")
    print(f"✅ test:  {len(test_src)} / {len(test_tgt)}")

    # 2. 训练 SentencePiece 模型
    sp_model_prefix = os.path.join(SAVE_DIR, "spm_bpe_8k")
    sp = train_and_load_spm(sp_model_prefix, train_src, train_tgt)

    if sp:
        # 3. 编码为张量
        print("\n🔄 Encoding train/dev/test splits...")
        train_tensors = encode_split(train_src, train_tgt, sp)
        dev_tensors = encode_split(dev_src, dev_tgt, sp)
        test_tensors = encode_split(test_src, test_tgt, sp)

        # 4. 保存
        torch.save({"src": train_tensors[0], "tgt_in": train_tensors[1], "tgt_out": train_tensors[2]},
                   os.path.join(SAVE_DIR, "train_data.pt"))
        torch.save({"src": dev_tensors[0], "tgt_in": dev_tensors[1], "tgt_out": dev_tensors[2]},
                   os.path.join(SAVE_DIR, "dev_data.pt"))
        torch.save({"src": test_tensors[0], "tgt_in": test_tensors[1], "tgt_out": test_tensors[2]},
                   os.path.join(SAVE_DIR, "test_data.pt"))

        json.dump({
            "vocab_size": sp.get_piece_size(),
            "pad_id": sp.pad_id(),
            "unk_id": sp.unk_id(),
            "sos_id": sp.bos_id(),
            "eos_id": sp.eos_id(),
            "lang_pair": LANG_PAIR
        }, open(os.path.join(SAVE_DIR, "vocab_info.json"), "w"), ensure_ascii=False, indent=2)

        print("\n✅ 数据预处理完成！输出：")
        print(f"  - {SAVE_DIR}/spm_bpe_8k.model")
        print(f"  - {SAVE_DIR}/train_data.pt, dev_data.pt, test_data.pt")
        print(f"  - {SAVE_DIR}/vocab_info.json")