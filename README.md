# Transformer-IWSLT17: 基于 PyTorch 的 Transformer 德英翻译项目

## 简介 (Introduction)

本项目基于 **PyTorch** 实现了经典的 **Transformer** 架构（Vaswani et al., 2017），用于解决**机器翻译 ($\text{Machine Translation, MT}$)** 任务。我们使用 **IWSLT 2017 德语-英语 ($\text{De-En}$)** 翻译数据集进行训练和评估。

该项目包含了数据预处理、模型训练、验证、测试以及 BLEU 分数评估的全套流程，旨在提供一个清晰、高效、可复现的 Transformer 模型基线实现。

## ✨ 主要特性 (Features)

* **全功能 Transformer:** 实现完整的编码器-解码器架构、多头注意力机制 ($\text{MHA}$) 和位置编码 ($\text{Positional Encoding}$)。
* **子词化 ($\text{Subword}$):** 使用 $\text{SentencePiece}$ 进行 $\text{BPE}$ 子词化，有效处理开放词汇问题。
* **多环境支持:** 提供 $\text{Shell}$ 脚本 (`.sh`) 支持基线训练和消融实验。
* **模型检查点:** 训练过程中自动保存模型检查点和损失曲线。
* **BLEU 评估:** 使用标准 $\text{BLEU}$ 分数作为评价指标，进行定量性能分析。

## 🛠️ 技术栈与依赖 (Technology Stack & Dependencies)

* **语言:** Python 3.8+
* **核心框架:** PyTorch (推荐 1.10.x 及以上)
* **关键依赖:** NumPy, SentencePiece

### 安装依赖

建议创建一个虚拟环境并安装所需的库：

```bash
# 创建并激活 conda 环境 (推荐)
conda create -n transformer_mt python=3.9
conda activate transformer_mt

# 安装 PyTorch 和其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple




### 📦 数据准备 (Dataset Preparation)

本项目使用 IWSLT17 英德翻译数据集。
# 导航到 dataset 目录
cd dataset

# 解压 zip 文件到上一级的 data 目录
# 假设 unzip 命令可用，且解压后会产生 data/en-de-data 结构
data/en-de-data/
├── train.tags.en-de.en
├── train.tags.en-de.de
├── IWSLT17.TED.dev2010.en-de.en.xml
├── IWSLT17.TED.dev2010.en-de.de.xml
├── IWSLT17.TED.tst2010.en-de.en.xml
└── IWSLT17.TED.tst2010.en-de.de.xml

unzip iwslt17_en_de.zip -d ../data



# 返回项目根目录
cd ..


###  数据预处理
preprocessed/
├── train_data.pt
├── dev_data.pt
├── test_data.pt
├── spm_bpe_8k.model
├── spm_bpe_8k.vocab
└── vocab_info.json
#  运行data.py 
python src/data.py

###项目结构 (Project Structure)
Transformer/
├── data/                # 原始数据
├── preprocessed/         # 预处理结果 (tensor, vocab)
├── src/                  # 源代码
│   ├── model.py          # Transformer 模型定义
│   ├── train.py          # 训练主循环
│   ├── eval.py           # BLEU 评估
│   ├── data.py           # 数据加载与预处理
│   └── utils.py          # 工具函数
├── scripts/              # 实验脚本
│   ├── run_baseline.sh
│   ├── run_ablations.sh
│   ├── eval_baseline.sh
│   └── eval_ablations.sh
├── results/              # 模型输出与日志
│   ├── checkpoints/
│   ├── *.json
│   ├── *.txt
│   └── *.png
└── README.md



#运行基线模型
bash scripts/run_baseline.sh
#运行消融实验
bash scripts/run_ablations.sh
#运行评估函数基线模型
bash scripts/eval_baseline.sh
# 评估结果将保存在 ./results/ 目录下的 .json 和 .txt 文件中
#评价所有消融实验的模型
bash scripts/eval_ablations.sh
