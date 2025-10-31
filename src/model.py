import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# 0. 位置编码
class PositionalEncoding(nn.Module):
    """
    位置编码 (PE) 层，用于注入序列中 token 的位置信息。
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够大的PE矩阵
        pe = torch.zeros(max_len, d_model)
        # 创建位置 (pos) 向量: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 创建除数 (div_term) 向量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # 应用正弦函数到偶数索引
        pe[:, 0::2] = torch.sin(position * div_term)
        # 应用余弦函数到奇数索引
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加批次维度，使其可与输入 x 相加
        pe = pe.unsqueeze(0)
        
        # 将 pe 注册为 buffer，它不是模型参数，但需要随模型一起保存/加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        # 截取预先计算好的 pe 矩阵，长度与输入序列长度一致
        # requires_grad=False 保证 PE 矩阵在训练中不会更新
        x = x + self.pe[:, :x.size(1)]
        return x

# 1. 核心子层

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads  # 每个头的维度
        self.num_heads = num_heads
        self.d_model = d_model
        
        # 线性层用于 Q, K, V 投影
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # 最后的线性层用于合并多头输出
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch_size, seq_len, d_model)
        mask: (batch_size, 1, seq_len, seq_len) 或 (batch_size, 1, 1, seq_len)
        """
        batch_size = q.size(0)

        # 1. 线性投影并拆分成多头
        # 结果形状: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        # -> (batch_size, num_heads, seq_len, d_k) 方便矩阵乘法
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数
        # attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # 3. 应用 Mask 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. softmax 得到注意力权重
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # 5. 乘以 V 得到最终输出
        # x 形状: (batch_size, num_heads, seq_len, d_k)
        x = torch.matmul(p_attn, v)

        # 6. 合并多头并进行最后的线性投影
        # 形状: (batch_size, seq_len, num_heads * d_k) = (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(x)

class PositionwiseFeedForward(nn.Module):
    """
    前馈网络 (Position-wise Feed-Forward Network)
    FFN(x) = max(0, x * W1 + b1) * W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 放大层
        self.w_1 = nn.Linear(d_model, d_ff)
        # 缩小层
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 激活函数使用 ReLU
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    残差连接和层归一化 (Residual Connection + Layer Normalization)
    x + Dropout(Sublayer(LayerNorm(x)))
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, sublayer):
        """
        x: 输入， sublayer: 要应用的子层函数
        """
        # 先进行 LayerNorm，再应用子层，然后 Dropout，最后残差连接
        return x + self.dropout(sublayer(self.norm(x)))

# 2. 编码器 (Encoder)

class EncoderLayer(nn.Module):
    """
    编码器层 (Encoder Layer): 包含一个自注意力子层和一个前馈网络子层
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 两个 SublayerConnection 分别用于两个子层
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask):
        """
        x: (batch_size, src_seq_len, d_model)
        mask: (batch_size, 1, 1, src_seq_len)
        """
        # 1. 自注意力层
        # q, k, v 都使用输入 x
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        
        # 2. 前馈网络层
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    """
    编码器 (Encoder): 堆叠 N 个 EncoderLayer
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # 使用深拷贝创建 N 个相同的 EncoderLayer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        # 最后的 LayerNorm
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, mask):
        """
        x: (batch_size, src_seq_len, d_model)
        mask: (batch_size, 1, 1, src_seq_len)
        """
        for layer in self.layers:
            x = layer(x, mask)
        # 编码器输出前再进行一次 LayerNorm
        return self.norm(x)

# 3. 解码器 (Decoder)

class DecoderLayer(nn.Module):
    """
    解码器层 (Decoder Layer): 包含自注意力、编码器-解码器注意力、前馈网络
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 解码器-编码器注意力
        self.src_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 三个 SublayerConnection
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])
        self.d_model = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x: (batch_size, tgt_seq_len, d_model) (目标序列的 embedding)
        memory: (batch_size, src_seq_len, d_model) (编码器输出)
        src_mask: (batch_size, 1, 1, src_seq_len) (源序列 padding mask)
        tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) (目标序列 padding mask + lookahead mask)
        """
        m = memory
        
        # 1. Masked 自注意力层
        # 保证当前 token 只能关注到之前的 token
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        # 2. 编码器-解码器注意力层
        # Q 来自于解码器上一个子层的输出 (x)
        # K, V 来自于编码器输出 (m)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        
        # 3. 前馈网络层
        x = self.sublayer[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    """
    解码器 (Decoder): 堆叠 N 个 DecoderLayer
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x: 目标序列的 embedding
        memory: 编码器输出
        src_mask, tgt_mask: 掩码
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# 4. 主 Transformer 模型

class Transformer(nn.Module):
    """
    完整的 Transformer 模型 (Encoder-Decoder)
    """
    def __init__(self, src_vocab, tgt_vocab, d_model, num_heads, num_layers, d_ff, dropout, max_len, use_pos_enc=True):
        super(Transformer, self).__init__()
        
        # 共享权重: source 和 target 的 token embedding 层
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        
        # 位置编码层
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.use_pos_enc = use_pos_enc # 控制位置编码是否启用

        # 编码器
        c = copy.deepcopy
        attn = MultiHeadAttention(d_model, num_heads, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, num_layers)

        # 解码器
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, num_layers)
        
        # 最终的线性层 (将 d_model 映射回 target 词汇表大小)
        self.generator = nn.Sequential(
            nn.Linear(d_model, tgt_vocab)
        )

        # 初始化参数：遵循 Attention Is All You Need 论文中的标准初始化
        for p in self.parameters():
            if p.dim() > 1:
                # 使用 Xavier Uniform 初始化
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt_in):
        """
        src: (batch_size, src_seq_len)
        tgt_in: (batch_size, tgt_seq_len)
        """
        # 1. 创建掩码
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt_in)

        # 2. 编码器
        memory = self.encode(src, src_mask) # (batch_size, src_seq_len, d_model)

        # 3. 解码器
        output = self.decode(memory, src_mask, tgt_in, tgt_mask) # (batch_size, tgt_seq_len, d_model)

        # 4. 预测层
        return self.generator(output) # (batch_size, tgt_seq_len, tgt_vocab)

    def encode(self, src, src_mask):
        # 词嵌入 (Embedding)
        x = self.src_embed(src)
        # 乘以 sqrt(d_model) 是为了让 embedding 的方差与位置编码一致，保持尺度稳定
        x = x * np.sqrt(self.src_embed.embedding_dim)
        
        # 位置编码 (Positional Encoding)
        if self.use_pos_enc:
            x = self.pos_enc(x)
        
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt_in, tgt_mask):
        # 词嵌入 (Embedding)
        x = self.tgt_embed(tgt_in)
        x = x * np.sqrt(self.tgt_embed.embedding_dim)
        
        # 位置编码 (Positional Encoding)
        if self.use_pos_enc:
            x = self.pos_enc(x)
            
        return self.decoder(x, memory, src_mask, tgt_mask)

    def make_src_mask(self, src):
        """
        创建源序列的 Padding Mask (用于 Encoder 自注意力和 Decoder 交叉注意力)
        只有非 PAD 的 token 才能被关注 (True)
        返回形状: (batch_size, 1, 1, src_seq_len)
        """
        # src: (batch_size, src_seq_len)
        return (src != 0).unsqueeze(1).unsqueeze(1) 

    def make_tgt_mask(self, tgt_in):
        """
        创建目标序列的 Mask (用于 Decoder 自注意力)
        包含 Padding Mask (行维度) 和 Lookahead Mask (列维度)
        返回形状: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        tgt_seq_len = tgt_in.size(-1)
        
        # 1. Padding Mask (行维度): (batch_size, 1, tgt_seq_len) -> 扩展到 (batch_size, 1, tgt_seq_len, 1)
        tgt_pad_mask = (tgt_in != 0).unsqueeze(1).unsqueeze(-1) # 假设 PAD_ID 为 0

        # 2. Lookahead Mask: 上三角矩阵，禁止关注未来的 token
        # (1, 1, tgt_seq_len, tgt_seq_len)
        # triu(diagonal=1) 返回的上三角部分为 True，我们希望 True 表示不屏蔽，所以取反
        tgt_sub_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt_in.device), diagonal=1).eq(0).unsqueeze(0).unsqueeze(0)
        
        # 3. 结合两个 Mask: 只有两个 mask 都为 True 的位置才能被关注
        # (batch_size, 1, tgt_seq_len, tgt_seq_len)
        # 广播规则: (batch_size, 1, tgt_seq_len, 1) & (1, 1, tgt_seq_len, tgt_seq_len) 
        # -> (batch_size, 1, tgt_seq_len, tgt_seq_len)
        return tgt_pad_mask & tgt_sub_mask

