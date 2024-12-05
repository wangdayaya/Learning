# 前言

本文参考大神[蓝斯诺特](https://space.bilibili.com/7877324)的代码实现简单结构的 LLAMA 模型，不仅复现了 LLAMA 的结构，并且在实现模型结构的过程中，将涉及到的技术点都进行了介绍，方便大家学习。

为了简单实现，这里将层数修改为 4 ，隐层维度修改为1024 。

# LlamaRMSNorm

```python

import math,torch
class LlamaRMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1024))

    def forward(self, x):
        var = x.pow(2).mean(2, keepdim=True)
        x = x * (var + 1e-5).rsqrt()
        return self.weight * x
```

RMSNorm（Root Mean Square Normalization）的数学原理是一种归一化方法，它通过规范化输入的均方根（RMS）值来平衡特征的尺度。它既保留了归一化层的重要功能，又简化了计算流程，在性能和效率之间取得了很好的平衡，是现代大模型中的常用技术之一。RMSNorm 的核心公式如下：

$$y_i = \frac{x_i}{\text{RMS}(x) + \epsilon} \cdot g$$

-   $$x_i$$：输入的第 i 个特征值
-   $$\text{RMS}(x)$$：输入张量的均方根（Root Mean Square），对输入进行 RMS 归一化后，可以将数据的整体幅度缩放到一个固定范围（理论上 RMS 接近 1）。
-   ϵ：一个很小的常数，用于数值稳定性（防止分母为零）
-   $$\c g$$：可学习的缩放权重参数，用于调节每个特征的幅度。

计算均方根 RMS ：

$$\text{RMS}(x) = \sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2}$$

RMSNorm 优点：

-   RMSNorm 假设输入数据已经被中心化（均值为 0）。这种假设在很多模型（如深度学习中的 Transformer 等）中普遍成立。
-   RMSNorm 更简单、更高效，仅依赖均方值，而不需要计算均值和方差，降低了计算复杂度，在性能表现与 LayerNorm 接近的情况下，减少了计算开销，尤其是当输入的维度较高时（如 1024 或更多）。
-   RMSNorm 的主要目的是解决特征的尺度不平衡问题，它对特征值进行归一化，保证模型的输入和输出在训练过程中稳定，避免梯度消失或爆炸。
-   通过可学习的参数 g，它还能调整每个通道的幅度，从而增强模型的表达能力。

# RoPE

```python
# 旋转位置编码生成
@torch.no_grad()
def llama_rotary_embedding(length):
    #这一部分计算了一个频率范围，用于生成旋转的位置编码。inv_freq 是位置编码的逆频率，它决定了每个位置的周期性。计算 500000.0**inv_freq 是一种高频率的衰减方式，用于让较小的频率影响较远的位置。
    inv_freq  = torch.arange(0, 32, 2) / 32
    inv_freq = 1. / (500000.0**inv_freq )
    inv_freq = inv_freq.reshape(1, 16, 1)
    # 位置 ID 
    position_ids = torch.arange(length).reshape(1, 1, -1).float()
    # 通过矩阵乘法（matmul），将位置 ID 与逆频率（inv_freq）结合，计算出每个位置的频率
    freqs = inv_freq.matmul(position_ids).transpose(1, 2)
    # 表示每个位置的正余弦频率
    emb = torch.cat((freqs, freqs), 2)
    return emb.cos(), emb.sin()
    
# 应用旋转位置编码  
def apply_rotary_pos_emb(x, cos, sin):
    def rotate_half(x):
        # 前 16 维的特征表示左半部分 
        left = x[..., :16]
        # 后 16 维的特征表示右半部分的负数
        right = -x[..., 16:]
        return torch.cat((right, left), -1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x = (x * cos) + (rotate_half(x) * sin)
    return x
# 用法示范 
cos, sin = llama_rotary_embedding(length)
q = apply_rotary_pos_emb(q, cos ,sin)
k = apply_rotary_pos_emb(k, cos, sin)
```

这段代码实现了 [RoPE（Rotary Positional Embedding，旋转位置编码）](https://spaces.ac.cn/archives/8265)，原作者苏神介绍“这是一种配合Attention机制能达到“绝对位置编码的方式实现相对位置编码”的设计。而也正因为这种设计，它还是目前唯一一种可用于线性Attention的相对位置编码。”这是一种用于自然语言处理模型中表示位置信息的技术，特别是在处理序列数据时非常有用。RoPE 的本质目的是通过引入旋转的方式，在 Transformer 模型中将每个位置的位置信息与模型的特征相结合，如 q、k、v 这三个常见的特征，Attentio 的核心运算是内积，所以它们的内积结果带有相对位置信息，这种方式能够改善传统位置编码方法（如绝对位置编码）对序列长距离依赖的处理能力。使得每个位置的编码不仅包含其绝对位置的信息，还能够反映出该位置与其他位置的相对关系。

优点：

1.  不仅包含其绝对位置的信息，还能够反映出该位置与其他位置的相对关系
1.  尤其对于长序列数据（如长文本、长时间序列）表现尤为突出
1.  增强模型在长序列上的泛化能力

  


`llama_rotary_embedding(length)`：

这个函数用于生成旋转位置编码，主要分为以下几步：

-   **计算反频率（inv_freq）** ：使用一个固定的公式生成反频率。`inv_freq` 是通过一个固定的常数和频率序列来计算的，目的是生成不同位置的频率信息。
-   **计算位置ID（position_ids）** ：位置ID是表示每个位置在序列中的相对位置。`position_ids` 是一个从 0 到 `length-1` 的序列，表示每个位置的编号。
-   **计算旋转频率**：通过将位置ID和反频率相乘，得到旋转频率（`freqs`）。这个频率信息被用于后续的位置编码。
-   **返回余弦和正弦**：最终生成旋转位置编码的余弦（cos）和正弦（sin）值，作为旋转位置编码的表示。

  


`apply_rotary_pos_emb(x, cos, sin)`：

这个函数将旋转位置编码应用到输入张量 `x` 上。具体步骤如下：

-   **`rotate_half(x)`** ：这个操作将输入 `x` 的前半部分与后半部分交换。它是旋转位置编码的核心操作，使得每个位置的表示能够以旋转的方式进行编码。
-   **应用旋转位置编码**：通过将输入 `x` 和 `cos`、`sin` 的余弦和正弦值结合，按照旋转的方式将位置信息加到输入张量 `x` 上。具体来说，是将 `x` 的每一部分与 `cos` 和 `sin` 相乘，然后将交换过的部分与 `sin` 相乘并加上。

  


# LlamaMLP

```python
class LlamaMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(1024, 14336, bias=False)
        self.up_proj = torch.nn.Linear(1024, 14336, bias=False)
        self.down_proj = torch.nn.Linear(14336,1024, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        # 这个部分的作用是生成一个“门控”信号，控制信息的流动
        left = self.act_fn(self.gate_proj(x))
        # 这部分的作用是从原始输入中提取更多的特征
        right = self.up_proj(x)
        # 这个输出是经过门控机制调节后的特征，包含了原始输入的丰富特征信息，并且通过激活函数和逐元素相乘的操作来控制信息的流动
        return self.down_proj(left * right)
```

这个 `LlamaMLP` 类实现了一个非常经典的多层感知机结构，但采用了一些创新设计，能够处理复杂的模式和非线性关系，提升模型的性能，尤其是在对模型容量和表达能力有较高要求时，作为 Transformer 或其他类似架构的中间层。主要有以下三个部分内容：

**门控机制**：`left` 向量是通过`激活的 gate_proj(x)` 获得的，代表了输入 `x` 在高维空间中的非线性变换。`right` 向量是通过 `up_proj(x)` 获得的，代表了输入 `x` 在高维空间中的线性映射。 通过逐元素相乘的方式将这两个向量结合，形成了一种门控机制，控制了信息流的传递。

**高维映射和降维**：`gate_proj` 和 `up_proj` 都将输入映射到一个较高的维度（14336），然后 `down_proj` 将其降回到原始维度（1024）。这种设计通过先在高维空间中计算信息流，再将其压缩回较低的维度，可以帮助模型更好地捕捉非线性和复杂的关系。

**激活函数（SiLU）** ：[SiLU（Sigmoid Linear Unit）激活函数](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)，也被称为Swish函数，由谷歌大脑的研究者提出。它是一种自门控的激活函数，意味着输出的值是由输入值本身决定是否需要被放大或抑制。使用 `SiLU` 激活函数而非传统的 `ReLU` 或 `Tanh`，是为了提高非线性变换的能力。SiLU函数在一些深度学习模型中表现出了良好的性能，尤其是在自然语言处理（NLP）和计算机视觉任务中。它被认为是 ReLU 和 Leaky ReLU 的一种替代品。

SiLU函数的公式如下：

$$\text{SiLU}(x) = x \cdot \sigma(\beta x) $$

其中，$$ \sigma $$ 是 sigmoid 函数， $$\beta $$ 是一个可学习的参数（在实际应用中通常设为1以简化计算），$$x$$ 是输入值。当 $$\beta $$ 设置为 1 时，SiLU 函数简化为：

$$\text{SiLU}(x) = x \cdot \sigma(x) $$

如图所示：

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/f3a993f4a2fb433e9bede6f53b9ffc53~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732955488&x-orig-sign=z6HJE4G%2Bdjr8G9T2QjhtLKZmRYY%3D)

Sigmoid 函数的定义是：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

SiLU函数的特点包括：

1.  **平滑性**：由于SiLU函数是连续可微的，因此它在反向传播过程中可以提供平滑的梯度。
1.  **非单调性**：SiLU函数在输入值较小的时候接近于0，在输入值较大的时候接近于1，这意味着它可以根据输入值的大小自适应地调整输出。
1.  **参数效率**：SiLU函数不需要额外的参数，除了输入值本身，这使得它在参数效率上具有优势。
1.  **自门控**：SiLU函数可以根据输入值自动决定是否需要激活，这有助于减少信息在深层网络中的丢失。

# LlamaAttention

```python
# 用于确保在自注意力机制中每个 token 只能关注到它之前的 token（即“因果”关系）。该函数通常用于训练中的自回归任务，比如生成任务。
def get_causal_mask(attention_mask):
    b, length = attention_mask.shape
    min_value = -1e15
    # 是一个上三角矩阵，表示一个 token 只能关注之前的 token，防止未来的信息泄漏
    causal_mask = torch.full((length, length), min_value).triu(diagonal=1)
    causal_mask = causal_mask.reshape(1, 1, length, length).repeat(b, 1, 1, 1)
    causal_mask = causal_mask.to(attention_mask.device)
    # 将原本 attention_mask 中为 0 的位置也填充为 min_value ，表示这些位置不需要关注
    mask = attention_mask.reshape(b, 1, 1, length) == 0
    causal_mask = causal_mask.masked_fill(mask, min_value)
    return causal_mask
    
def repeat_kv(x):
    shape = list(x.shape)
    shape[1] *= 4
    return x.unsqueeze(2).repeat(1,1,4,1,1).reshape(shape)
    
class LlamaAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(1024, 1024, bias=False)
        self.k_proj = torch.nn.Linear(1024, 256, bias=False)
        self.v_proj = torch.nn.Linear(1024, 256, bias=False)
        self.o_proj = torch.nn.Linear(1024, 1024, bias=False)

    def forward(self, hidden_states, attention_mask):
        b, length, _ = hidden_states.shape
        # qkv 线性转换，并拆分多头
        q = self.q_proj(hidden_states).reshape(b, length, 32, 32).transpose(1,2)
        k = self.k_proj(hidden_states).reshape(b, length, 8, 32).transpose(1,2)
        v = self.v_proj(hidden_states).reshape(b, length, 8, 32).transpose(1,2)
        # 加入 RoPE 
        cos, sin = llama_rotary_embedding(length)
        cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)
        q = apply_rotary_pos_emb(q, cos ,sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        # 复制多份方便计算
        k = repeat_kv(k)
        v = repeat_kv(v)
        # 计算注意力
        attn = q.matmul(k.transpose(2,3)) / math.sqrt(32)
        attention_mask = get_causal_mask(attention_mask)
        attn = (attn + attention_mask).softmax(3)
        attn = attn.matmul(v)
        # 合并多头
        attn = attn.transpose(1,2).reshape(b, length, 1024)
        attn = self.o_proj(attn)
        return attn
 
```

这里就是常见的注意力机制实现，有两点不同的是：

-   q、k、v 的多头数量不同
-   对 q、k、v 融入旋转位置编码

这里需要解释的是第二点，在多头注意力机制中，`q`（查询），`k`（键），`v`（值）的 `head` 数量不同并不常见，但它是可以这样设计的，具体原因通常与模型目标或优化目标相关。以下是一些可能的原因和解释：

1. **不同的表示能力**

-   **查询（q）** ：一般来说，查询向量用于表示当前目标或输入对其他输入的“关注”程度。可能需要更多的维度来捕捉更细粒度的信息，尤其是当问题需要较高的表达能力时。因此，`q` 的维度可能需要更多的头数（32）来表示更多的“关注”模式。
-   **键（k）与值（v）** ：相比之下，键和值通常用于对输入进行编码，并传递给注意力机制。它们的头数（这里是 8）可能比查询更少，因为它们的作用是提供信息，而不是直接控制注意力分布。减少键和值的头数可能是为了降低计算量，特别是在训练大型模型时，减少冗余的表示。

2. **计算复杂度优化**

-   如果 `q`, `k`, 和 `v` 的维度数相同，通常它们会被拆分为相同数量的头。在这种设计下，头数不相同可以在不同头之间进行计算量的分配，平衡性能和计算效率。通过减少 `k` 和 `v` 的头数，可以降低整体计算的复杂度，尤其是在计算键值对时（比如在多头注意力中的点积计算）。
-   比如，`q` 的头数多（32）表示它需要更多的细粒度查询，而 `k` 和 `v` 的头数少（8）意味着它们更注重对全局信息的编码和传递。

# LlamaDecoderLayer

```python
class LlamaDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = LlamaAttention()
        self.mlp = LlamaMLP()
        self.input_layernorm = LlamaRMSNorm()
        self.post_attention_layernorm = LlamaRMSNorm()

    def forward(self, hidden_states, attention_mask):
        res = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask) + res
        res = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + res
        return hidden_states
```

该 `LlamaDecoderLayer` 的实现遵循了典型 Transformer 解码器层的设计，包含以下几个关键组件：

-   **自注意力层**：利用 `self_attn` 计算输入序列的自注意力表示，能够捕捉序列中各位置之间的依赖关系。
-   **前馈神经网络（MLP）** ：通过 `mlp` 层进一步增强模型的表达能力，处理经过自注意力计算后的隐藏状态。
-   **层归一化（RMSNorm）** ：通过 `LlamaRMSNorm` 对每一层的输入进行标准化，以提高训练稳定性，防止梯度爆炸或消失。
-   **残差连接**：每一层（自注意力层和 MLP 层）都采用了残差连接，使得输入可以直接绕过每一层传递，帮助加速收敛，并且改善深度网络的训练效果。

# LlamaModel

```python
class LlamaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(128256, 1024, None)
        self.layers = torch.nn.ModuleList([LlamaDecoderLayer() for _ in range(4)])
        self.norm = LlamaRMSNorm()

    def forward(self, input_ids, attention_mask):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

可以很明显看出来，和传统的 transformer 的嵌入层不同，这里只有一个简单的 embed_tokens ，位置嵌入已经放入了后面的注意力机制计算中和 q、k、v 进行融合了。

# LlamaForCausalLM

```python
class LlamaForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LlamaModel()
        self.lm_head = torch.nn.Linear(1024, 128256, bias=False)

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(logits)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, 128256)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
        return loss, logits
```

整个 llama 因果模型，其实就是在 LlamaModel 之上加入了一层 lm_head ，其实就是一个全连接层，用来映射到全词库。

# 比对

```python
from transformers import LlamaConfig, LlamaForCausalLM as LM
config = "{'vocab_size': 128256, 'max_position_embeddings': 8192, 'hidden_size': 4096, 'intermediate_size': 14336, 'num_hidden_layers': 32, 'num_attention_heads': 32, 'num_key_value_heads': 8, 'hidden_act': 'silu', 'initializer_range': 0.02, 'rms_norm_eps': 1e-05, 'pretraining_tp': 1, 'use_cache': True, 'rope_theta': 500000.0, 'rope_scaling': None, 'attention_bias': False, 'attention_dropout': 0.0, 'mlp_bias': False, 'return_dict': True, 'output_hidden_states': False, 'output_attentions': False, 'torchscript': False, 'torch_dtype': 'bfloat16', 'use_bfloat16': False, 'tf_legacy_loss': False, 'pruned_heads': {}, 'tie_word_embeddings': False, 'chunk_size_feed_forward': 0, 'is_encoder_decoder': False, 'is_decoder': False, 'cross_attention_hidden_size': None, 'add_cross_attention': False, 'tie_encoder_decoder': False, 'max_length': 20, 'min_length': 0, 'do_sample': False, 'early_stopping': False, 'num_beams': 1, 'num_beam_groups': 1, 'diversity_penalty': 0.0, 'temperature': 1.0, 'top_k': 50, 'top_p': 1.0, 'typical_p': 1.0, 'repetition_penalty': 1.0, 'length_penalty': 1.0, 'no_repeat_ngram_size': 0, 'encoder_no_repeat_ngram_size': 0, 'bad_words_ids': None, 'num_return_sequences': 1, 'output_scores': False, 'return_dict_in_generate': False, 'forced_bos_token_id': None, 'forced_eos_token_id': None, 'remove_invalid_values': False, 'exponential_decay_length_penalty': None, 'suppress_tokens': None, 'begin_suppress_tokens': None, 'architectures': ['LlamaForCausalLM'], 'finetuning_task': None, 'id2label': {0: 'LABEL_0', 1: 'LABEL_1'}, 'label2id': {'LABEL_0': 0, 'LABEL_1': 1}, 'tokenizer_class': None, 'prefix': None, 'bos_token_id': 128000, 'pad_token_id': None, 'eos_token_id': 128001, 'sep_token_id': None, 'decoder_start_token_id': None, 'task_specific_params': None, 'problem_type': None, '_name_or_path': '', 'transformers_version': '4.38.2', 'model_type': 'llama'}"
config = LlamaConfig.from_dict(eval(config))
config.hidden_size = 1024
config.num_hidden_layers = 4
model1 = LM(config)
model2 = LlamaForCausalLM()
model2.load_state_dict(model1.state_dict())

input = {
    'input_ids': torch.randint(100, 50000, [4, 125]),
    'attention_mask': torch.ones(4, 125).long(),
    'labels': torch.randint(100, 50000, [4, 125])
}
input['attention_mask'][:, 120:] = 0
out = model1(**input)
loss, logits = model2(**input)
print(out.loss, out.logits.shape)
print(loss, logits.shape)
out.loss == loss, (out.logits == logits).all()
```

```
tensor(11.9773, grad_fn=<NllLossBackward0>) torch.Size([4, 125, 128256])
tensor(11.9773, grad_fn=<NllLossBackward0>) torch.Size([4, 125, 128256])
```

通过加载原始 LLAMA 模型，并将两个重要的参数修改成和我们一样，其他保持不变，然后和我们实现的模型进行比对，发现计算的结果完全一样，说明我们的模型结构没有问题。

# 参考

https://github.com/lansinuote/Simple_RLHF_Llama3/blob/main/1.model.ipynb