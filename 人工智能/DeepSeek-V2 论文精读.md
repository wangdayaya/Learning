https://arxiv.org/pdf/2405.04434 2024年6月19日

https://youcans.blog.csdn.net/article/details/145493937

# 摘要

1.  DeepSeek-V2，一个强大的专家混合(MoE)语言模型，其特点是经济高效的训练和推理。
1.  它包含 236B 个参数，其中每个 token 激活 21B 个参数，并支持 128K 个 token 的上下文长度。
1.  DeepSeek-V2 采用了创新的架构，包括多头潜在注意力(MLA)和 DeepSeekMOE 。MLA 通过显着地压缩键值(KV)缓存到潜在向量来保证高效的推理，而 DeepSeekMoE 通过稀疏计算在经济成本下训练强大的模型。
1.  与 DeepSeek67B 相比，DeepSeek-V2 实现了显著更强的性能，同时节省了 42.5% 的训练成本，减少了 KV 缓存 93.3% ，并将最大生成吞吐量提高到 5.76 倍。
1.  对 DeepSeek-V2 进行了预训练，使用由 8.1T 个 token 组成的高质量和多源语料库，并进一步进行了监督微调(SFT)和强化学习(RL)，以充分释放其潜力。评估结果显示，即使只有 21B 激活参数，DeepSeek-V2 及其聊天版本在开源模型中仍然达到顶级性能，超过了 LLaMA3 70B 。
1.  模型检查点可在 https://github.com/deepseek-ai/DeepSeek-V2 找到。
1.  DeepSeek-V2 及其聊天版本具有其他 LLMs 中常见的公认局限性，包括预训练后缺乏持续的知识更新、生成非事实性信息的可能性以及产生幻觉的机会。
1.  由于我们的数据主要由中文和英文内容组成，我们的模型在其他语言方面的表现可能有限。在中文和英文以外的场景中，应谨慎使用。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4fb124d915814721bb4136b6868adc23~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=pr1ZGf1qbj1JZmq92v9LumZsqm8%3D)

# 介绍

1.  一般来说，LLM的智能随着参数数量的增加而提高，使其能够在各种任务中表现出涌现能力。然而这种提高是以更大的训练计算资源和潜在的推理吞吐量的减少为代价的。

1.  我们引入DeepSeek-V2，一个强大的开源混合专家(MoE)的语言模型，其特点是通过一个创新的 Transformer 架构的经济训练和有效的推理。它总共配备了 236B 参数，其中 21B 被激活用于每个令牌，并支持 128K 个 token 的上下文长度。

1.  我们使用我们提出的多头潜在注意力(MLA) 和 DeepSeekMoE 优化了 Transformer 框架中的注意力模块和前馈网络(FFN)。通过结合这两种技术DeepSeek-V2同时具有强大的性能(图1(a))、经济的训练成本和高效的推理吞吐量(图1(b))。

    1.  (1)在注意力机制的背景下，多头注意力(MHA)的键值(KV)缓存对LLM 的推理效率构成了重大障碍。已经探索出来地方法如：分组查询注意力(GQA)和多查询注意力(MQA)等这些方法，在试图减少KV缓存时往往会牺牲性能。为了同时实现最佳效果，我们引入了MLA，这是一种带有低秩键值联合压缩的注意力机制。实验结果表明，MLA在性能上优于MHA，同时在推理过程中显著减少了KV缓存，从而提高了推理效率。

        1.  MHA（Multi-head Attention）是标准的多头注意力机制，包含 h 个Query、Key 和 Value 矩阵。所有注意力头的 Key 和 Value 矩阵权重不共享
        1.  MQA（Multi-Query Attention ）是多查询注意力的一种变体，也是用于自回归解码的一种注意力机制。与MHA不同的，MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量。
        1.  GQA（Grouped-Query Attention ）是分组查询注意力，GQA 将查询头分成 G 组，每个组共享一个 Key 和 Value 矩阵。GQA-G 是指具有 G 组的 grouped-query attention 。GQA-1 具有单个组，因此具有单个 Key 和 Value，等效于 MQA 。若 GQA-H 具有与头数相等的组，则其等效于 MHA 。

    1.  ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ca292603d0184431bd597e91249db6a3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=8gZElVDziyhUDd7Dm34a6oGylaI%3D)

    1.  (2)对于前馈网络(FFNS)，我们遵循 DeepSeekMoE 架构，该架构采用精细粒度的专家和共享专家隔离，以实现专家专业化的更高潜力。 DeepSeekMoE架构与传统MoE架构相比具有显著优势，使我们能够以经济成本训练强大的模型。在训练过程中，我们利用专家级并行性，还设计了补充机制来控制通信开销并确保负载平衡。

1.  我们构建了一个高质量的多源预训练语料库，包含 8.1T 个 token 。与 DeepSeek67B(我们之前的版本)使用的语料库相比，这个语料库具有扩展的数据量，特别是中文数据，并且数据质量更高。我们首先在完整的预训练语料库上对 DeepSeek-V2 进行预训练。然后，我们收集了 150万个对话会话，涵盖数学、代码、写作、推理、安全等多个领域，对DeepSeek-V2 Chat(SFT)进行监督微调(SFT)。最后，我们跟随 DeepSeekMath 使用 Group Relative Policy Optimization (GRPO) 策略优化来进一步将模型与人类偏好对齐，并生成 DeepSeek-V2 Chat(RL)。

1.  我们对 DeepSeek-V2 在各种英语和中文基准上进行评估，并将其与具有代表性的开源模型进行比较。评估结果显示，即使只有 21B 亿个激活参数，DeepSeek-V2在开源模型中仍然表现出顶级性能，并成为最强大的开源多模语言模型。此外，AlignBench的评估表明，在中文中，DeepSeek-V2 Chat(RL) 的表现优于所有开源模型甚至击败了大多数闭源模型。

1.  为了促进 MLA 和 DeepSeekMoE 的进一步研究和开发，我们还为开源社区发布了 DeepSeek-V2-Lite，这是一个较小的模型，配备了 MLA 和 DeepSeekMoE 。它总共有 15.7B 亿个参数，其中每个 token 激活 2.4B 亿个参数。

  


# 架构

总的来说，DeepSeek-V2 仍然在 Transformer 架构中，其中每个 Transformer 块由一个注意模块和一个前馈网络(FFN)组成。然而对于注意力模块和 FFN ，我们设计并采用了创新的结构。我们设计了 MLA ，它利用低秩 key-value 联合压缩来消除推理时 key-value 缓存的瓶颈，从而支持高效的推理。对于 FFN，我们采用了 DeepSeekMoE 架构，这是一种高性能的 MoE 架结构，能够以经济的成本训练强大的模型。DeepSeekV2 的架构示意图如图2所示，除非特别说明，DeepSeek-V2 遵循 DeepSeek67B 的设置。

## 多头潜意识 Multi-Head Latent Attention :提高推理效率

传统 Transformer 模型通常采用多头注意(MHA) ，但在生成过程中，其成本较大的 Key-Value (KV)缓存将成为限制推理效率的瓶颈。为了减少 KV 缓存，提出了多查询注意(MQA)和分组查询注意(GQA)。它们需要更小的KV缓存，但它们的性能不匹配MHA 。

对干 DeepSeek-V2 ，我们设计了一种创新的注意力机制，称为多头潜在注意力(MLA)。MLA 配备了低秩 Key-Value联合压缩，比 MHA 具有更好的性能，但需要的 KV 缓存显著减少。

### 准备知识:标准多头注意力

我们首先介绍标准的 MHA（Multi-Head Attention）机制作为背景。设 $$d$$ 为嵌入维度，$$n_h$$ 为注意力头的数量，$$d_h$$ 为每个头的维度，且 $$h_t$$∈ $$R^d$$ 为注意力层中第 t 个 token 的注意力输入。标准 MHA 首先通过三个矩阵 $$W^Q$$、$$W^K$$、$$W^V$$ ∈ $$R^{d_hn_h*d}$$产生 $$q_t$$、$$k_t$$、$$v_t$$ ∈ $$R^{d_hn_h}$$分别通过。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5858932e32cc41b3b74e1b8027e6625e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=%2FAJNq5Osu%2FnKJLrL13Qjtt5yfuQ%3D)

  


然后， $$q_t$$、$$k_t$$、$$v_t$$ 将被切割成 $$n_h$$ 个头，用于多头注意力计算:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/91bd2ba7d6bf47e8ba5628f9cc898baf~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=Ej86G%2Fqcmm0LAYM9SKjHPtC3bBo%3D)

其中，$$q_{t,i}$$、$$k_{t,i}$$、$$v_{t,i}$$分别表示第 i 个注意力头的 query, key, value， $$𝑊^O ∈ R^{𝑑×𝑑_ℎ𝑛_ℎ} $$表示输出投影矩阵。在推理过程中，为了加速推理，所有 key 和 value 都需要被缓存，因此 MHA 需要为每个 token 缓存 $$2n_hd_hl$$ （每个token对应一个key 和一个 value 所以是 2 ，一共 $$l$$ 层，每个都是$$n_hd_h$$=$$d$$）个元素。在模型部署中，这种沉重的 KV 缓存是一个巨大的瓶颈，限制了最大批次大小和序列长度。

### 低秩 Key-Value 联合压缩

#### MLA 所有公式

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1894a94c525d4227bd20069114241fd4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=p%2BT69Axg7TKBgXvRugiN4LGpG8Q%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/11f76b4df6ab4dc58df7a90436a269c9~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=jNHrhsFv20G6fxuULA0yFPKW4WE%3D)

MLA的核心是对 Key-Value 进行低秩联合压缩，以减少KV缓存:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5d3d4ba783ad437cb8eb358a3879bdda~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=Emm0oiSd7X2HsV8F8StY%2Bmqb4Wo%3D)

其中， $$c^{KV}_t$$∈ $$R^{d_c}$$是键和值的压缩潜在向量， $$d_c(< 表示 KV 压缩维度，$$W^{DKV}$$∈ $$R^{d_c*d}$$ 是下投影矩阵， $$W^{UK}$$, $$W^{UV}$$∈ $$R^{d_hn_h*d_c}$$ 分别是键和值的上投影矩阵。在推理过程中，MLA 只需要缓存 $$c^{KV}_t$$，因此其为每个 token 缓存 KV 缓存只有 $$d_cl$$ 个元素，其中 $$l$$ 表示层数。此外，在推理过程中，由于 $$W^{UK}$$可以被吸收为 $$W^{Q}$$ ，$$W^{UV}$$可以被吸收为 $$W^{O}$$，我们甚至不需要计算 Key-Value 用于注意力。图3直观地说明了 MLA 中的 KV 联合压缩如何减少 KV缓存。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c7dc818b5ae54b1e9b97651067853719~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=XAFr0Mg2CW1AuRhhpcoJRU965tk%3D)

此外，为了减少训练中的激活内存，我们还进行了对 query 进行低秩压缩，即使它不能减少 KV 缓存:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4dbd3568667540cdb45af5d5932c435f~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=IB2aNMw5M68OtNeZzz%2FESnk1uxg%3D)

其中 $$c^Q_t$$ ∈ $$R^{d^{'}_c}$$是查询的压缩潜在向量， $$d^{'}_c(<表示查询压缩维度， $$W^{DQ}$$∈ $$R^{d^{'}_c*d}$$，$$W^{UQ}$$∈ $$R^{d_hn_h*d^{'}_c}$$分别是查询的下投影矩阵和上投影矩阵。

### 解耦旋转位置嵌入

继 DeepSeek 67B 之后，我们打算在 DeepSeek-V2 中使用旋转位置嵌入(RoPE) 。然而，RoPE 与低秩 KV 压缩不兼容。具体地说，RoPE 对键和查询都具有位置敏感性。如果我们将 RoPE 应用于键 $$k^C_t$$，则公式 10 中的 $$W^{UK}$$ 将与位置敏感的 RoPE 矩阵耦合。这样，在推理过程中 $$W^{UK}$$ 就不能再被 $$W^{Q}$$ 吸收，因为与当前生成的 token 相关的 ROPE 矩阵将位于 $$W^{Q}$$ 和 $$W^{UK}$$之间，并且矩阵乘法不遵循交换定律。因此，我们必须在推理过程中重新计算所有前缀 token 的 keys ，这将显著降低推理效率。

作为解决方案，我们提出了解耦的 RoPE 策略，该策略使用额外的多头查询 $$q^{R}_{t,i}$$ ∈ $$R^{d^R_h}$$和一个共享 key $$k^{R}_{t}$$ ∈ $$R^{d^R_h}$$来承载 RoPE，其中 $${d^R_h}$$ 表示解耦查询和键的每个头维度。借助解耦的 RoPE 策略，MLA 执行以下计算:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1e5126f84ba54ab3b67c86caec0c9bef~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=ZUQg5eEHGvOVT8oE%2FUXPu0Jd6ds%3D)

其中 $$W^{QR}$$∈ $$R^{d^R_hn_h*d^{'}_c}$$ 和 $$W^{KR}$$∈ $$R^{d^R_h*d}$$ 是分别产生解耦查询和键的矩阵；RoPE(·) 表示应用 RoPE 矩阵的操作； [·; ·] 表示级联操作。在推断过程中，解耦的 key 也应该被缓存。因此，DeepSeek-V2 需要一个包含 $$(d_c+d^R_h)l \approx \frac{9}{2}d_hl$$ 个元素的总 KV 缓存。

### Key-Value Cache 的比较

我们在表1中展示了不同注意力机制中每个令牌的 KV 缓存对比。MLA 只需要少量的 KV 缓存，仅相当于 2.25组的GQA ，但可以比 MHA 实现更强的性能。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6795ca776c4249e08dd03f5c21fd10ca~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=1UFB46WraIMiuxK5EoM3GYJv3cQ%3D)

## DeepSeekMoE:以经济成本训练强有力的模型

### 基本架构

对于 FFNs ，我们采用 DeepSeekMoE 架构 。DeepSeekMoE 有两个关键的想法:将专家细分为更细的粒度，以更高的专家专业化和更准确的知识获取；并隔离一些共享的专家，以减轻路由专家之间的知识冗余。在相同数量的激活和总专家参数下，DeepSeekMoE 可以显著优于传统的 MoE 架构，如GShard。假设 $$u_t$$ 是第 t 个 token 的 FFN 输入，我们将 FFN 输出 $$h'_t$$ 计算如下:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c8750e4459654df6a8578038836a66ad~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=LSD%2F2geFFuqCaxDkGjCldFct9aY%3D)

其中， $$N_s$$和 $$N_r$$ 分别表示共享专家和路由专家的数量； $$FFN^{(s)}_i(·)$$ 和 $$FFN^{(r)}_i(·)$$ 分别表示第 i 个共享专家和第 i 个路由专家； $$K_r$$ 表示激活的路由专家数量；$$g_{i,t}$$ 是第 i 个专家的闸门值；$$s_{i,t}$$ 是令牌-专家亲和度；$$e_{i}$$ 是第 i 个路由专家在本层的质心; Topk(·, 𝐾) 表示由第 t 个 token 和所有路由专家计算出的亲和度分数中 K 个最高分数组成的集合。

### 设备限制路由

我们设计了一种设备受限的路由机制，以限制与 MoE 相关的通信成本。当使用专家并行时，路由的专家将被分布在多个设备上。对于每个 token ，其 MoE 相关的通信频率与其目标专家覆盖的设备数量成正比。由于 DeepSeekMoE 中的精细专家分段，激活的专家数量可能很大，因此如果使用专家并行，则 MoE 相关的通信将更昂贵。

对于 DeepSeek-V2 ，除了对路由专家进行简单的 top-K 选择外，我们还确保每个 token 的目标专家将分布在最多 M 个设备上。具体来说，对于每个 token ，我们首先选择有专家亲和力得分最高的专家的 M 个设备。然后，我们在这 M 个设备上对专家进行 top-K 选择。在实践中，我们发现当 𝑀 ⩾ 3 时，设备限制的路由可以取得与无限制的 top-K 路由大致相同的好性能。

### 负载平衡的辅助损失

我们考虑了负载平衡，以自动学习路由策略。首先，不平衡的负载会增加路由崩溃的风险，阻止一些专家得到充分的训练和利用。其次，当使用专家并行时，不平衡的负载会降低计算效率。在 DeepSeek-V2 的训练过程中，我们设计了三种辅助损失，分别用于控制专家级负载平衡、设备级负载平衡和通信平衡。

专家级平衡损失。我们使用专家级平衡损失来降低路由崩溃的风险，其中 $$\alpha_1$$ 是一个称为专家级平衡因子的超参数， $$1(\cdot)$$ 表示指示器函数；T 表示序列中的 token 数。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5772d1de8a2f42848a9ceae62a1842ef~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=1egJjk%2Fd2Z%2F04ZJ7nLEKKaImgGQ%3D)

  


设备级平衡损失。除了专家级平衡损失，我们还设计了一个设备级平衡损失，以确保不同设备之间的平衡计算。在 DeepSeek-V2 的训练过程中，我们将所有路由的专家分成 D 组 $$\{\mathcal{E}_1, \mathcal{E}_2, \ldots, \mathcal{E}_D\}$$，并将每个组部署在单个设备上。设备级平衡损失的计算如下:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b7717e8637024a80b2cc1bde0d45a47a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=f04TvFcZ%2BUbtGiQU0tT3TmnkbIU%3D)

**通信平衡损失**。最后，我们引入了一个通信平衡损失，以确保每个设备的通信是平衡的。虽然 device-limited 的路由机制保证了每个设备的发送通信是有限的，但如果某个设备接收的 token 比其他设备多，实际通信效率也会受到影响。为了缓解这一问题，我们设计了如下的通信平衡损失:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6f89dac419124a5f989823acdccf2f61~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=col4rAnHr7ai049swdCKsOqOL2c%3D)

其中， $$\alpha_3$$ 是一个超参数，称为通信平衡因子。device-limited 的路由机制基于确保每个设备最多向其他设备传输MT隐藏状态的原则运行。同时，使用通信平衡损失来鼓励每个设备从其他设备接收大约 $$MT$$ 隐藏状态。通信平衡损失保证了设备之间的信息平衡交换，促进了高效通信。

### Token-Dropping 策略

虽然平衡损失旨在鼓励平衡负载，但重要的是要认识到它们不能保证严格的负载平衡。为了进一步减少不平衡负载导致的计算浪费，我们在训练过程中引入了 device-level 的 token-dropping 策略。该方法首先计算每个设备的平均计算预算，这意味着每个设备的容量因子等于 1.0 。受Riquelme等人的启发，我们在每个设备上丢弃亲和度得分最低的 token ，直到达到计算预算。此外，我们确保属于大约 10% 训练 sequences tokens 永远不会被丢弃。这样，我们可以根据效率要求灵活决定在推理过程中是否丢弃 token ，并始终确保训练和推理的一致性。

# 预训练

## 实验设置

### 数据构建

1.  在保持与 DeepSeek67B 相同的数据处理阶段的同时我们扩展了数据量并提高了数据质量。
1.  数据来源于各种渠道，包括互联网。
1.  纳入了更多的中文数据。
1.  改进基于质量的过滤算法，改进后的算法确保大量无用数据将被删除，而有价值的数据将大部分保留。
1.  此外，我们从预训练语料库中过滤出有争议的内容，以减轻特定地区文化引入的数据偏差。
1.  采用与 DeepSeek67B 相同的分词器，该分词器基于字节级别的字节对编码(BBPE)算法构建，词汇量为 10 万。我们的分词预训练语料库包含 8.1T 个 token ，其中中文 token 比英文 token 大约多 12% 。

### 超参数

模型超参数。我们将 Transformer 层的数量设置为 60 ，隐藏维度设置为 5120 。所有可学习的参数随机初始化，标准差为 0.006 。在 MLA 中，我们将注意头数 $$n_{h}$$ 设置为128，并将每个头维度 $$d_{h}$$ 设置为 128 。KV 压缩维度 $$d_{c}$$设置为 512 ，query 压缩维度 $$d'_{c}$$ 设置为 1536 。对于解耦的 query 和 key ，我们将每个头部维度 $$d^R_{h}$$ 设置为64。除第一层外，我们用 MoE 层替代所有 FFN 。每个 MoE 层由 2 个共享专家和 160 个路由专家组成，其中每个专家的中间隐藏维度为 1536 。在路由专家中，每个 token 将激活 6 个专家。此外，低秩压缩和精细专家分割将影响一层的输出规模。因此，在实践中，我们在压缩的潜在向量后使用额外的 RMSNorm 层，并在宽度瓶颈处(即压缩潜在向量和路由专家的中间隐藏状态)乘以额外的缩放因子，以确保训练的稳定性。在这种配置下，DeepSeek-V2 包含 236B 总参数，其中 21B 被激活用于每个 token 。

训练超参数。我们使用 AdamW 优化器 ，超参数设置为 $$\beta_1$$=0.9， $$\beta_2$$=0.95，weight_decay=0.1。学习率使用 warmup-and-step-decay 策略进行规划 。最初，在前 2K 的步骤中，学习速率从 0 线性增加到极大值。随后，在训练约 60% 的 token 后，学习率乘以 0.316，在训练大约 90% 的 token 之后，再乘以 0.3116 。最大学习率设置为 2.4x10^-4，梯度剪切规范设置为 1.0 。我们还使用批量大小调度策略，在训练前 225B 个标记时，批量大小逐渐从 2304 增加到 9216 ，然后在剩余的训练中保持 9216 。我们将最大序列长度设置为 4K ，并在 8.1T 个 token 上训练 DeepSeek-V2 。我们利用管道并行性在不同设备上部署模型的不同层，对于每一层，路由专家将在 8 个设备上均匀部署(D=8)。至于设备受限的路由，每个 token 最多将被发送到 3 个设备(M=3)。至于均衡损失我们将 $$\alpha_1$$设置为0.003，$$\alpha_2$$设置为0.05，$$\alpha_3$$设置 为0.02 。我们在训练中采用 token-dropping 策略进行加速，但评估时候没有丢弃任何 token 。

### 基础设施

DeepSeek-V2 是基于 HAI-LLM 框架进行训练的，这是一个由我们工程师内部开发的效率高、轻量级的训练框架。它采用了16路 zero-bubble 管道并行、8路专家并行和 ZeRO-1 数据并行。由于 DeepSeek-V2 激活的参数相对较少，并且部分操作被重新计算以节省激活内存，因此可以在不需要张量并行的情况下进行训练，从而减少通信开销。此外，为了进一步提高训练效率，我们将共享专家的运算与专家并行 all-to-all 全通信重叠。

我们还为不同专家之间的通信、路由算法和融合线性计算定制了更快的 CUDA 内核。此外，MLA 还基于改进版 FlashAttention-2 进行了优化。

我们在配备 NVIDIA H800 GPU 的集群上进行所有实验。H800 集群中的每个节点包含 8 个 GPU，使用节点内的 NVLink 和 NVSwitch 连接。在节点之间，InfiniBand 互连用于促进通信。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5e9a91c3554a4c0d92b0f6fb960ddd6c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=aM9aKDLtg%2B5dvz%2B%2BDgV8BR8SVmM%3D)

### 长语境扩展

在 DeepSeek-V2 的初始预训练后，我们使用 YaRN 将默认上下文窗口长度从 4K 扩展到 128K 。YaRN 专门应用于解耦共享密钥 $$k^R_{t}$$ 因为它负责携带承载 RoPE 。对于 YaRN ，我们将尺度 s 设置为 40 ，$$\alpha$$设置为1，$$\beta$$ 设置为 32 ，目标最大上下文长度设置为 160K 。在这些设置下，我们可以预期模型在 128K 上下文长度的响应良好。与原始 YaRN 略有不同，由于我们独特的注意机制，我们调整了长度尺度因子以调节注意力熵因子 $$\sqrt{t}$$ 被计算为 $$\sqrt{t}= 0.0707 lns + 1$$ ，旨在最小化困惑。

我们还对模型进行了 1000 步的训练，序列长度为 32K ，批次大小为 576 。虽然训练仅以 32K 的序列长度进行，但在 128K 的上下文长度上进行评估时，模型仍然表现出稳健的性能。如图4所示，“NeedleIn AHaystack”(NIAH)测试结果表明，DeepSeek-V2 在所有 128K 的上下文窗口长度上表现良好。

## 评价

### 评价基准

Deepseek-V2预先训练了双语语料库，因此我们根据一系列中英文基准对其进行评估。我们的评估基于我们集成的内部评估框架。在我们的 HAI-LLM 框架中。包含的基准分类如下，下划线的基准为中文:

-   多学科多选择数据集包括MMLU 、C-Eval 和CMMLU 。
-   语言理解和推理数据集包括Hellaswag 、PIQA 、ARC 和BigBenchHard 。
-   封闭式问答数据集包括TriviaQA 和Natu-ralQuestions 。
-   阅读理解数据集包括RACELai等人 ，DROP ，C3 和CMRC 。
-   参考的消歧数据集包括 WinoGrande Sakaguchi 等人 和CLUEWSC 。
-   语言建模数据集包括Pile 。
-   中国理解和文化数据集包括CHID 和CCPM 。
-   数学数据集包括GSM8K 、MATH 和CMath 。
-   代码数据集包括HumanEval 、MBPP 和 CRUXEval 。
-   标准化考试包括AGIEval 。请注意，AGIEval包括英文和中文子集。

我们采用基于困惑度的评估方法来评估包括 Hellaswag、PIQA、WinoGrande、RACE-Middle、RACE-High、MMLU、ARC-EaSy、ARC-Challenge.CHID、C-EVal、CMMLU、C3和CCPM在内的数据集。

并采用基于生成的评估方法来评估TriviaQA、NaturalQuestiOnS、DROP、MATH、GSM8K、HUmanEVaLMBPP、CRUXEVaLBBH、AGIEVaLCLUEWSC、CMRC和CMath。

此外，我们还对Pile-test进行语言模型评估，并使用比特每字节(BPB)作为指标，以确保不同分词器之间的模型公平比较。

### 评价结果

在表2中，我们比较了 Deepseek-V2 与几个具有代表性的开源模型，包括 DeepSeek67B (我们之前的版本)、Qwen1.5 72B 、LLaMA3 70B 和 Mixtral8x22B 。我们使用我们的内部评估框架评估所有这些模型，并确保它们具有相同的评估设置。总体而言，仅具有 21B 激活参数的 DeepSeek-V2 在几乎所有基准测试中均显著优于Deepseek 67B ，并在开源模型中达到顶级性能。 值得一提的是，某些先前的研究在预训练阶段纳入了SFT数据，而DeepSeek-V2在预训练过程中从未接触过SFT数据。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/25a8403dfc73404ba54249a77fa5bbec~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=SXsE3BJ9L7JYXVsbsuSV%2Fb%2FhQmE%3D)

### 训练与推理效率

训练成本。由于 DeepSeek-V2 为每个 token 激活的参数较少，且所需的 FLOPs 比 DeepSeek67B 少，因此理论上训练 DeepSeek-V2 将比训练 DeepSeek 67B 更经济虽然训练MoE模型会增加额外的通信开销，但通过我们的操作员和通信优化，DeepSeek-V2 的训练可以达到相对较高的模型 FLOPs 利用率(MFU)。在我们的 H800 集群实际训练期间，对于每万亿个 token 的训练，DeepSeek 67B 需要 300.6K 个GPU小时，而 DeepSeek-V2 只需要 172.8 K个 GPU小时，即稀疏的 DeepSeek-V2 与密集 的DeepSeek 67B 相比可以节省 42.5% 的训练成本。

推理效率。为了高效地为服务部署 DeepSeek-V2 ，我们首先将其参数转换为 FP8 的精度。此外，我们还对DeepSeek-V2 进行 KV 缓存量化，将 DeepSeek-V2 的 KV 缓存中的每个元素进一步压缩到平均 6 位。得益于 MLA 和这些优化，实际上部署的 DeepSeek-V2 所需的 KV缓存 比 DeepSeek 67B 少得多，降低了 93.3% ，因此可以处理更大的批量大小。我们评估了 DeepSeek-V2 的生成吞吐量，基于实际部署的 DeepSeek 67B 服务中的提示和生成长度分布。在一台具有 8 个 H800 GPU的节点上，DeepSeek-V2 实现了超过每秒 50000 个 token 的生成吞吐量，是 DeepSeek 67B 最大生成吞吐量的 5.76 倍。此外 DeepSeek-V2 的提示输入吞吐量超过每秒 100000 个令牌。

# 对齐

## 监督精细调校

基于我们之前的研究，我们精心挑选了指令微调数据集，包括150万个实例，其中120万个有用实例，30万个安全实例。与初始版本相比，我们提高了数据质量，以减轻幻觉性响应并提高写作能力。我们对 DeepSeek-V2 进行了 2 个 eopch 的微调，学习率设置为 5 x10^-6 。

## 强化学习

为了进一步释放 DeepSeek-V2 的潜力并使其与人类偏好保持一致，我们进行了强化学习(RL)来调整其偏好。

**强化学习算法**。为了节省 RL 的训练成本，我们采用 GRPO 算法，它放弃了通常与策略模型相同大小的评论家模型，而是从群体分数中估计基线。具体来说，对于每个问题 q ，GRPO 从旧策略 $$\pi_{\theta_{\text{old}}}$$中采样一组输出$$\left\{o_1, o_2, \cdots, o_G\right\}$$，然后通过最大化以下目标来优化策略模型 $$\pi_{\theta}$$:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7e41c8965086455cb6de50cc9e66a762~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=w6gxCfZx3P876ZZJmy8Hjq0NFgI%3D)

其中e和β是超参数，A 是优势，使用与每个组内的输出对应的一组奖励{r1,r2.…,rg}计算:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d8e0476c6fcd4e6aa80a8a3402827107~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=1Op750Q%2FIfyv3GAyNoteFDxwh6w%3D)

**训练策略**。在我们的初步实验中，我们发现对推理数据进行强化学习训练，如代码和数学提示，表现出与一般数据训练不同的独特特征。例如，我们的模型的数学和编码能力可以在较长的训练步骤中持续改进。因此，我们采用两阶段的强化学习训练策略，首先进行推理对齐然后进行人类偏好对齐。在第一个推理对齐阶段，我们训练一个奖励模型 $$RM_{reasoning}$$ 用于代码和数学推理任务，并通过 $$RM_{reasoning}$$ 的反馈来优化策略模型:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d4e1c9d178ed4eb2922dada42df4dcc4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=k7rcDooKwjA7BlfSPksuQ0OuYI0%3D)

在第二个人类偏好对齐阶段，我们采用一个多奖励框架，从帮助奖励模型 $$RM_{helpful}$$ 、安全奖励模型 $$RM_{safety}$$ 和基于规则的奖励模型 $$RM_{rule}$$ 中获取奖励。一个响应 $$o_{i}$$ 的最终奖励如下，其中 $$c_{1}$$、 $$c_{2}$$、 $$c_{3}$$ 是相应的系数:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b98905ec9bda44e8a67e85e8a6c99257~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=IAxAiz0Dy7QWKMSpwbCpQgL3dWE%3D)

  


为了获得在强化学习训练中起关键作用的可靠奖励模型，我们仔细收集偏好数据，并精心进行质量过滤和比例调整。我们基于编译器反馈获取代码偏好数据，基于真实标签获取数学偏好数据。在奖励模型训练中，我们使用DeepSeek-V2 Chat(SFT) 初始化奖励模型，并使用 point-wise 或者 pair-wise 损失进行训练。在我们的实验中，我们发现强化学习训练可以充分利用和激活模型的潜力，使其能够从可能的响应中选择正确和满意的答案。

**优化训练效率**。在非常大的模型上进行强化学习训练对训练框架提出了很高的要求，它需要仔细的工程优化来管理 GPU 内存和 RAM 压力，同时保持快速的训练速度为了实现这一目标，我们实施了以下工程优化。(1)首先，我们提出了一种混合引擎，分别采用不同的并行策略进行训练和推理，以实现更高的 GPU 利用率。（2)其次，我们使用大的 batch size 的 vLLM 作为我们的推理后端以加速推理速度。(3)第三，我们精心设计了一种调度策略，用于将模型卸载到 CPU 并重新加载到 GPU 上，从而在训练速度和内存消耗之间实现近乎最优的平衡。

## 评价结果

对标准基准的评估。首先，我们对 DeepSeek-V2 Chat(SFT) 和 DeepSeek-V2 Chat(RL) 在标准基准上进行评估。值得注意的是，与基础版本相比，DeepSeek-V2 Chat(SFT) 在 GSM8K、MATH 和 HumanEval 评估中表现出显著的改进。这一进步可以归因于我们的 SFT 数据，其中包括大量的数学和代码相关内容。此外，DeepSeek-V2 Chat(RL) 在数学和代码基准上进一步提高了性能。 另外，这些比较突显了DeepSeek-V2 Chat 在各种领域和语言中相对于其他语言模型极具竞争力和优势。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/9aeb53f3f66140099180a92fdbb7c419~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=6Rxm%2BUU%2BLGDb3IA9ZuZx%2FbeJvaI%3D)

对开放式对话生成的评估。表4 中的评估结果展示了 DeepSeek-V2 Chat(RL) 在 DeepSeek-V2 Chat(SFT) 基础上训练的显著性能优势。这一结果展示了我们的 RL 训练在实现改进对齐方面的有效性。DeepSeek-V2Chat(RL) 在两个基准上均表现出优于其他开源模型性能。这些结果突显了DeepSeek-V2 Chat(RL) 在生成高质量且与上下文相关的响应方面的强大性能，尤其是在基于指令的对话任务中。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e9721a7762a84fc7ab2cf3ff7d78a1fb~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=Ul2TZiYHz%2F3vqVq0doKvJCQjs3Q%3D)

此外，我们根据 AlignBench 评估了中国的开放式生成能力。如表5所示，DeepSeek-V2 Chat(RL) 比 DeepSeek-V2 Chat(SFT) 略占优势。值得注意的是，DeepSeek-V2 Chat(SFT ) 大幅领先于所有开源中文模型，显著优于排名第二的开源模型 Qwen1.5 72B Chat 。 具体来说，DeepSeek-V2 Chat(RL) 在中文语言理解方面表现出色，优于包括 GPT-4-Turbo-1106-Preview 在内的所有模型。另一方面，DeepSeek-V2-Chat(RL) 的推理能力仍然落后于大型模型，如s Erniebot-4.0 和 GPT-4s。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b2a69b7086754740b80c6b4c77415d87~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=rBUKwRStsFpir%2FwhwQrm4HY6fUI%3D)

## 讨论

SFT数据量。之前的工作认为，少于 10K 个 SFT 数据实例就足以产生令人满意的结果。然而，在我们的实验中，我们发现如果使用少于 10K 个实例，在 IFEval 基准上的性能会显著下降。一个可能的解释是，语言模型需要一定量的数据来发展特定的技能。 虽然随着模型大小的增加，所需的数据量可能会减少，但不能完全消除。我们的观察强调了为 LLM 配备所需功能所需充足数据的迫切需求。此外，SFT数据的质量也非常重要，特别是对于涉及写作或开放式问题的任务。

强化学习的对齐税。在人类的偏好对齐过程中，我们在开放式生成基准上观察到显著的性能提升，无论是 AI 还是人类评估者给出的评分都有所提高。然而，我们也注意到“对齐税”现象，即对齐过程可能对某些标准基准的性能产生负面影响。为了缓解对齐税，在强化学习阶段，我们在数据处理和改善训练策略方面做出了重大努力，最终在标准基准和开放式生成基准之间实现了可接受的权衡。

在线强化学习。在我们的偏好对齐实验中，我们发现在线方法显著优于离线方法。因此，我们投入了大量精力来实施一个在线 RL 框架，用于对 DeepSeek-V2 进行偏好对齐。

# 注意机制的消融

## MHA、GQA和MQA的消融

我们在表8中展示了 7B 密集模型在 MHA、GQA和MQA上对四个难基准的评估结果，这三个模型都训练在 1.33T 亿个标记上，除了注意力机制外，共享相同的架构此外，为了公平比较，我们通过调整层数将它们的参数数量调整到大约 7B 。从表中可以看出，MHA 在这些基准上比 GQA 和 MQA 具有显著优势。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ba1aacab63a34c32b04c97792039c630~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=4387zXVWl4jv3qPb%2BJ89hz97TJo%3D)

## MLA 与 MHA的比较

在表9中，我们分别展示了配备 MLA 和 MHA 的 MOE 模型在四个难基准上的评估结果。为了得出可靠的结论，我们在两个尺度上训练和评估模型。两个小型 MoE 模型包含约 16B 个总参数，我们使用 1.33T 个 token 进行训练。两个大型 MoE 模型包含约 250B 个总参数，我们使用 420B 个 token 进行训练。此外，两个小型 MoE 模型和两个大型 MoE 模型除了注意力机制外，共享相同的架构。从表中可以看出，MLA 的表现优于 MHA 。更重要的是，MLA所需的 KV 缓存量显著小于(小型MoE模型为14%，大型MoE模型则为4%)。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1356d1a0a3cf42c8ae3395c246f64ece~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=qqipraUvs7Wsk2cEuCD7T8hyC6U%3D)

# 关于训练前数据去偏化的讨论

在预训练数据准备期间，我们识别并过滤掉有争议的内容，如受地区文化影响的值，以避免我们的模型在这些有争议的话题上表现出不必要的偏见。因此，我们发现DeepSeek-V2在紧密关联特定地区文化的测试集上的表现略差。例如，在MMLU上进行评估时，尽管DeepSeek-V2在大多数测试集上的表现与Mixtral8x22B等竞争对手相当或更好，但在主要与美国价值观相关的 Humanity-Moral 子集上仍然落后。

此外，我们对这个子集进行手动分析。三名受过良好教育的人类标注员对 MMLU Humanity-Moral 子集中的 420 个道德场景进行独立标注。然后，我们计算他们的标注和 ground-truth之间的协议。如表10所示，三名人类标注员和 ground-truth之间的协议较低。因此，我们将 DeepSeek-V2 在价值敏感测试集上的异常表现归因于我们在去除预训练语料库偏见方面的原因。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e3566ddb4e3c46a68ede3a1e2cf94ff3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=zq9KrMWuBmbBYr2v3Ke4SNgJMKw%3D)

# 关于数学和代码的额外评价

DeepSeek-V2 Chat(RL) 在数千个中文数学问题的预料库的表现优于所有中文 LLMS ，包括开源和闭源型。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/73e73a1bdf6d432bbae3a34da476be79~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=3jbAQspLFEcofEKvCzN4OTTOIoc%3D)

  


![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7aa20719a1c14603b87f5af48696bce2~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741257633&x-orig-sign=J9pTHCfXjS1filV%2B0Kn0FwTjNZU%3D)

# 评估格式

我们分别在表 12-37 中展示了我们对每个基准测试上的评估格式。