https://arxiv.org/pdf/2401.06066 2024年 1月 11日

[【DeepSeek论文精读】3. DeepSeekMoE:迈向混合专家语言模型的终极专业化-CSDN博客](https://youcans.blog.csdn.net/article/details/145485468)

# 摘要

1.  Mixture-of-Experts(MoE) 在 LLM 领域是一种有前途的架构:可以在增加模型参数时管理计算成本。
1.  传统的 MoE 架构，如 GShard，激活 N 个专家中的 top-K 个，无法确保专家的专业化能力，即每个专家获得不重叠且专注的知识。
1.  我们提出了 DeepSeekMoE 架构以实现终极的专家专业化能力。它涉及两个主要策略:

(1) 将专家细分为 $$mN$$ 个，并从其中激活 $$mK$$个，允许更灵活地组合激活的专家;

   (2)将$$Ks$$个专家隔离为共享专家，旨在捕捉通用知识并减轻路由专家中的冗余。

4.  介绍了 DeepSeekMoE 架构，用于 MoE 语言模型，目标是实现终极专家专业化。通过细粒度的专家细分和共享专家隔离，DeepSeekMoE 与现有的 MoE架构相比，实现了显著更高的专家专业化和性能。

# 引言

1.  研究表明，通过增加训练数据、模型参数、计算预算可以产生非常强大的模型。但是将模型扩展到极大规模的需要考虑计算成本，混合专家(MoE)架构已成为一个流行的解决方案。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/32321d5c68e843d3b2c1cbf920498e23~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=a6Q1T6JPTI4DKKqpW9x2fMPKMQ0%3D)

2.  MoE 可以启用参数扩展，同时将计算成本保持在一个适度的水平。最近 MoE 架构在 Transformers 中的应用已经成功将语言模型扩展到一个相当大的规模 ，并伴有出色的表现。

2.  尽管 MoE 架构有很大的潜力，但现有的 MoE 架构可能存在知识混合和知识冗余的问题，这限制了专家的专业化。传统的 MoE 架构使用 MoE 层取代了 Transformer 中的前馈网络(FFNs) 。每个 MoE 层由多名专家组成，每个专家在结构上与标准的 FFN 相同，每个 token 被分配给一名或两名专家。这种架构表现出两个潜在问题:

    1.  (1)知识混淆:现有的模型实践通常设置有限数量的专家(例如8或16)，因此分配给特定专家的 token 很可能涵盖多元化的知识。因此被指定的专家将在其参数中想汇集多种不同的知识类型，这很难被同时利用。
    1.  (2)知识冗余:分配给不同专家的 token 可能需要共同知识。因此，多个专家可能会在各自的参数中趋同于获取共享知识，从而导致专家参数中的冗余。

2.  我们引入了DeepSeekMoE 架构用于实现专家的最终专业化。 DeepSeekMoE 中的这些架构创新，提供了训练有效参数的 MoE 语言模型的机会，其中每个专家都非常专业化。我们的架构涉及两个主要策略:

    1.  (1)细粒度专家分割:在保持参数数量不变的情况下，我们通过分割FFN中间隐藏维度将专家分割成更细粒度。在保持不变的计算成本的同时，我们还激活了更细粒度的专家，以实现更灵活和适应性更强的被激活专家组合。细粒度的专家细分可以更精细地分解各种知识，并更精确地学习到不同的专家，每个专家都将保持更高的专业水平。此外，在组合已激活的专家方面的更大灵活性也有助于更准确和更有针对性的知识获取。
    1.  (2)共享专家隔离:我们将某些专家隔离，作为始终激活的共享专家，旨在捕获和整合不同上下文中的共同知识。通过将共同知识压缩到这些共享专家中，可以减轻其他路由专家之间的冗余。这可以提高参数效率，并确保每个路由专家专注于独特方面，从而保持高度专业化。

2.  我们的贡献总结如下:

    1.  架构创新。我们引入创新的MoE架构 DeepSeekMoE ，旨在实现专家中的最终专业化，它采用两种主要策略:细粒度专家分割和共享专家隔离。
    1.  实证验证。我们进行了广泛的实验，以实证验证 DeepSeekMoE 架构的有效性。实验结果验证了 DeepSeekMoE 2B 的高水平专家专业化，DeepSeekMoE 2B 与 GShard 2.9B(具有1.5倍专家参数和计算量) 的性能相当，在总参数数量相同的情况下几乎接近其密集竞品的性能，并表明 DeepSeekMoE 2B 几乎可以接近 MoE 模型的上限性能。
    1.  可扩展性。我们将 DeepSeekMoE 扩展到训练一个 16B 模型，2T tokens 语料库上进行训练，并表明仅使用约 40% 的计算量，DeepSeekMoE 16B 就能达到与 DeepSeek 7B （相同的 2T tokens 语料库上训练的密集模型） 和 LLaMA2 7B 相当的性能。我们还初步尝试将 DeepSeekMoE 扩展到145B，突显了其相对于 GShard 架构的一贯优势，并展示了与 DeepSeek 67B 相当的性能，仅使用了28.5%(甚至18.2%)的计算量。
    1.  MoE 的对齐。我们成功对 DeepSeekMoE 16B 进行了监督微调，以创建一个对齐的聊天模型，达到了与DeepSeek Chat 7B 和 LLaMA2 SFT 7B 相当的性能，展示了 DeepSeekMoE 16B 的适应性和多功能性。
    1.  公开发布。本着开放研究的精神，我们将 DeepSeekMoE 16B 的模型检查点向公众发布。值得注意的是，该模型可以在单个具有 40GB 内存的 GPU 上部署，而无需进行量化。

# 准备:Transformers的专家组合

我们首先介绍了Transformer语言模型中常用的一种通用的 MoE 体系结构。一个标准的 Transformer 语言模型是通过叠加 L 层的标准 Transformer 块构建的，其中每个块可以如下所示，为了简洁起见，我们在上述公式中省略了 layer normalization :

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/f4c69e352e5e488b8f37b99cbcd4f7f1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=bJOHFc2lZUM7gHWg6%2BF8RGNk1Ek%3D)

一种构建 MoE 语言模型的典型方法，是在 Transformer 中按照指定的间隔，用 MoE 层替换 FFN 层。一个 MoE 层由多个专家组成，每个专家的结构与标准的 FFN 层相同。然后每个 token 将被分配到一个或两个专家。若将第 l 层的 FFN 替换为 MoE 层，则其输出隐藏状态的计算表示如下：

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/13b55afc66fc4f7a988a2d49b360b837~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=ATmVg7d6xA6ZPpJ4Hy2w5tQIEhE%3D)

这种稀疏性确保了 MoE 层内的计算效率，即每个 token 将仅分配给 K 个专家计算。此外，为了简洁起见，我们省略了层归一化操作。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/47872dd578f84e4e83cd20046de01485~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=C7s%2FUVbzGVTa5vlRQio8mJXCgGA%3D)

# DeepSeekMoE架构

上面介绍了通用的 MoE 架构，我们引入了DeepSeekMoE，它专门设计用于挖掘专家专业化的潜力。如图2所示，我们的架构包括两种主要策略:细粒度专家分割和共享专家隔离。这两种策略都是为了提升专家专业化的水平。

## 细粒度专家分割

1.  在专家数量有限的情况下，分配给特定专家的 token 更有可能涵盖不同类型的知识。因此，指定的专家将在其参数中学习截然不同的知识类型，这些知识类型很难同时使用。然而，如果每个 token 都可以路由到更多的专家，那么不同的知识将有潜力在不同专家中分别进行分解和学习。在这种情况下，每个专家仍然可以保持高水平的专业化，有助于在专家之间实现更集中的知识分布。
1.  为了追求这一目标，在保持专家参数数量和计算成本一致的同时，我们将专家进行更细粒度的分割。更细粒度的专家分割使得激活的专家组合更加灵活和适应性强。具体来说，在图2(a)所示的典型MoE架构上，我们通过将 FFN 的中间隐藏维度减少到其原始大小的 m 倍来分割每个专家 FFN 。由于每个专家都变得更小，相应地，我们也将激活的专家数量增加到 m 倍，以保持相同的计算成本，如图2(b)所示。带有细粒度的专家分割后，MoE 层的输出可表示为:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3e8394d2bd374c3fb3cf3e7db84a92e3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=4xEGcsQTqKQ5lIuHvQA98ss%2Fzvk%3D)

3.  其中，专家参数总数等于 N 乘以标准 FFN 中的参数数量，mN 表示细粒度专家的总数。通过细粒度专家分割策略，激活门控的数量也会增加到 mK 。
3.  从组合的角度来看，细粒度专家细分策略显著提高了被激活专家的组合灵活性。作为一个说明性的例子，我们考虑 N=16 的情况。典型的 Top-2 路由策略可以产生 $$\binom{16}{2}$$=120 个可能的组合。相比之下，如果将每个专家分成4个较小的专家，细粒度路由策略可以产生 $$\binom{64}{8}$$=4426165368 个（约44亿）潜在组合。组合灵活性的激增增强了实现更准确和有针对性的知识获取的可能。

## 共享专家隔离

1.  采用传统的路由策略，分配给不同专家的 token 可能需要一些共同的知识或信息。因此，多个专家可能会在各自的参数中趋向于获取公共知识，从而导致专家参数中的冗余。然而，如果有共享的专家专门用于捕捉和整合在不同上下文中的共同知识，其他路由的专家之间的参数冗余将得到缓解。这种冗余的缓解将有助于构建更参数高效的模型，其中专家更加专业化。
1.  为了实现这一目标，除了细粒度的专家分割策略外，我们还进一步隔离 $$Ks$$ 个专家，作为共享专家。无论路由模块如何，每个 token 都将被确定性地分配给这些共享专家。为了保持恒定的计算成本，其他路由专家中激活的专家数量将减少 Ks ，如图2(c)所示通过集成共享专家隔离策略，完整的 DeepSeekMoE 架构中的 MoE 层可以表示为:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ec7ca250dced40a0bd1a1b477530b6e6~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=SyZi0VjOex%2F52DK0eM7rIZFHoeo%3D)

3.  最后，在DeepSeekMoE中，共享专家的数量为 $$Ks$$，路由专家的总数量为 $$mN-Ks$$ ，激活门控的数量为 $$mK-Ks$$ ，因为 $$Ks$$ 始终是激活的 。

## 负载均衡的考虑

自动学习的路由策略可能会遇到负载不平衡的问题，这表现出两个显著的缺陷。首先，存在路由崩溃的风险，即总是只选择少数专家，阻止其他专家获得足够的训练。其次，如果专家分布在多个设备上，负载不平衡会加剧计算瓶颈。

专家级的均衡损失。为了降低路由崩溃的风险，我们还采用了专家级均衡损失。均衡损失的计算如下，其中$$\alpha_1$$是一个称为专家级平衡因子的超参数，简而言之，$$N'$$等于 $$mN-Ks$$ ，$$K'$$等于 $$mK-Ks$$ 。$$1(\cdot)$$表示指示函数。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/2953a08efd824f0698806bc2e3bab07b~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=0AXiFGeE9%2BuMu9joMvX%2F9gbprIk%3D)

设备级均衡损失。除了专家级的均衡损失外，我们还引入了设备级的均衡损失。在缓解计算瓶颈时，没有必要在专家级强制执行严格的均衡约束，因为对负载平衡的过度约束将影响模型性能。相反，我们的主要目标是确保设备之间的平衡计算。如果我们将所有路由的专家分成 D 组 $$\{\mathcal{E}_1, \mathcal{E}_2, \ldots, \mathcal{E}_D\}$$，并将每个组部署在单个设备上，则设备级的平衡损失计算如下:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/84b89f74c28b449a961fc558620f0b58~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=DgA%2F28l2yNKzZUVTOQH1c01MDTk%3D)

其中，$$\alpha_2$$是一个超参数，称为设备级平衡因子。在实践中，我们设置一个较小的专家级平衡因子来缓解路由崩溃的风险，同时设置一个较大的设备级平衡因子来促进设备之间的平衡计算。

# 验证实验

## 实验设置

### 训练数据和分词

1.  训练数据由 DeepSeek-AI 创建，主要关注英语和中文，但也包括其他语言。
1.  数据来自多种来源，包括网页文本、数学材料、编码脚本、已发表的文献和各种其他文本材料。
1.  从语料库中抽取 100B 个 token 的子集来训练我们的模型。
1.  对于分词，我们使用 HuggingFace Tokenizer 工具在训练语料库的较小子集中训练字节对编码(BPE)分词器。
1.  在验证实验中，我们准备了一个词汇量为 8K 的分词器，当训练更大的模型时，词汇量将扩大。

### 基础设施

1.  我们基于 HAI-LLM 进行了实验，HAI-LLM 是一个高效且轻量级的训练框架，集成了多种并行策略，包括张量并行、ZeRO 数据并行、PipeDream 管道并行以及更具体的专家并行，通过结合数据和张量并行来实现。为了优化性能，我们开发了 CUDA 和 Triton 的GPU内核，用于门控算法和在不同专家中的线性层之间的计算融合。
1.  所有实验都是在配备 NVIDIA A100 或 H800 GPU 的集群上进行的。A100 集群中的每个节点包含 8 个 GPU ，通过 NVLink 桥接器相互连接。H800 集群中的每个节点也包含 8 个 GPU，通过节点内的 NVLink 和 NVSwitch 进行互连。对于 A100 和 H800 集群，都使用 InfiniBand 互连来促进节点之间的通信。

### 超参数

**模型设置**。在验证实验中，我们将 Transformer 的层数设置为 9 ，隐藏维度设置为 1280 。我们采用多头注意机制，共有 10 个注意头，每个注意头的维度为 128 。对于初始化，所有可学习的参数都是随机初始化的，标准差为0.006 。我们用 MoE 层替换所有 FFN ，并确保专家参数总数等于标准 FFN 的 16 倍。此外我们会将激活的专家参数(包括共享的专家参数和激活的路由专家参数)保持为标准FFN的2倍。在这种配置下，每个 MoE 模型大约有2B总参数，激活的参数数约为 0.3B 。

**训练设置**。我们使用了 AdamW 优化器 ，超参数设置为 $$\beta_1$$=0.9,$$\beta_2$$=0.95, weight decay=0.1 。学习速率是使用 warmup-and-step-decay 策略来规划的。最初在前 2K 的步骤中，学习速率从 0 线性增加到极大值。随后在80%的训练步骤中学习率乘以 0.316 ，在 90% 的训练步骤时再乘以 0.316 。验证实验的最大学习率设置为 1.08x10-3，梯度剪切规范设置为 1.0 。批量大小设置为 2K ，最大序列长度为 2K , 每个训练批次包含 4M 个标记。相应地，总训练步数设置为 25000 ，以达到 100B 个训练标记。由于训练数据的丰富性，我们在训练过程中不使用 Dropout 。考虑到模型相对较小的尺寸，所有参数，包括专家参数，都部署在单个 GPU 设备上，以避免计算不平衡。相应地，我们在训练过程中不丢弃任何标记，也不使用设备级别的平衡损失。为了防止路由崩溃，我们设置了专家级平衡因子为 0.01 。

### 评价基准

我们对涵盖各类任务的广泛基准进行评估。我们列出以下基准:

-   语言建模。对于语言建模，我们评估 Pile 测试集上的模型，评估指标是交叉熵损失。
-   语言理解和推理。对于语言理解和推理，我们考虑了HellaSwag、PIQA、ARC-challenge和ARC-easy。这些任务的评估指标是准确性。
-   阅读理解。对于阅读理解，我们使用RACE-high和RACE-middleLai等，评估指标是准确性。
-   代码生成。对于代码生成，我们在HumanEval和MBPP上评估模型。评估指标是Pass@1，表示仅一次生成尝试的通过率。
-   闭卷问答。对于闭卷问答，我们考虑Trivi-aQA和NaturalQuestions。评估指标是精确匹配率。

## 评价

**基线**。包括 DeepSeekMoE 在内，我们比较了五种模型以进行验证实验。dense 表示总参数为 0.2B 的标准密集 Transformer 语言模型。哈希层是 MoE 基于 top-1 哈希路由的体系结构，包含 2.0B 总参数和 0.2B 激活参数，并与密集基线对齐。Switch Transformer 是另一种著名的 MoE 基于顶级可学习路由的架构，其总参数和激活参数与 HashLayer 相同。GShard 采用了一种可学习的 top-2 路由策略，总参数为 2.0B ，激活参数为 0.3B ，因为与 top-1 路由方法相比，激活了一个专家。DeepSeekMoE 有 1 个共享专家和 63 个路由专家，每个专家是标准 FFN 大小的 0.25 倍。包括 DeepSeekMoE 在内，所有比较模型共享相同的训练语料库和训练超参数。所有比较的 MoE 模型具有相同的总参数数量，GShard 的激活参数与 DeepSeekMoE 相同。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6c5eb691485d420ea033261dfc710391~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=Ca5J7FNFSXta3o1ET1assyyOwFY%3D)

**结果**。我们在表1中列出了评估结果。对于所有演示模型，我们在 100B 个 token上进行训练后报告最终评估结果。从表中，我们观察到以下几点:(1)随着稀疏架构和总参数的增多，哈希层和 Switch Transformer算法在相同的激活参数数目下，性能明显强于稠密基线算法。(2)与Hash层和Switch Transformer相比，GShard具有更多的激活参数，性能略好于SwitchTransformer。(3)在总参数和激活参数数量相同的情况下，DeepSeekMoE展示了相对于GShard的压倒性优势。所有比较模型共享相同的训练语料库和训练超参数。所有比较的 MoE 模型具有相同的总参数数量，这些结果展示了我们的 DeepSeekMoE 架构在现有 MoE 架构体系中的优越性。

## DeepSeekMoE 与 MoE 模型上限紧密对齐

我们已经证明，DeepSeekMoE 的性能优于密集基线和其他 MoE 架构。为了更准确地了解 DeepSeekMoE 的性能，我们将其与具有更多总参数或激活参数的较大基线进行比较。这些比较使我们能够估计 GShard 或 Dense 基线所需的模型大小，以实现与 DeepSeekMoE 相当的性能。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/19e23b5f7ec34f48a881a765633831fc~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=ME43GVpvKqUOXBtTCfMSxaRkrXE%3D)

**与 GShardx1.5 的比较**。表2显示了 DeepSeekMoE 与一个更大的 GShard 模型(专家规模是DeepSeekMoE的1.5倍)的比较，这导致了专家参数和专家计算量的1.5倍。总的来说，我们观察到 DeepSeekMoE 与 GShardx1.5 的性能相当，突显了 DeepSeekMoE 架构固有的显著优势。

此外，我们将 DeepSeekMoE 的总参数数增加到 13.3B ，并将其与 GShardx1.2 和 GShardx1.5 进行比较，它们的总参数数分别为 15.9B 和 19.8B 。我们发现在更大的规模下，DeepSeekMoE 甚至可以明显优于 GShardx1.5 。

**与 Densex16 的比较**。表2还显示了 DeepSeekMoE 与更大密度模型之间的比较。为了公平比较，我们不使用广泛使用的注意力和 FFN 参数之间的比例(1:2)。相反，我们配置了16个共享专家，每个专家的参数数量与标准 FFN 相同。此架构模仿了一个密集模型，其参数是标准 FFN 参数的16倍。从表中我们发现，DeepSeekMoE 几乎接近Desex16 的表现，它设置了严格的的 MoE 模式的上限能力。这些结果表明，至少在约 2B 参数和 100B 训练 token 的规模下，DeepSeekMoE 的表现与 MoE 模型的理论上界密切相关。

## 消融研究

为了证明细粒度专家分割和共享专家隔离策略的有效性，我们对 DeepSeekMoE 进行了消融研究，并将结果展示在图3中。为了公平比较，我们确保所有比较模型都具有相同的总参数和激活参数数量相同。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e8747e672ab74235b90e449d5caa909f~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=L6jMa%2B2KNO6X4pNYPmVODnUPxeA%3D)

**共享专家隔离**。为了评估共享专家隔离策略的影响，我们基于 GShard 将一个专家隔离为共享专家。从图3中可以看出，与GShard相比，有意隔离共享专家在大多数基准测试中都能获得更好的性能。这些结果支持共享专家隔离策略有助于提高模型性能的假设。

**细粒度专家分割**。为了评估细粒度专家分割策略的有效性，我们通过进一步将专家细分为更细的粒度来进行更详细的比较。具体来说，我们将每个专家细分为2个或4个较小的专家，总共得到32个(1个共享+31个路由)或64个(1个共享+63个路由)专家。图3显示了专家分割粒度的连续细化对应于整体模型性能的连续提升。这些研究结果为细粒度专家细分战略的有效性提供了实证。

**共享专家和路由专家的比例**。此外，我们调查了共享专家和路由专家的最佳比例。基于总共 64 个专家的最精细粒度，并保持总专家和激活专家数量不变，我们尝试将 1、2 和 4 个专家隔离为共享专家。我们发现，共享专家和路由专家的不同比例对性能没有显著影响，1、2 和 4 个共享专家分别实现 Pile loss 为 1.808 、1.806 和 1.811 。考虑到 1:3 的比率略微提高了 Pile loss ，在扩大 DeepSeekMoE 规模时，我们将共享专家和已激活路由专家之间的比率保持为1:3。

## 专家专业化分析

在本节中，我们对 DeepSeekMoE 2B 的专家专业化进行了实证分析，本节中的 DeepSeekMoE 2B 参考了表1中报告的模型，即包含 2.0B 总参数，其中 1 个共享专家和 63 个路由专家中的 7 个被激活。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/787ff3af65c747789ab96d0805201238~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=DwVREvC6p3Su72SI7ZW1%2BwnxCJk%3D)

**DeepSeekMoE 在路由专家中表现出较低的冗余性**。为了评估路由专家之间的冗余性，我们禁用不同比例的 top 路由专家并评估 Pile loss 。具体来说，对于每个 token ，我们屏蔽一定比例的路由概率最高的专家，然后从剩余的路由专家中选择前 K 个专家。为了公平起见，我们将 DeepSeekMoE 与 GShardx1.5 进行比较，因为在没有禁用专家的情况下，它们具有相同的 Pile loss 。如图4所示，与 GShardx1.5 相比，DeepSeekMoE 对禁用顶级路由专家更为敏感这种敏感性表明 DeepSeekMoE 中的参数冗余水平较低，因为每个路由专家更加不可替代相比之下， GShardx1.5 中的专家参数冗余程度更高，因此在禁用顶级路由专家时，它可以缓冲性能下降。

**共享专家不能被路由专家所取代**。为了研究共享专家在DeepSeekMoE中的作用，我们禁用共享专家并激活一个额外的路由专家。对 Pile 的评估显示，尽管我们保持相同的计算成本，但 Pile 损失显著增加，从 1.808 上升到 2.414 。这个结果强调了共享专家的关键作用，表明共享专家捕获了与路由专家不共享的基本和必要知识，使其不可替代。

**DeepSeekMoE更准确地获取知识**。为了验证我们的说法，即结合激活专家的更高灵活性有助于更准确和有针对性的知识获取，我们研究了 DeepSeekMoE 是否可以用更少的激活专家获得必要的知识。具体来说，我们将激活的路由专家的数量从 3 个变为 7 个，并评估由此产生的 Pile loss 。如图5所示，即使只有 4 名路由专家被激活，DeepSeekMoE 实现了与 GShard 相当的 Pile loss 。这一观察支持了 DeepSeekMoE 可以更准确、更高效地获取所需知识的假设。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c648e87f197f47428366a4f1e778eeac~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=nvxgVf%2FhwsvIze73QKFcV6uVO%2BE%3D)

受这些发现的鼓舞，为了更严格地验证 DeepSeekMoE 的专家专业化和准确的知识获取，我们从零开始训练一个新的模型。该模型包括 1 个共享专家和 63 个路由专家其中只有 3 个路由专业人员被激活。如图6所示的评估结果表明，即使在相同的专家总参数和只有一半激活的专家参数的情况下，DeepSeekMoE 的表现仍然优于 GShard 。这突显了 DeepSeekMoE 利用专家参数的能力更高效，即有效参数在激活专家中的比例远高于 GShard 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8bccfb0788f8406584c41451870c40f6~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=8jZ9prNT5OgWZNtxrfPW2ol0%2Fu4%3D)

# 扩展至 DeepSeekMoE 16B

通过 DeepseekMoE 架构，我们将我们的 MoE 模型扩展到更大的规模，总参数数为 16B ，并在 2T token 上进行训练。我们的结果表明，与 LLaMA2 7B 相比，DeepSeekMoE 16B 仅在约 40% 的计算量下就实现了卓越的性能。

## 实验设置

### 训练数据和分词

1.  我们从与[训练数据和分词](https://hl8xut0wpb.feishu.cn/docx/GW6VdKyZuoi2CKx0RYRc9rVCnwg#share-SkFDd4nFso6aEKxo0vHcLuXZn5b)中描述的相同语料库中采样训练数据。
1.  与验证实验不同，我们采样了更大的数据量，有 2T 个 token ，与 LLaMA2 7B 的训练 token 数量相匹配。
1.  使用 HuggingFace Tokenizer 工具训练一个 BPE 分词器，但 DeepSeekMoE 16B 的词汇表大小设置为 100K 。

### 超参数

**模型设置**。对于 DeepSeekMoE 16B ，我们将 Transformer 层的数量设置为 28 ，隐藏维度设置为 2048 。我们采用多头注意机制，共 16 个注意头，每个注意头的维度为 128 。至于初始化，所有可学习的参数都是随机初始化的，标准差为 0.006 。我们用 MoE 层替换除了第一层之外的所有 FFN ，因为我们观察到负载平衡状态在第 1 层的收敛速度特别慢。每个 MoE 层由 2 个共享专家和 64 个路由专家组成，每个专家的大小是标准 FFN 的 0.25 倍。每个 token 将被路由到这 2 个共享专家以及 64 个路由专家中的 6 个。由于专家规模过小可能会降低计算效率，因此没有采用更精细的专家细分粒度。在超过 16B 的较大规模上，仍然可以采用更精细的粒度。在我们的配置下，DeepSeekMoE 16B 大约有 16.4B 的总参数，激活参数的数量约为 2.8B 。

**训练设置**。我们使用了 AdamW 优化器 ，超参数设置为 $$\beta_1$$=0.9,$$\beta_2$$=0.95, weight decay=0.1 。学习速率也采用 warmup-and-step-decay 策略进行规划。最初，在前 2K 的步骤中，学习速率从 0 线性增加到极大值。随后，在 80% 的训练步骤中学习率乘以 0.316 ，在 90% 的训练步骤时再乘以 0.3116 。DeepSeekMoE 16B 的最大学习率设置为 4.2x10-4，梯度剪切规范设置为 1.0 。批次大小设置为 4.5K ，最大序列长度为 4K ，每个训练批次包含 18M 个 token 。相应地总训练步数设置为106449，以达到 2T 的训练token。由于训练数据的丰富性，我门在训练过程中不使用dropout。我们利用管道并行性将模型的不同层部署在不同的设备上，对于每个层所有专家都将部署在同一设备上。因此，我们在训练过程中也不会丢弃任何标记，也不使用设备级别的平衡损失。为了防止路由崩溃，我门设置了一个相当小的专家级平衡因子为 0.001 ，因为我们发现，在我们的并行化策略下，更高的专家级平衡因子不能提高计算效率，反而会损害模型性能。

### 评价基准

除了验证实验中使用的基准外，我们还纳入了额外的基准，以进行更全面的评估。我们将验证实验中使用的基准的差别介绍如下：

-   语言模型。对于语言模型，我们还在Pile 的测试集上评估模型。由于 DeepSeekMoE16B 中使用的分词器与 LLaMA2 7B 中使用的分词器不同，为了公平比较，我们使用比特每字节(BPB)作为评估指标。
-   阅读理解。对于阅读理解，我们还考虑了DROP 。评估指标是完全匹配(EM)率。
-   数学推理。对于数学推理，我们还加入了GSM8K 和MATH ，使用 EM 作为评估度量。
-   多主题多选择。对于多主题多选项，我们额外评估 MMLU上 的模型。评估指标是准确性。
-   消歧义。为了消歧义，我们还考虑了WinoGrande ，评估指标是准确性
-   中文基准。由于 DeepSeekMoE 16B 是在双语语料库上预训练的，我们也在四个中文基准上进行评估。CLUEWSC 是一个中文歧义消解基准。CEval 和CMMLU 是两个中文多主题多选题基准，形式类似于MMLU。CHID 是一个中文成语完成基准，旨在评估对中华文化的理解。上述中文基准的评价指标是准确性或EM.
-   开源 LLM 排行榜。我们根据内部评估框架评估上述所有基准。为了公平方便地将 DeepseekMoE 16B 与开源模型进行比较，我们还在 OpenLLM 排行榜上评估了 DeepSeekMoE 16B 。0penLLM 排行榜是由 Hug-gingFace 支持的公共排行榜，它由六项任务组成:ARC 、HellaSwag 、MMLU 、TruthfulQA 、Winogrande 和GSM8K 。

## 评价

### 与 Deepseek 7B 的内部比较

我们首先对 DeepSeekMoE 16B 和 DeepSeek7B ，一个包含 6.9B 个参数的密集语言模型。为了确保公平性，两个模型都使用 2T 个标记的相同语料库进行训练。这使得我们能够准确评估我们的 MoE 架构的有效性，不受训练数据的影响。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7a675aef07094b5a842289c95418c4ff~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=I40i0crXyNC854dxD%2F8aAlJkoUY%3D)

评估结果如表3所示，得出以下观察结果:(1)总体而言，在大约40%的计算量下，DeepSeekMoE 16B 达到了与 DeepSeek7B 相当的性能。(2)DeepSeekMoE 16B 在语言建模和知识密集型任务方面表现出显著的优势，比如 Pile, HellaSwag, TriviaQA 和 NaturalQuestions 。鉴于在 MoE 模型中，FFN 参数比注意力参数重得多，这些结果与 Transformers 中的 FFN 表现出知识记忆能力的假设一致。(3)与在其他任务上的卓越表现相比，DeepSeekMoE 在处理多项选择题任务时表现出局限性。这种不足源于 DeepSeekMoE 16B 中有限的注意力参数 (DeepSeekMoE 16B 只有约 0.5B 的注意力参数，而 DeepSeek 7B 有 2.5B 的注意力参数)。我们之前对 DeepSeek7B 的研究表明，注意力容量与多项选择题任务上的表现之间存在正相关。例如，配备了多查询注意力机制的 DeepSeek7BMQA 在 MMLU-like 任务中也表现不佳。

关键的是，由于 DeepSeekMoE16 B的参数数量适中，它可以在具有 40GB 内存的 GPU 上实现单设备部署。通过适当的运算符优化，它可以达到 7B 密集模型推理速度的近 2.5 倍。

  


### 与开源模型的比较

**与 LLaMA2 7B 的内部比较**。在开源模型领域，我们主要比较 DeepSeekMoE 16B 和 LLaMA2 7B ，参数为6.7B。DeepSeekMoE 16B 和 LLaMA2 7B 都在 2T token 上进行了预训练。与 LLaMA2 7B 相比，DeepSeekMoE 拥有 245% 的总参数，但只需要 39.6% 的计算量。我们的内部基准结果见表4，由此得出以下意见。(1)在评估的基准中，仅有约 40% 的计算量，DeepSeekMoE16B 在大多数基准上的表现优于 LLaMA2 7B 。(2)DeepSeekMoE 16B 的数学推理和代码生成能力更加强大，这归因于其预训练语料库中数学和代码相关文本的丰富存在。(3)鉴于预训练语料库中存在中文文本，DeepSeekMoE 16B 在中文基准上表现出比 LLaMA2 7B 大得多的性能优势。(4)尽管训练的英文文本较少，但 DeepSeekMoE 16 在英文理解或知识密集型基准上与 LLaMA2 7B 的表现相当或更好，这展示了 DeepSeekMoE 16的卓越能力。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/a437493c3069421bb139ae46a592d32a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=HyV1VDMklMqKrxm5L%2B3H7pMBaGM%3D)

  


**对 Open LLM 排行榜的评估**。除了我们内部的评估外，我们还在OpenLLM排行榜上评估了 DeepSeekMoE 16B ，并将其与其他开源模型进行比较。除了 LLaMA2 7B 之外，我们还考虑了一组更广泛的开源模型，包括LLaMA 7B 、Falcon 7B GPT-J 6B 等，如图1所示评估结果显示，DeepSeekMoE 16B 始终比具有类似激活参数的模型有显著优势。此外它在与 LLaMA2 7B(激活参数约为 DeepSeekMoE 2.5 倍的模型)的对比中表现相当。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/2b4632d2c2e54c87a64ccee5140dd420~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=D7KQAKogOmUTLD7%2FYTNHhLBMf6M%3D)

# 对齐 DeepSeekMoE 16B

先前的研究表明，MoE 模型通常不会从微调中获得显著增益 。然而Shen等人的研究结果表明，MoE 模型确实可以从指令微调中受益。为了评估 DeepSeekMoE 16B 是否可以从微调中受益，我们对 DeepSeekMoE 16B 进行了 SFT ，以构建基于 DeepSeekMoE 16B 的聊天模型。实验结果表明，DeepSeekMoE Chat 16B 也实现了与 LLaMA2 SFT 7B 和 DeepSeek Chat 7B 相当的性能。

## 实验设置

**训练数据**。为了训练聊天模型，我们在内部精选的数据上进行监督微调(SFT)该数据集包含140万个训练样本，涵盖广泛的类别，包括数学、代码、写作、问题回答、推理、摘要等。我们的SFT训练数据大部分为英语和中文。

**超参数**。在监督微调期间，我们将批处理大小设置为 1024 个，并使用 AdamW 优化器 进行8个 epoch 的训练。我们使用最大序列长度为 4K ，并尽可能密集地打包训练样本，直到达到序列长度限制。在监督微调中，我们不使用 Dropout 并简单地设置一个恒定的学习率 10-5 ，而不采用任何学习率调度策略。

**评估基准**。为了评估聊天模型，我们使用了类似于[评估基准](https://hl8xut0wpb.feishu.cn/docx/GW6VdKyZuoi2CKx0RYRc9rVCnwg#share-X3vidOQnNoFHFdxRqdbcnVd1nZc)中使用的基准，但进行了以下调整:(1)我们排除了Pile ，因为聊天模型很少用于纯语言建模。(2)我们排除了CHID 由于观察到的结果不稳定，阻碍了得出可靠结论的过程。(3)我们额外包括BBH 以提供更全面的评估，以评估聊天型的推理能力。

## 评价

**基线。** 为了验证 DeepSeekMoE 16B 在对齐后的潜力，我们对 LLaMA2 7B 、DeepSeek 7B 和 DeepSeekMoE 16B 进行了监督微调，我们完全利用相同的数据进行微调，以确保公平性。相应地，我们构建了三个聊天模型，包括LLaMA2 SFT 7B、DeepSeek Chat 7B 和 DeepSeek MoE Chat 16B 。随后，我们比较 DeepSeekMoE Chat 16B 与另外两个密集聊天模型(浮点运算量约为2.5倍)在各种下游任务上的表现。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ece08730dfa4433cb16869e570fc2c07~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074915&x-orig-sign=KpljhXd9F%2FdO1MMQGkyPfbGjzlQ%3D)

**结果**。评估结果如表5所示。我们的主要观察包括:(1)DeepSeek MoE Chat 16B 虽然消耗了近40%的计算资源，但在语言理解和推理(PIQA，ARC，BBH)、机器阅读理解(RACE)、数学(GSM8K，MATH)和知识密集型任务(TriviaQA，NaturalQuestions)方面，其性能与 7B 密集模型相当。(2)在代码生成任务上，DeepSeekMoE Chat 16B 显著优于 LLaMA2 SFT 7B ，在 HumanEval 和 MBpage 上表现出显著的改进。此外它也超越了DeepSeek Chat 7B。(3)在包括MMLU、CEval 和 CMMLU 在内的多项选择题问答基准测试中，DeepSeekMoE Chat 16B 仍然落后于 DeepSeekChat 7B，这与之前提到的基础模型的观察结果一致。然而，值得注意的是，在监督微调后，DeepSeekMoE 16B 和 DeepSeek 7B 之间的性能差距缩小了。(4)得益于在双语语料库上的预训练，DeepSeekMoE Chat 16B 在所有中文基准测试中明显优于LLaMA2 SFT 7B 。这些结果展示了 DeepSeekMoE 16B 在中文和英文方面的平衡能力，增强了其在不同场景中的多样性和适用性。总之对聊天模型的评估突显了 DeepSeekMoE 16B 在利用对齐方面的潜力，并验证了它在实现与密集模型相当的性能时具有持续优势，同时仅使用约 40% 的计算资源。

# DeepSeekMoE 145B 正在进行中

在 DeepSeekMoE16B 的出色表现鼓舞下，我们进一步进行初步尝试，将 DeepSeekMoE 的规模扩大到 145B 。在这项初步研究中，DeepSeekMoE 145B 在 245B 的 token 上进行训练，但与 GShard 架构相比表现出持续的优势，并显示出有望达到或超过 DeepSeek 67B (Dense) 的性能。此外，在完成 DeepSeekMoE 145B 的最终版本和全面训练后，我们还计划将其公开。

## 实验设置

**训练数据和分词**。对于 DeepSeekMoE 145B ，我们使用了与 DeepSeekMoE 16B 完全相同的训练语料库和分词器，唯一的区别是 DeepSeekMoE 145B 在初始研究中使用了 245B 个 token 进行训练。

**模型设置**。对于 DeepSeekMoE 145B ，我们将 Transformer 层的数量设置为 62 ，隐藏维度设置为 4096 。我们采用多头注意机制，共 32 个注意头，每个注意头的维度为 128 。至于初始化，所有可学习的参数都是随机初始化的，标准差为 0.006 。正如在 DeepSeekMoE 16B 中一样，我们也将除第一层外的所有 FFN 替换为 MoE 层。每个 MoE 层由 4 个共享专家和 128 个路由专家组成，每个专家的大小是标准 FFN 的 0.125 倍。每个 token 将被路由到这 4 个共享专家，以及 128 个路由专家中的 12 个。在这种配置下，DeepSeekMoE 145B 大约有 1446 亿个总参数，激活参数的数量约为 22.2B 亿个。

**训练设置**。我们使用 AdamW 优化器 ，其中超参数设置为 $$\beta_1$$=0.9,$$\beta_2$$=0.95, weight decay=0.1 。对于DeepSeekMoE 145B 的初步研究，我们采用了一个 warmup-and-constant 的学习速率调度器。最初，在前 2K 的步骤中，学习速率从 0 线性增加到极大值。随后在剩余的训练过程中，学习率保持不变。DeepSeekMoE 145B 的最大学习率设置为 3.0x10-4 ，梯度裁剪范数设置为 1.0 。批量大小设置为 4.5K ，最大序列长度为 4K ，每个训练批次包含 18M 个 token 。我们训练 DeepSeekMoE 145B 为 13000步，达到 245B 个训练token。此外，我们在训练过程中不使用 dropout 。我们利用管道并行性将模型的不同层部署在不同的设备上，对于每个层所有路由专家将均匀地部署在 4 个设备上(即专家并行性与数据并行性的结合)。由于我们在 DeepSeekMoE 145B 中使用了专家级并行性，应考虑设备级别的负载平衡以减少计算瓶颈。作为回应，我们将设备级别的平衡因子设置为 0.05 ，以鼓励设备之间的平衡计算。此外，我们仍然设置了一个较小的专家级平衡因子为 0. 003 ，以防止路由崩溃。

评估基准。我们根据与 DeepSeekMoE 16B 完全相同的内部工作。

## 评价

**基线**。除了 DeepSeekMoE 145B 之外，我们还考虑了三个额外的模型进行比较。DeepSeek 67B(Dense) 是一个密集模型，总参数数为 67.4B 。GShard 137B 与 DeepSeekMoE 145B 具有相同的隐藏维度和层数，但遵循GShard架构。请注意，DeepSeekMoE 145B 在计算效率方面将每个专家的隐藏维度对齐为 64 的倍数，因此其模型大小比 GShard 137B 大6%。DeepSeekMoE 142B(半激活) 的架构与 DeepSeekMoE 145B 类似，但它只包含2个共享专家，并且 128 个路由专家中只有 6 个被激活。值得注意的是，所有比较的模型，包括 DeepSeekMoE 145B，都共享相同的训练语料库。此外，所有比较的 MoE 模型都是从零开始训练的，并且共享相同的训练超参数。

**结果**。从表6中的评估结果中，我们有以下观察:(1)尽管具有可比较的总参数和计算，DeepSeekMoE 145B 明显优于 GShard 137B，再次突显了 DeepSeekMoE 架构的优势。(2)总体而言，仅使用 28.5% 的计算，DeepSeekMoE 145B 达到了与 DeepSeek67B(密集) 相当的性能。与 DeepSeekMoE 16B 的结果一致，DeepSeekMoE 145B 表现出在语言建模和知识密集型任务上的显著优势，但在多项选择题任务上存在局限性。(3)在更大的规模下，DeepSeekMoE 142B(半激活)的表现并没有比 DeepSeekMoE 145B 落后太多。此外，尽管只有一半的激活专家参数，DeepSeekMoE 142B(半激活) 仍然与 DeepSeek67B(密集) 的性能相当，计算量仅为 18.2% 。它还优于 GShard 137B ，这与[专家专业化分析](https://hl8xut0wpb.feishu.cn/docx/GW6VdKyZuoi2CKx0RYRc9rVCnwg#share-LxAsd2g1LoVibAx4HCWcTPHdnqh)中的的结论一致。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d9c9639579374d11b292d21b4998320d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741074916&x-orig-sign=72g4rRHQEi4gRiVbBr1ZxSJNy10%3D)

# 相关工作

专家混合(MoE)技术最早在1991年由雅各布斯等人提出。用独立专家模块处理不同的样本。沙泽尔等人在2017年，将MoE引入语言模型培训，并建立一个基于语言模型的大规模 LSTM-based MoE模型。随着 Transformer 成为最流行的架构对于NLP，许多尝试将 Transformer 中的 FFN 扩展为 MoE 层 ，以构建 MoE 语言模型 2021年的 GShard 和 Switch Transformer 是采用可学习的 top-2 或 top-1 路由策略将 MoE 语言模型扩展到一个非常大的规模的先驱。Hash Layer 和 StableMoE 使用固定路由策略来实现更稳定的路由和训练。2022年周等人提出了一种专家选择路由策略，其中每个令牌可以分配给不同数量的专家。Zoph 专注于 MoE 模型中的训练不稳定性和微调困难问题，并提出了 ST-MoE 来克服这些挑战。除了对 Mo E架构和训练策略的研究外，近年来还见证了基于现有 MoE 架构的大量大型语言或多模态模型的出现。总的来说，以前的 MoE 模型大多基于传统 top-1 或 top-2 路由策略，为改善专家专业化留下了很大的空间。作为回应，我们的 DeepSeekMoE 架构旨在最大程度地提高专家专业化。