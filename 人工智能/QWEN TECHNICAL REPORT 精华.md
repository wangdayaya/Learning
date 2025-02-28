<https://arxiv.org/pdf/2309.16609>

# 摘要

1.  QWEN 系列包括`基础预训练语言模型 QWEN` ，以及通过`人类对齐技术进行微调的聊天模型 QWEN-CHAT`。
2.  基础语言模型在各种下游任务中始终表现出色，而使用人类反馈强化学习(RLHF)训练的模型具有高度竞争力。
3.  聊天模型拥有先进的`工具使用和计划`能力，可以创建 `agent` 应用程序。
4.  此外，我们还开发了专门用于`编码的模型 CODE-QWEN 和 CODE-QWEN-CHAT`，以及专注于`数学模型 MATH-QWEN-CHAT` ，这些模型都是基于基础语言型构建的。与开源模型相比，这些模型表现出显著的性能提升。还有`多模态模型 QWEN-VL 和 QWEN-VL-CHAT` 具有理解视觉和语言指令的通用能力。

# 1 导言

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0738f01b0ee74f87adbd54bd160344cd~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=38iRnasQRluARn2I8D16QuuhV74%3D)

1.  QWEN 是一个全面的语言模型系列，包括具有不同参数数量的独特模型。该系列模型包括基础预训练语言模型、通过人类对齐技术(即监督微调(SFT)、人类反馈强化学习(RLHF)等)进行微调的聊天模型，以及编码和数学方面的专业模型。现在，我们正式开源了 14B 参数和 7B 参数的基础预训练模型 QWEN 和对应的聊天模型 QWEN-CHAT2 。 具体细节如下:

    1.  基础语言模型，即 QWEN 训练使用了多达 `3 万亿`个不同文本和代码的 token ，模型在各种下游任务中一直表现出色。
    2.  QWEN-CHAT 模型在 `SFT` 与任务执行、聊天、工具使用、代理、安全等相关的数据集上微调。基准评估表明，SFT 模型可以实现卓越的性能。此外，我们训练了`奖励模型`来模仿人类偏好，并将其应用于 `RLHF` ，以生成人类偏好的聊天模型响应。使用 RLHF 训练的 QWEN-CHAT 模型具有高度竞争力，但仍落后于 GPT-4 。
    3.  提出了代码模型 CODE-QWEN ，包括 `CODE-QWEN-7B 和 CODE-QWEN-14B` ，以及它们的聊天模型 `CODE-QWEN-14B-CHAT 和 CODE-QWEN-7B-CHAT` 。具体来说，CODE-QWEN 在大量的代码数据集上进行预训练，并进一步微调以处理与代码生成、调试和解释相关的对话。实验结果表明，CODE-QWEN 在高水平的代码理解和生成能力方面表现出色。
    4.  提出了数学问题的 MATH-QWEN-CHAT 。MATH-QWEN-7B-CHAT 和 MATH-QWEN-14B-CHAT 在相同大小的开源模型中表现优异，近 GPT-3.5 。
    5.  此外，我们还开源了 QWEN-VL 和 QWEN-VL-CHAT ，它们具有理解视觉和语言指令的通用能力。这些模型在各种评估基准上优于现有的开源视觉语言模型，并支持中文和英文的文本识别、视觉定位、多图对话、讲故事。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e2844f88d3d14576b571ef4689740c2a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=YEW7BkPikPsbmU5alAGWgzS2ZV4%3D)

# 2预训练

## 2.1数据

1.  事实证明，数据规模是开发强大大型语言模型的关键因素，并且必须确保数据具有多样性并涵盖广泛范围。包括公共网络文档、百科全书、书籍、代码等。我们的数据集是多语言的，其中很大一部分数据是`英语和中文`。
2.  为了确保预训练数据的质量，我们开发了一套全面的数据预处理流程。
3.  对于公共网络数据，我们从 HTML 中提取文本，并使用语言识别工具确定语言。
4.  为了提高数据的多样性，我们采用`去重技术`，包括标准化后的精确匹配去重以及使用 MinHash 和 LSH 算法的模糊去重。
5.  为了过滤掉低质量的数据，我们`采用基于规则和基于机器学习`的组合方法。具体来说，我们使用多个模型对内容进行评分，包括语言模型、文本质量评分模型和识别潜在冒犯或不当内容的模型。我们还会从各种来源手动抽样文本并进行审查，以确保其质量。为了进一步提高我们数据的质量，我们选择性地从某些来源上采样数据，以确保我们的模型在多样化的高质量内容上进行训练。
6.  已经证明使用多任务指令预训练语言模型可以增强其在要样本和少量样本上的性能。为了进一步提高我们模型的性能，我们在预训练过程中加入了高质量的 instruction data 。为了保护基准评估的完整性，并去除了评估中使用的测试集中的任何数据有 `13-gram` 的重叠样本。考虑到下游任务的巨大数量，重复对所有任务进行这种过滤过程是不可行的。
7.  我们构建了一个多达 `3 万亿个 token `的数据集。

## 2.2 加工

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/70b4a3b8543149bfb0c383ce6688aeae~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=XDhWG0zcLfag7vIRUw%2BxrUxtaPs%3D)

1.  我们采用`字节对编码(BPE)`作为我们的分词方法，遵循GPT-3.5和GPT-4。我们从开源的 fast BPE 分词器 tiktoken 开始，选择 cl100k 基础词汇作为起点。为了提高模型在多语言下游任务上的性能，特别是在中文方面，我们增加了常用汉字和词语，以及其他语言的词汇。此外，我们将数字拆分为个位数，最终的`词汇量大约为 152K` 。
2.  QWEN 分词器在压缩性能方面的表现如图3所示。我们的研究发现，`QWEN在大多数语言中实现了比其竞争对手更高的压缩效率`。这意味着服务成本可以显著降低，因为 QWEN 的少量分词可以比其竞争对手传达更多信息。此外，我们进行了初步实验，以确保扩大 QWEN 的词汇量不会对预训练模型的下游性能产生负面影响。尽管词汇量增加，我们的实验表明 QWEN 在下游评估中保持了其性能水平。

## 2.3架构

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/cf77e9ca4cb64ef3a0a461391fd7d5a0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=ZmHRep8eV94ucukydVki1bMg3HM%3D)

QWEN 采用了 LLaMA 架构来训练大型语言模型 ，它被广泛认为是顶级的开源 LLM。我们对架构的修改包括:

*   **嵌入和输出投影**。我们选择了`解耦嵌入`的方法，而不是捆绑输入嵌入和输出投影的权重。做出这个决定是为了以内存成本为代价实现更好的性能。
*   **位置嵌入**。我们选择了 `ROPE(旋转位置嵌入) `作为将位置信息纳入我们模型。RoPE 在 PaLM 和LLaMA 等模型上广泛使用。特别是，我们选择使用 FP32 精度进行逆频率矩阵而不是 BF16 或 FP16 ，以优先考虑模型性能并实现更高的准确性。
*   **Bias**。`大多数层删除了偏置`，但在关注度 QKV 层中添加了偏置以增强模型的外推能力。
*   **Pre-Norm & RMSNorm**。在现代 Transformer 模型中，pre-normalization 是比 post-normalization 更广泛使用的方法，它可以提高训练的稳定性。我们已经用 `RMSNorm `取代了传统层规范化技术。这一变化带来了相同的性能，同时也提高了效率。
*   **激活函数**。选择了`SwiGLU` 作为我们的激活函数，它是 Swish 和门控线性单元（Gated Linear Unit）的组合。基于 GLU 的激活函数通常比其他基线选项表现更好。我们将前馈网络(FFN)的维度从隐藏尺寸的4倍减少到 $\frac{8}{3}$。

## 2.4 训练

1.  为了训练 QWEN，我们遵循自回归语言建模的标准方法， 根据之前提供的上下文预测下一个token。
2.  我们使用上下文长度为 `2048` 的模型进行训练。为了创建数据批次，我们随机打乱和合并文档，然后将其截断到指定的上下文长度。
3.  为了提高计算效率并减少内存使用，我们在注意力模块中使用了 `FlashAttention` 。
4.  我们采用标准优化器`AdamW`进行预训练优化。我们设置超参数 $\beta_1$=0.9，$\beta_2$=0.95，和 $\epsilon = 10^{-8}$ 。
5.  我们使用`余弦学习速率调度 cosine learning rate schedule`，每个模型大小都有一个指定的峰值学习速率。学习率衰减到峰值学习率 10% 的最低学习率。
6.  所有模型均`采用 BFloat16 混合精度训练`，以提高训练稳定性。

## 2.5 上下文长度扩展

1.  Transformer 模型在其注意力机制的上下文长度方面有很大的限制。随着上下文长度的增加，二次复杂度计算导致计算和内存成本的急剧增加。在这项工作中，我们实现了简单的无训练技术，这些技术仅在推理期间应用，以延长模型的上下文长度。我们使用的`关键技术之一是 NTK-aware interpolation` 。
2.  与位置插值(PI) 相同地对 RoPE 的每个维度进行标度不同，NTK-aware interpolation 以无训练的方式调整 RoPE 的基准，以防止高频信息的丢失。为了进一步提高性能，我们还实施了一个名为动态 NTK-aware interpolation 的微小扩展 。它动态地按 chunks 来改变尺度，避免了严重的性能下降。这些技术使我们能够有效地扩展 Transformer 模型的上下文长度，而不影响其计算效率或精度。
3.  `QWEN 还包括两种注意力机制:LogN-Scaling 和窗口注意力` 。LogN-Scaling 依赖上下文长度与训练长度的比率的因子对查询和值的点积进行缩放，确保随着上下文长度的增加，注意力 value 的 entropy 保持稳定。
4.  窗口注意力将注意力限制在一个有限的上下文窗口内，防止模型关注距离过远的 token 。我们模型的长上下文建模能力在各个层中有所不同，较低层在上下文长度扩展方面比较高层更敏感。我们为较低层使用较短的窗口，较高层使用较长的窗口。

## 2.6 实验结果

1.  为了评估我们的模型在 zero-shot 和 few-shot学习方面的能力， 我们的评估涵盖了 7 个流行的基准测试。在这次评估中，我们专注于没有对齐的基础语言模型，并从它们的官方结果和 OpenCompass中收集了基线的最佳分数。结果如表2所示。结论如下：

    1.  `在 3 个任务上也 QWEN-14B 比 LLaMA2-70B 表现更好`。
    2.  `QWEN-7B 超越了 LLaMA2-13B` 。
    3.  `尽管 QWEN-1.8B 的参数数量相对较少，但在某些任务上优于较大的模型`。
    4.  `这些发现突显了QWEN模型，特别是QWEN-14B的惊人能力竞争性性能，并表明较小的模型，在某些应用中仍能实现强大的性能。`

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0a85c63d35bb4eb9910ad8b49b146133~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=NPgqTOcgzqFAwc8Ap2Gq36txQYQ%3D)

2.  为了评估上下文长度扩展的有效性，表3以困惑度(PPL)的形式展示了在 arXiv 上的测试结果。这些结果表明，通过结合 NTK-aware interpolation 、LogN-Scaling 和逐层窗口分配，我们可以有效地在超过 8192 个token 的上下文中保持模型的性能。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/a2b329dff68c423f973a4c35fc9c72ec~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=SNMk1IhMN0IdeZ8WHGRamUUmsWE%3D)

# 3 对齐

预训练的大型语言模型被发现与人类行为不一致，使用对齐技术，如监督微调(SFT)和人类反馈的强化学习(RLHF)，可以显著提高语言模型进行自然对话的能力。

## 3.1 监督完成

为了理解人类行为，第一步是进行 SFT ，即在聊天风格的数据(包括查询和响应)上微调预训练的LLM。

## 3.1.1 DATA

1.  为了提高我们监督微调数据集的能力，我们在传统指令数据上面`增加了人工标注的数据`。旨在通过专注于自然语言生成来提高模型在各种任务中的实用性。为了确保模型能够泛化到各种场景，我们特别排除了格式为提示模板的数据，这可能会限制其能力。此外，我们还标注了暴力、偏见和色情等涉及语言模型安全的数据。

2.  除了数据质量，我们`使用了ChatML风格`的格式，这是一种通用的元语言，能够描述回合的元数据(如角色)和内容。这种格式使模型能够有效区分不同类型的信息，包括系统设置、用户输入和助手输出等。通过利用这种方法，我们可以增强模型准确处理和分析复杂对话数据的能力。

### 3.1.2 训练

根据预训练，我们还将下一个令牌预测作为 SFT 的训练任务。我们`对系统和用户输入应用了损失掩码`。该模型的训练过程使用了 AdamW 优化器，具有以下超参数: $\beta_1 = 0.9$ ，$\beta_2=0.95$ ，$\epsilon = 10^{-8}$。序列长度限制为 2048 ，批次大小为 128 。模型总共经历了 4000 步，在最初的 1430 步中学习率逐渐增加，达到 $2.0 * 10^{-6}$ 的峰值。为了防止过拟合，使用了权重衰减，值为 0.1，dropout 设置为 0.1 ，并强制实施梯度裁剪为 1.0 。

## 3.2 人类反馈强化学习

SFT 的泛化能力和创造力可能有限，并且容易过拟合，我们实施了人类反馈强化学习(RLHF)来进一步使 SFT 模型与人类偏好保持一致。这个过程包括训练一个奖励模型和使用近邻策略优化(PPO)进行策略训练。

### 3.2.1 奖励模型

1.  `预训练阶段`。要创建一个成功的奖励模型，就像构建大型语言模型(LLM)一样，首先进行预训练然后进行微调。这种预训练过程，也称为`偏好模型预训练(preference model pretraining ，PMP)` ，需要大量的对比数据。该数据集由样本对组成，每个样本对包含单个查询的两个不同响应及其相应的偏好。同样，这种对比数据用于微调，但由于存在高质量的标注，因此质量更高。

2.  `微调阶段`，我们收集各种提示，并根据人类对 QWEN 模型响应的反馈调整奖励模型。为了确保用户提示的多样性和复杂性得到适当考虑，我们创建了一个包含约 6600 个详细标签的分类系统，并在奖励模型选择标注提示时实施了平衡采样算法，该算法在考虑多样性和复杂性的同时进行采样。为了生成广泛的响应，我们使用了不同大小和采样策略的 QWEN 模型，因为多样化的响应有助于减少注释难度并提高奖励模型的性能。然后，注释员根据标准注释准则对这些答复进行评估，并根据其得分形成对比对。

3.  在创建奖励模型时，我们`使用了相同大小的预训练语言模型 QWEN 来初始化`。另外我们在` QWEN 原始模型中加入了个一个池化层基于特定的结束标记来提取句子的奖励`。在此过程中，学习率已设置为恒定值 $3.0 * 10^{-6}$, 批量大小为 64 。此外，序列长度设置为 2048，训练过程持续 1 个 epoch 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/107beb5b449c41ba92419429cb7c5bad~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=%2BIcvAhyzBx8sMqU5hiY%2FJU6R%2F7U%3D)

4.  我们在测试数据集上采用了准确性作为奖励模型的一个重要但不是唯一的评估指标。在表4中，我们报告了PMP和奖励模型在各种人类偏好基准数据集上的测试准确性。结果表明，`PMP 模型对分布外数据具有很高的泛化能力，而奖励模型对我们的 QWEN 奖励数据集表现出显著的改进`。

### 3.2.2 强化学习

1.  我们的近端策略优化(PPO)过程涉及四个模型:`策略模型、价值模型、参考模型和奖励模型`。在开始 PPO 程序之前，我们暂停策略模型的更新，仅专注于更新价值模型 50 步。这种方法确保价值模型能够有效地适应不同的奖励模型。

2.  在PPO操作中，我们使用同时采样每个查询的两个响应的策略。根据我们的内部基准评估，该策略已被证明更有效。我们将 KL 散度系数设置为 0.04 ，并根据运行均值对奖励进行归一化。

3.  该策略和价值模型的学习率分别为 $1.0 * 10^{-6}$ 和 $5.0 * 10^{-6}$ 。为了提高训练稳定性，我们使用价值损失剪裁，剪裁值为 0.15 。在推理时，策略 top-p 设置为 0.9 。我们的研究发现，尽管略低于 top-p 设置为 1.0 时的熵，但奖励增长更快，最终在类似条件下始终获得更高的评估奖励。

4.  此外，我们实施了预训练的梯度来减轻对齐成本。经验研究表明，使用这种特定的奖励模型，KL 惩罚足够稳健，可以抵消在本质上不是严格代码或数学的基准中的对齐成本。与 PPO 数据相比，必须使用大量得多的预训练数据，以确保预训练梯度的有效性。此外，我们的实证研究还表明，这个系数的值过大可能会严重阻碍与奖励模型的匹配，最终影响最终的匹配效果，而值过小只会对减少对齐税产生微小影响。

## 3.3 对对齐模型的自动化和人性化评价

1.  表5中的结果证明了我们的对齐模型在理解人类指令和生成适当响应方面的有效性。 QWEN-14B-Chat 的表现优于所有其他模型。
2.  除了ChatGPT 和LLAMA2-CHAT-70B 特别是，QWEN 在衡量生成代码质量的 HumanEval 中的表现显著高于其他开源模型。
3.  此外，QWEN的 性能始终优于类似规模的开源模型。这表明我们的对齐方法，即在人类对话的大型数据集上微调模型，在提高模型理解和生成人类语言的能力方面是有效的。
4.  我们相信人类评估至关重要，收集`300条中文指令`，涵盖了广泛的主题，包括知识、语言理解、创造性写作、编码和数学。为了评估不同模型的性能，我们选择了 QWEN-CHAT-7B 的 SFT 版本 和 QWEN-CHAT-14B 的SFT 和 RLHF 版本，并添加了两个强大的基线模型，GPT-3.5 和 GPT-4 。对于每条指令，我们要求三名标注员根据有用性、信息性、有效性和其他相关因素的总体评分对模型响应进行排名。
5.  图4显示了各种模型的获胜率。实验结果清楚地表明`RLHF模型显著优于SFT模型，这表明RLHF可以鼓励模型生成更受人类偏好的响应，但落后于GPT-4`。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/11141c0442864bc186725627a31adc8e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=Yzoci%2FT2XMzAljeTvYG5ik2duZo%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ff4b6f4f55a542a18828ab208495c5ee~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=F6kxUfQ5KQtRsxmI%2BqqXUtBqOvM%3D)

## 3.4 TOOL USE, CODE INTERPRETER, AND AGENT

QWEN 模型旨在具有多功能性，通过利用其在工具使用和计划方面的技能，它们具有出色的能力，可以协助(半)自动化日常任务例如：

*   通过ReAct提示使用看不见的工具
*   使用Python代码解释器来增强数学推理、数据分析等功能
*   作为一个代理，访问 Hugging Face 广泛的多模态模型，同时与人类互动

为了增强 QWEN 作为代理或副驾驶的能力，我们采用自指导策略进行SFT。具体来说，我们利用QWEN的上下文学习能力进行自我指导。通过提供一些示例，我们可以提示QWEN生成更相关的查询，并生成遵循特定格式的输出，例如ReAct。然后，我们应用规则并涉及人工注释员来过滤掉任何嘈杂的样本。之后，这些样本被纳入QWEN的训练数据中，从而生成更新版本的QWEN，使其更适合自我指导。我们多次重复这个过程，直到收集到数量充足且质量优秀且多样性广泛的样本。最终，`我们的样本集由大约2000个高质量样本组成。在微调过程中，我们将这些高质量的样本与所有其他通用SFT样本混合在一起，而不是引入额外的训练阶段。通过这样做，我们能够保留必要的通用能力，这些能力也适用于构建代理应用程序`。

**通过 ReAct Prompting 使用工具。** 我们已经创建并公开了一个基准，用于评估QWEN使用ReAct Prompting调用插件、工具、函数或API的能力 。为了确保公平评估，我们从评估集中排除了QWEN训练集中包含的任何插件。该基准评估模型从最多五个候选者中正确选择插件的准确性:以及传递给插件的参数的合理性和假阳性频率。也就是模型在响应查询时不正确地调用插件。

表 6 显示`随着模型大小的增加，QWEN 在识别查询与可用工具的相关性方面始终具有较高的准确性`。超过某个点在选择适当工具和提供相关论据方面，性能几乎没有提高。这表明当前的初步基准可能相对容易，可能需要在未来迭代中进一步改进。值得注意的是，GPT-3.5作为例外脱颖而出，在这个特定的基准上表现出次优性能。这可能归因于基准主要关注中文语言，这可能与GPT-3.5的能力不匹配。此外，我们观察到GPT-3.5倾向于尝试使用至少一个工具，即使所提供的工具无法有效处理查询。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/78436375ace44ea78ecb62905867f1fb~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=8mR2scqXOxXfXgfmTIwVL1JhzuI%3D)

**使用代码解释器进行数学推理和数据分析**。Python代码解释器被广泛认为是增强LLM代理能力的强大工具。值得研究的是，QWEN能否充分利用这种解释器的全部潜力，以增强其在不同领域的表现，如数学推理和数据分析。为了促进这一探索，我们开发并公开了一个专门为此目的设计的基准。

该基准包括三个主要类别的任务:数学问题解决、数据可视化和其他通用任务（文件处理和网络爬虫）。在可视化任务中，我们将难度分为两个级别。较容易的级别可以通过编写和执行单个代码片段来实现，而无需高级规划技能。然而，更具挑战性的级别需要战略规划和以顺序方式执行多个代码片段。这是因为后续的代码必须基于之前代码的输出进行编写。例如，在继续编写和执行创建图表的附加代码之前，代理可能需要使用一个代码片段检查CSV文件的结构。

**关于评估指标**，我们考虑了`生成的代码的可执行性和正确性`。为了详细说明正确性指标，对于数学问题，我们通过检查代码执行结果和最终响应中是否存在实际数值答案来测量准确性。在数据可视化方面，我们通过使用 QWEN-VL 来评估准确性。关于可执行性和正确性的结果分别列在表7和表8中。QWEN-7B-CHAT 和 QWEN-14B-CHAT 尽管是通用模型，但显著超过了所有其他类似规模的开源替代方案。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5756f86d0fbe4f16887779243012fcd1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=rtYDmmQwRlHh7F5yrICLdRws3ns%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/574ba2a1202e4ab1b8214f3e13215182~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740725433&x-orig-sign=xtYcimpZqz9N8Mg4aJ1kMNIEe2o%3D)

**作为 Hugging Face 代理** 提供了一个框架，称为 Hugging Face Agent 或者 Transformers Agent 该框架允许 LLM agent 与人类交互，解释自然语言命令，并在需要时使用所提供的工具。 `QWEN 表现相当出色，仅略微落后于专有GPT-4，充分证明了 QWEN 作为 HuggingFace Agent 的有效性`。

# 4 CODE-QWEN:编码专用模型

通过代码数据进行强化训练的语言模型可以作为编码、调试和解释等任务的有用工具。我们`使用 QWEN 的基础语言模型`创建了针对编码的特定领域模型，包括继续预训练的模型 CODE-QWEN 和 SFT 的模型 CODE-QWEN-CHAT 。这两个模型都有 14B 和 7B 参数的版本。

## 4.1代码预训练

`以基础模型 QWEN 为开始，然后继续用代码数据进行预训练`。我们继续对模型进行预训练，总共约 `90B 个 token `。在预训练阶段，我们使用基础语言模型 QWEN 初始化模型。我们对模型进行上下文长度最多 8192 的训练解决长的代码编译等场景。类似于2.4节中的基本模型训练，我们在注意力模块中使用了 Flash Attention ，并采用了标准优化器 AdamW 设置 $\beta_1$ =0.9， $\beta_2$ =0.95，和 $\epsilon = 10^{-8}$。我们将 CODE-QWEN-14B 的学习率设置为 $6.0 * 10^{-5}$ ，CODE-QWE-7B 设置为 $3.0 * 10^{-5}$ ，3% 的 warm up 和没有学习率衰减。

## 4.2 代码 SFT

我们确定多阶段的 `SFT` 策略比其他方法表现最佳。在 SFT 阶段，由代码基础模型 CODE-QWEN 初始化模型 CODE-OWEN-CHAT ，通过AdamW 优化器($\beta_1$ =0.9， $\beta_2$ =0.95，和 $\epsilon = 10^{-8}$)进行优化，学习率为 $2.0 * 10^{-6}$ 和 $1.0 * 10^{-5}$ (分别用于14B和7B模型)。学习率随着余弦学习率调度(3%预热步长)增加到峰值，然后保持不变。

## 4.3 评估

`CODE-QWEN 和 CODE-QWEN-CHAT显著优于之前同等规模的基线模型，落后于最先进的模型，如GPT-4`。

# 5 MATH-QWEN:数学推理的专业模型

我们创建了一个名为 MATH-QWEN-CHAT 的数学专业模型系列，它`建立在 QWEN 预训练语言模型的基础上`，专门设计用于在算术和数学方面表现出色，并符合人类行为。我们正在发布该模型系列的两个版本，分别是 MATH-QWEN-14B-CHAT 和 MATh-QWEN-7B-CHAT ，分别具有 14B 和 7B 个参数。

## 5.1训练

我们`在数学推理的增强数学指令数据集上进行了数学SFT`，因此我们直接获得了聊天模型 MATH-QWEN-CHAT 。由于数学 SFT 数据的平均长度较短，我们使用 1024 个序列长度进行更快的训练。数学 SFT 数据集中的大多数用户输入都是考试问题，模型很容易预测输入格式，对于模型预测可能随机的输入条件和数字是没有意义的。因此，我们掩盖了系统和用户输入，以避免在这些输入上进行损失计算，并且在我们的初步实验中发现掩蔽它们加速了收敛。为了优化，我们使用AdamW优化器，除了使用 $2 \times 10^{-5}$的峰值学习率和 50000 的训练步长外，其余超参数与 SFT 相同。

## 5.2 评估

`MATH-QWEN-CHAT 模型在数学推理和算术能力方面表现优于同等规模的开源模型`。
