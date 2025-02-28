[arxiv.org](https://arxiv.org/pdf/2401.02954)

[【DeepSeek论文精读】2. DeepSeek LLM:以长期主义扩展开源语言模型-CSDN博客](https://youcans.blog.csdn.net/article/details/145472809)

# 摘要

1.  缩放规律的不同结论给缩放LLMs蒙上了一层阴影，我们深入研究了缩放规律发现有助于在两种常用的开源配置(7B和67B)中实现大规模模型的缩放。
2.  在缩放规律的指导之下，我们引入了DeepSeek LLM。预训练阶段使用`2万亿个token`的数据集获得 DeepSeek LLM Base 。然后进一步进行`监督微调(SFT)和直接偏好优化(DPO)`，从而创建了 DeepSeek Chat 模型。
3.  `DeepSeek LLM 67B 在多项基准测试中超过了 LLaMA-2 70B` ，特别是在代码、数学和推理领域。此外开放式评估显示，我们的 DeepSeek LLM 67B Chat在性能上优于 GPT-3.5 。

# 引言

1.  研究了语言模型的缩放行为，并应用于两个大型模型配置，即 7B 和 67B 。
2.  具体来说，我们首先`研究了批量大小和学习率的缩放规律`，并发现它们随模型大小的变化趋势。在此基础上，我们对数据和模型规模的扩展规律进行了全面研究，成功揭示了最优的模型/数据扩展分配策略，并预测了LLM的预期性能。在开发过程中，我们发现来自不同数据集的缩放规律显示出显著的差异，这表明，数据集的选择显着影响的缩放行为，表明应谨慎行使跨数据集的缩放规律。
3.  在缩放定律的指导下，我们从零开始构建开源的大型语言模型，我们收集了2万亿个token进行预训练，主要是在中文和英文中。在模型层面，我们遵循LLaMA的架构，但用多步学习率调度器替换余弦学习率调度器，在保持性能的同时促进持续训练。我们从各种来源收集了`超过100万个样本进行监督微调(SFT)`。此外，我们使用了直接偏好优化(DPO)来改善模型的对话性能。

# 预训练

## 数据

1.  为了全面增强数据集的丰富性和多样性，进行三个阶段:`去重、过滤和混洗`。去重和混洗阶段通过抽取唯一样本确保数据的多样性表示。过滤阶段提高信息的密度，从而更有效地进行模型训练。
2.  去重策略是扩大了重复数据删除的范围。与在单个 dump中 重复数据删除相比，对整个 CommonCrawl 语料库进行重复数据删除可以更有效地删除重复实例。表 1 显示通过对 91 个 dump 进行重复数据删除比单个 dump 方法删除的文档数量多四倍。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7c217741470148968a8f6e909a80d567~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=rZyeCVfpyolI6BFSXW2mL30Suys%3D)

3.  在过滤阶段，我们专注于为文档质量评估制定稳健的标准。这包括结合语言和语义评估的详细分析，从个体和全局角度提供数据质量视图。
4.  在混洗阶段我们调整方法以`解决数据不平衡问题，重点增加未充分代表领域的样本出现`。这种调整旨在实现更平衡和包容的数据集，确保充分代表各种观点和信息。
5.  tokenizer 实现了`字节级字节对编码( Byte-level Byte-Pair Encoding，BBPE)算法`，预分词技术防止合并来自不同字符类别的标记 。将数字分割为单个数字。基于我们之前的经验，我们将词汇表中常规标记的数量设置为 100000 。tokenizer 是在大约 24GB 的多语言语料库上训练的，我们用 15 个特殊令牌来增强最终的词汇量，使总大小达到 100015 。为了确保训练过程中的计算效率并为未来可能需要的任何额外特殊标记保留空间，我们将模型的词汇表大小配置为 102400 以进行训练。

## 架构

1.  DeepSeekLLM 在微观设计上很大程度`遵循了 LLaMA 的设计`，采用具有 RMSNorm 函数的 Pre-Norm 结构，并使用 SwiGLU作为前馈网络(FFN)的激活函数，中间层维度为 8/3 d\_model 。它还采用了旋转嵌入(Rotary Embedding) 进行位置编码。为了优化推理成本，`67B 模型使用分组查询注意力(GQA)而不是传统的多头注意力(MHA）`。
2.  在宏观设计上，DeepSeek LLM 略有不同。具体来说，DeepSeek LLM 7B 是-个 30 层的神经网络，而DeepSeek LLM 67B 有 95 层。这些层数的调整在保持与其他开源模型参数一致性的同时，也便于模型流水线的划分，以优化训练和推理。
3.  与大多数使用分组查询注意力(GQA)的模型不同，我们扩展了 67B 模型的网络深度参数，而不是常见的增加 FFN 层的中间宽度，以追求更好的性能，见下表：

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/daeab6e5f0714f8ca7ba72a96aa6e95d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=9IRIy3c0BmqWxIih15RwaxkhNn8%3D)

## 超参数

1.  DeepSeek LLM 以 0.006 的标准差初始化，并使用 AdamW 优化器进行训练，以及使用以下超参数: 𝛽1 =0.9，𝛽2=0.95，weight\_decay=0.1。
2.  在预训练中使用多步学习速率调度器代替典型的余弦调度器。具体而言该模型的学习率在 2000 个热身步骤后达到极大值，然后在处理 80% 的训练 tokens 后下降到最大值的 31.6% 。在处理 90% 的 tokens 之后，它进一步减少到最大值的 10% 。训练阶段的梯度剪切设置为 1.0 。
3.  我们观察到尽管训练过程中损失减少趋势存在差异，使用多步学习率调度器的最终表现基本上与余弦调度器一致如图1(a)所示。在保持模型大小不变的情况下调整训练规模时，多步学习率调度器允许从第一阶段重新使用训练，为连续训练提供了独特的便利。因此我们选择多步学习率调度器作为默认设置。我们还通过在图1(b)中展示有，调整多步学习率调度器中不同阶段的比重可以略微提高性能。然而，为了平衡持续训练和模型性能中的重复使用率，我们选择了上述分配方式，分别是 80%、10% 和 10% 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0f31a37399b84d649b630226f1cda913~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=3tc%2BmUxKFoPWgEJUv8tiDFpnQ2E%3D)

4.  批量大小和学习率随模型大小而变化，7B 和 67B 模型预训练阶段的具体参数见下表。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1d9634190b0042779679d2c551b04b73~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=0GbXA2fjhuQ9mfUb4LmuPc0x4Ms%3D)

## 基础设施

1.  我们使用一个高效且轻量级的训练框架HAI-LLM来训练和评估大型语言模型。数据并行性、张量并行性、序列并行性和1F1B管道并行性都被集成到这个框架中，我们还利用 Flash Attention 技术来提高硬件利用率。ZeR0-1 被用于在数据并行rank上划分优化器状态。
2.  还进行了重叠计算和通信的努力，以最小化额外的等待开销，包括ZeR0-1中最后一个微批次和 reduce-scatter 操作的反向过程，以及GEMM计算和 all-gather/reduce-scatter的序列并行。一些层/操作被融合以加速训练，包括尽可能的 LayerNorm 、GEMM 和 Adam 更新。
3.  为了提高模型训练的稳定性，我们使用 bf16 精度训练模型，但使用 fp32精度累积梯度。通过原地交叉熵来减少GPU内存消耗，即:在交叉熵CUDA内核中(而不是在HBM中预先转换)将bf16对数转换为fp32精度，计算相应的bf16梯度，并用其梯度覆盖 logits 。
4.  模型权重和优化器状态每5分钟异步保存一次，这意味着在最坏的情况下偶尔发生硬件或网络故障时，我们最多会损失5分钟的训练时间。这些临时的模型检查点会定期清除，以避免消耗过多的存储空间。我们还支持从不同的3D并行配置恢复训练恢复，以应对计算集群负载的动态变化。
5.  至于评估，我们在生成任务中使用 VLLM 框架，在非生成任务中采用连续批处理，以避免手动批处理大小调整和减少 token 填充。

# 缩放定律 Scaling Laws

1.  缩放规律表明，增加计算量C、模型规模 N （模型参数量） 和数据规模 D （ token 数量），C可以近似为C=6ND。因此，在增加计算预算时，如何优化模型规模和数据规模之间的分配也是缩放规律中的一个关键目标。规模法则的结果表明，增加计算预算继续产生显著的效益，这进一步鼓励了模型规模的增加。
2.  然而如表4所示，早期关于最优模型/数据缩放分配策略的研究出了不同的结论，这引发了关于缩放定律普遍适用性的质疑。此外，这些研究通常缺乏超参数设置的完整描述，使得在不同计算预算下达到最优性能模型的情况不确定。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/50ba3ad72a244556b1d28798603ec9fc~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=EtMcPgAlXKEVPnDGlvbxmZXn1tI%3D)

3.  为了确保在不同计算预算下的模型都能达到最佳性能，我们首先研究了超参数的缩放规律。经验表明，在训练过程中，大多数参数的最佳值在计算预算变化时保持不变。因此，这些参数与[超参数](https://hl8xut0wpb.feishu.cn/docx/JSxfdVnH1omMCRxe1GWcbD6YnXg#share-CUQTdnAzlo58Zbx9hQtccuWjn1P)一节中概述的参数一致，并且在不同的计算预算下保持不变。对性能影响最大的超参数是批次大小和学习率，进行了重新检查。

4.  通过广泛的实验，我们建立了计算预算C与最佳批量大小和学习率之间的幂律关系模型。我们称之为超参数缩放定律的这种关系为确定最佳超参数提供了一个经验框架。这种方法确保了不同计算预算的模型能够达到近乎最佳的性能。

5.  然后，我们研究了模型的缩放规律和数据尺度。为了降低实验成本和拟合难度，我们采用了 Chinchilla 的 IsoFLOP 轮廓方法来拟合缩放曲线。为了更准确地表示模型尺度，我们采用了一种新的模型尺度表示方法，即非嵌入FLOPs/token 作为 M，取代了之前使用的模型参数 N ，并替换了近似计算预算公式 C=6ND 为更精确的 C=MD。实验结果提供了关于最佳模型/数据缩放分配策略和性能预测的见解，并准确预测了DeepSeek LLM 7B和67B模型的预期性能。

6.  此外，在探索缩放规律的过程中，我们使用的数据经过了多次选代，质量不断提高。我们尝试在各种数据集上拟合缩放曲线，发现数据质量对最优模型/数据缩放比例分配策略有显著影响。数据质量越高，应分配给模型缩放的计算预算就越多。这意味着在相同的数据规模下，高质量的数据可以驱动更大模型的训练。最优模型/数据缩放分配策略的差异也可以作为评估数据质量的间接方法。

7.  总之，我们在缩放规律方面的贡献和发现可以归纳为以下几点:

    1.  我们建立了超参数的缩放规律，为确定最优超参数提供了经验框架，
    2.  我们不使用模型参数 N，而是采用非嵌入FLOPs/token M 来表示模型规模，从而得到更准确的优化模型/数据缩放分配策略，并更好地预测大规模模型的泛化损失。
    3.  预训练数据的质量影响最优模型/数据扩展分配策略。数据质量越高，应分配给模型扩展的计算预算就越多。

## 超参数的缩放定律

1.  我们最初在计算预算为 1e17 的小规模实验中对批次大小和学习率进行了网格搜索，特定模型大小(177MFLOPs/token) 的结果如图2(a)所示。`结果表明，在泛的批次大小和学习率选择范围内，泛化误差保持稳定。这表明在相对较宽的参数空间中可以实现接近最优的性能`。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/253eca771cdf4e0cb6a8785fd401df63~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=%2BKz7Vn4Ooof0Vg%2FYHaMSsza3GyI%3D)

2.  我们利用前面提到的多步学习速率调度器，通过重复使用第一阶段的方式，有效地训练了不同批量大小、不同学习率、计算规模在 1e17 到 2e19 的多个模型 。考虑到参数空间中的冗余，我们将模型使用的参数视为接近最优的超参数，其泛化误差不超过最小值的0.25% 。然后，我们根据计算预算 C 拟合批次大小 B 和学习率 n 。拟合结果如图3所示，`表明最优批次大小 B 随着计算预算 C 的增加而逐渐增加，而最优学习率 n 逐渐减小`。这符合在扩展模型时对批量大小和学习率的直观经验设置。此外，所有接近最优的超参数都落在较宽的范围内，这表明在这个区间内选择接近最优的参数相对容易。我们最终拟合的批量大小和学习率的公式:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e8f5654db7f34db0a76468952621d467~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=L6QIX1pJLU2r8gFjyTuqZ1dJ4vA%3D)

3.  我们在一系列模型上验证了我们的公式，计算预算为1e20，特定模型大小(每个token 2.94B FLOPS)的结果如图2(b)所示。结果表明拟合参数位于最佳参数空间中心位置。随后的章节还表明，我们为DeepSeek LLM 7B 和 67B 模型拟合的参数也实现了良好的性能。
4.  然而，需要注意的是，我们还没有考虑计算预算C之外的其他因素对最优超参数的影响。此外，我们观察到，在具有相同计算预算但模型/数据分配不同的模型中，最优参数空间略有不同。这表明需要进一步研究以了解超参数的选择和训练动态。我们将在今后的工作中探讨这些方面。

## 最优模型估算与数据尺度

1.  在推导出接近最优的超参数的拟合公式后，我们开始拟合缩放曲线，并分析最优模型/数据缩放分配策略。该策略分别找到满足 $N_{\text{opt}} \propto C^a$和 $D_{\text{opt}} \propto C^b$的模型缩放指数 a 和数据缩放指数 b 。数据规模 D 可以由数据集中的 token 数量表示。在以前的工作中，模型规模通常由模型参数表示，包括非嵌入参数 $N_{1} $ 和完整参数 $N_{2} $ 。计算预算 C 与模型/数据规模之间的关系可以近似表示为 C=6ND ，这意味着我们可以使用 $6N_{1} $ 或 $6N_{2} $来近似表示模型规模。然而，由于 $6N_{1} $ 或 $6N_{2} $ 都没有考虑注意力操作的计算机开销，而且 $6N_{2} $ 还包括对模型能力贡献较少的词汇计算，因此在某些设置下，它们都有显著的近似误差。
2.  为了减少这些错误，我们引入了一种新的模型规模表示:非嵌入FLOPs/token M ，M 包括注意力操作的运算开销，但不考虑词汇表的计算。使用 M 表示模型规模，计算预算 C 可以简单地表示为C=MD。 $6N_{1} $、 $6N_{2} $ 和 M 之间的具体差异如以下公式所示

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/9f030be2dc464c6aa305d5d2dd1754c3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=mGm%2FvTpU%2BkO6StnTAA2RgvTW294%3D)

3.  其中， $n_{layer} $表示层数，$d_{model} $表示模型宽度，$n_{vocab} $表示词汇表大小，$l_{seq} $表示序列长度。我们评估了这三个表示在不同规模模型之间的差异，如表3所示。`结果表明，无论是 $$6N_{1} $$ 或 $$6N_{2} $$ ，都高估或低估了不同规模模型的计算成本。这种差异在小规模模型中尤为明显差异高达50% 。`在拟合缩放曲线时，这种不准确性可能会引入大量的统计误差。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/71543e73da0f4ce5aa15d2bf8eb45cfe~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=wbddqE4yES9y0ClhR3DRtzuAGwM%3D)

4.  有关模型比例尺不同表示形式的进一步分析请参见图6。衡量标准是验证集上的每字节比特数。计算预算较高的时候，三种缩放定律区别不明显，但是在计算预算较少的时候，有明显的差异，前两种方式会出现高估或者低估的情况，第三种方式是估算最准确的。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/fe3d657007e34abc9005f00ce1541f39~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=OkfqxOhhBntP0LCP0LFpH51HRyA%3D)

5.  在采用 M 表示模型规模后，我们的目标可以更清晰地描述为: 给定计算预算C=MD，找到使模型泛化误差最小的最优模型规模 $M_{opt} $ 和最优数据规模 $D_{opt} $ 。这个目标可以形式化为:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/427865fe26c84428bb2193eae28b6b24~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=Qbw33Rzdpq3qo858Q2NZZfZmcHs%3D)

6.  为了降低实验成本和拟合难度，使用Chinchilla的 IsoFLOP 剖面方法 来拟合缩放曲线。我们选择了 8 种不同的方法计算预算范围从1e17 到 3e20，并为每个预算设计了大约 10 个不同的模型数据规模。每个预算的超参数由公式(1) 确定，并在独立的验证集上计算泛化误差，验证集与训练集分布类似，包含1 亿个 tokens 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c582ceee13344085871a6583f05f89c2~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=CY3KuUZlCZSwL3fVgYizgJ2Mf1A%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/94e4e7adeef441348cca18ba9427e915~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=17bKhMgb4iKjgMpjK3LTrqRB%2B5w%3D)

7.  图4显示了 IsoFLOP 曲线和模型/数据缩放曲线，这些曲线是通过为每个计算预算使用最佳模型/数据分配来拟合的。最佳非嵌入 FLOPs/token $M_{opt} $ 和最佳 tokens $D_{opt} $ 的具体公式如下:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/95bdfc6ea2584a30b876c8528e5704bf~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=qp89nB2PPdfMxZU2gM4RjsXA%2FdQ%3D)

8.  此外，我们根据计算预算 C 和最佳泛化误差拟合了损失缩放曲线，衡量标准是验证集上的每字节比特数，并预测了DeepSeek LLM 7B 和 67B 的泛化误差，如图5所示。结果表明，使用小规模实验可以准确预测具有 1000 倍计算预算的模型的性能。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1286448b2234464c95284d76dec01c29~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=6qf4VPbZZR0fNyDsYnwh96GdM%2Fg%3D)

## 不同数据的尺度法则

1.  在 DeepSeek LLM 的开发过程中，数据集经过多次迭代优化，调整了不同数据源的比例，同时提高了整体质量。这使我们能够进一步分析不同数据集对缩放规律的影响。
2.  我们研究了三个不同数据集的缩放规律:早期内部数据、当前内部数据和 OpenWebText2 。数据质量从前到后不断提升。
3.  一个有趣的观察是，这三个数据集的最佳模型/数据缩放分配策略与数据质量保持一致。如表4所示，`随着数据质量的提高，缩放指数 a 逐渐增加，而数据缩放指数 b 减少，这表明增加的计算预算应更多地分配给模型而不是数据`，这一发现可能也解释了在先前关于缩放定律的研究中观察到的最佳模型/数据缩放分配策略的巨大差异。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4a3d972f6ad5452aa321dc22619edd25~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=hzb0YzvtACgmotKPv96c1IOuK%2BM%3D)

4.  对这一发现的直观猜测是，`高质量的数据通常意味着逻辑清晰，在充分训练后预测难度较小。因此，在增加计算预算时，扩大模型规模更为有利。`

# 对齐

我们收集了大约`150万`个英语和中文指令数据实例，涵盖了广泛的有用和无害主题。我们的有用数据包含120万个实例，其中 31.2% 用于通用语言任务，46.6% 用于数学问题，22.2% 用于编码练习。安全数据包括 30 万个实例，涵盖各种敏感主题。

我们的对齐流程包含两个阶段：

1.  **监督微调**:我们对 7B 模型进行了4个 epochs 的微调，但对 67B 模型仅进行了2个 epochs ，因为我们观察到 67B 模型的过度矫正问题很严重。观察到 GSM8K 和 HumanEval 在7B 模式下持续改善，而 67B 模式很快达到上限。7B 和 67B 模型的学习率分别为 1e-5 和 5e-6 。除了监控基准的准确性之外，我们还会在微调过程中评估聊天模型的重复率。我们收集了总共3868个中文和英文 prompts ，并确定了生成的回应中未能终止并无休止而无限重复一段文本的比例。我们观察到，随着数学 SFT 数据量的增加，重复率往往上升。这可以归因于数学 SFT 数据偶尔在推理中包含头相似的模式。因此，较弱的模型难以掌握此类推理模式，导致重复性回答。了解决这个问题，我们尝试了两阶段(微调 和DPO） ，两种方法都几乎保持了基准分数并显著降低了重复率。
2.  **DPO**:为了进一步增强模型的能力，我们使用了直接偏好优化算法DPO。我们根据有用性和无害性构建了DPO训练所需的偏好数据。对于有用性数据，我们收集了多语言 prompts ，涵盖了创意写作、问题回答、指令遵循等类别。然后我们使用我们的 DeepSeek Chat 模型作为响应候选者生成响应。类似的操作也应用于无害偏好数据构建。我们为 DPO 训练 1 个 epoch ，学习率为 5e-6，批量大小为512，我们使用了学习率预热和余弦学习率调度器。我们发现DPO可以增强模型的开放生成技能，同时在标准基准上产生很小的性能差异。

# 评估

## 基础模型

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8657a3302ae542aaa877f5dc283eb63a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=WV9aJK3h36CQMU0ZITaloGTXW1g%3D)

## 聊天模型

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4753311bf684453aa43d85fae98d89bd~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1741339424&x-orig-sign=eti609%2F%2BR6KQTC0odD2kD9HL58k%3D)
