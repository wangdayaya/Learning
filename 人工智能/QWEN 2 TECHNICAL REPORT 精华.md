https://arxiv.org/pdf/2407.10671 2024年9月10日

# 摘要

1.  我们发布了一套全新的基础和指令微调的语言模型 Qwen2 系列，`参数 0.5B 、1.5B、70B和 720B 的密集模型和一个 57B 的混合专家模型，其中 14B 用于激活处理每个 token`。
1.  `Qwen2 超越了大多数开源模型`，包括其前身 Qwen1.5，并且在语言理解、生成、多语言能力、编码、数学和推理等各个基准上表现出与专业模型相当的性能。
1.  `旗舰模型 Qwen2-72B 展示了卓越的性能`。
1.  `Qwen2 精通约 30 种语言`。

# 1引言

1.  Qwen2 是一系列 LLM ，基于 Transformer 架构 ，使用下一个令牌预测进行训练。
1.  `Qwen2-0.5B 和 Qwen2-1.5B 设计用于在便携式设备`，较大的模型适合在不同规模的 GPU 上进行部署。
1.  所有模型都预先训练在一个高质量、大规模的数据集上，该`数据集包含超过 7 万亿个 token` ，涵盖广泛的领域和多种语言。在训练后，所有模型都进行了` SFT 和 DPO （直接偏好优化）`，通过从人类反馈中学习，使它们与人类偏好保持一致，这个过程赋予模型有效地遵循指令的能力。
1.  Qwen2 在基础语言能力和指令微调能力的评估中均优于竞争对手模型。

# 2 TOKENIZER & MODEL

## 2.1 TOKENIZER

1.  和 Qwen 一样采用基于`字节级字节对编码的分词器`。值得注意的是这种分词器表现出较高的编码效率，其压缩率优于其他分词器，这证明了 Qwen2 的多语言能力。
1.  所有大小的模型都使用一个共同的词汇表，包括` 151643 个常规 token 和3个控制标记`。由于分布式训练方面的考虑，嵌入的有效尺寸需要更大。

## 2.2 架构

### 2.2.1 QWEN2 密集模型

Qwen2 密集模型的架构包括多个Transformer层:每个层配备因果注意力机制和前馈神经网络(FFNs)。与 Qwen 的主要区别如下：

-   分组查询注意力：我们`采用分组查询注意力 GQA 而不是传统的多头注意力MH`A 。GQA在推理过程中优化了 KV 缓存的使用，显著提高了吞吐量。各种模型大小的详细KV头配置在2.2.3节中介绍。
-   Dual Chunk Attention with YARN ：为了扩展 Qwen2 的上下文窗口，我们`实现了双块注意力 DCA `，它将长序列分段为可管理的块。如果输入可以以块形式处理，则 DCA 将产生与原始注意力相同的结果。DCA 有助于有效捕获块内和跨块间的token之间的相对位置信息，从而改善长上下文性能。此外，我们还`使用 YARN 重新缩放注意力权重以进行更好的长度外推`。
-   此外，我们`使用 SwiGLU 进行激活`，使用旋转位置嵌入` ROPE `进行位置嵌入，使用 QKV 偏置进行注意力，使用 `RMSNorm `和预归一化进行训练稳定性。

### 2.2.2 QWEN2 混合专家模型

1.  Qwen2 MoE 模型的架构与 Qwen1.5-MoE-A2.7B 的架构非常相似。作为原始 FFN 的替代品，`MOE FFN 由 n 个单独的 FFN 组成，每个 FFN 充当专家`。每个令牌被定向到特定的专家 $$E_i$$，以基于门控网络 $$G$$分配的概率进行计算:

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/fb7545e3386a4eecb87e4d49a5cc0149~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=eo4g%2BbmC0mXXl5bl9b%2F5PAvqcpk%3D)

2.  **专家粒度**。MoE 模型和密集模型之间的关键结构性差异在于，`MOE 层包含多个 FFN ，每个 FFN 都作为单独的“专家”`，从密集架构过渡到 MoE 架构的一个直接策略是将每个专家的参数设置为原始密集模型中单个FFN的参数。例如，从Mistral-7B(Jianget al.，2023a)过渡到Mixtral8x7B(Jiangetal.2024)，需要同时激活八个专家中的两个。不同之处在于，我们的模型采用细粒度专家，在激活更多专家的同时创建更小的专家规模。考虑到专家参数和激活参数的总数量相等，细粒度专家提供了更丰富的专家组合。通过利用这些细粒度专家，Qwen2 MoE 能够促进更多样化和动态的专家利用，从而提升整体性能和适应性。
2.  **专家路由**。设计专家路由机制对于提高MoE模型的性能至关重要。最近，在MoE层中`整合共享和路由特定专家`的趋势显著。我们采用这种方法，因为它有助于在各个任务中应用共享专家，同时保留其他专家用于特定路由场景的选择性使用。引入共享和专业专家为开发 MoE 路由机制提供了更灵活和高效的方法。
2.  **专家初始化**。 我们以类似于升级的方式初始化专家，利用密集模型的权重。相比之下，我们的方法强调细粒度专家之间的多样化，以增强模型的代表性广度。给定指定的专家中间大小 $$h_E$$、专家数量 $$n$$和原始 FFN 中间大小$$h_{FFN}$$，FFN 被复制 $$\left[ \frac{n \times h_{E}}{h_{\text{FFN}}} \right]$$次。这种复制确保了与指定专家数量的兼容性，同时适应任何任意专家中间大小。为了促进每个FFN副本内的多样性，参数沿着中间维度进行洗牌。这保证了每个细粒度专家表现出独特的特征，即使在不同的 FFN 副本中也是如此。随后，从这些 FFN 副本中提取专家，并丢弃其余的维度。`对于每个细粒度专家，其 50% 的参数被随机重新初始化。这个过程在专家初始化中引入了额外的随机性，可能增强模型在训练过程中的探索能力`。

### 2.2.3 模型配置

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3ede19c81831484ba246c36d37a28a95~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=wMb2k1SscVUtUb68ZYuPVkSZfgg%3D)

1.  `Qwen2-57B-A14B 是从 Qwen2-7B 升级而来的`。
1.  与 Qwen1.5 相比 Qwen2 模型每个 token 的键值(KV)大小显著降低。这一特性转化为减少的内存占用，特别是在长上下文推理任务中特别有利。

# 3 预训练

## 3.1 训练前数据

1.  Qwen2 模型的预训练涉及开发一个新的、大规模的改进的高质量多语言数据集:

    1.  **质量提升。** 过滤算法已经通过额外的启发式方法和基于模型的方法进行了改进，包括使用 Qwen 模型来过滤掉低质量的数据。此外这些模型被用来合成高质量的预训练数据。
    1.  **数据扩展。** 与 Qwen1.5 相比，我们收集了大量的高质量代码、数学和30多种语言数据，增强了模型在各个领域的能力。。
    1.  **分布改进。** 为了确保模型学习类似于人类学习的分布，我们在缩小的模型上进行实验，以优化来自各种来源和领域的数据的混合。

1.  基于这些增强，预训练数据从Qwen1.5 的 `3 万亿`个标记扩展到了 `7 万亿`个标记。

1.  进一步放松质量门槛的尝试导致了一个 12 万亿个标记的数据集。然而，在这个数据集上训练的模型在性能上并没有比7万亿个标记的模型有显著提升。据推测`增加数据量并不一定有利于模型的预训练`。考虑到训练成本，我们选择使用质量更高的 7 万亿个标记数据集来训练更大的模型。

1.  所有 Qwen2 密集模型，除 Qwen2-0.5B 外，都是在这个超过 7 万亿个标记的大型数据集上进行预训练的。Qwen2-0.5B 使用 12 万亿个标记的数据集进行预训练。MoE 模型额外接受了 4.5 万亿个标记的预训练，符合升级再利用的原则。与之前的 Qwen 模型类似，高质量的多任务指令数据被整合到 Qwen2 的预训练过程中，以增强上下文学习和指令遵循能力。

## 3.2 长上下文训练

1.  为了增强 Qwen2 的长上下文能力，我们在预训练的最后阶段将`上下文长度从 4,096 个标记增加到 32768 个`标记。通过引入大量的高质量、长数据来完成这一扩展。结合这些改进，我们`将 RoPE 的基本频率从 10000 增加到1000000 `，以优化在长上下文场景下的性能。
1.  为了充分利用模型的长度外推潜力，我们采用了 `YARN 机制` 和`双块注意力机制` 。这些策略使模型能够处理多达 131072 个标记的序列，同时保持高性能，这在初步实验中表现为最小化的困惑度下降。

# 4 后训练

经过广泛的大规模预训练后，我们进入 Qwen2 的后期训练阶段。这个过程可以提高包括编码、数学、逻辑推理、指令遵循和多语言理解在内的能力。此外，它确保模型生成的结果与人类价值观相一致，符合 3H 标准。与传统方法严重依赖大量人工监督不同，我们的方法侧重于以最小的人类注释进行可扩展的对齐任务。具体来说，我们调查了用于 SFT 和 RLHF 的高质量演示和偏好数据的获取方法，旨在最大限度地提高数据质量和可靠性，同时尽量减少人类标记的需求。

## 4.1 后训练数据

1.  后训练数据主要由两个部分组成:演示数据 $$\mathcal{D} = \{ (x_i, y_i) \}$$和偏好数据 $$\mathcal{P} = \{ (x_i, y_i^+, y_i^-) \}$$，其中 $$x_i$$表示指令， $$y_i$$表示满意的响应， $$y_i^+, y_i^-$$ 对 $$x_i$$ 的两个响应，其中 $$y_i^+ $$是相对于 $$ y_i^-$$的优选选择。 $$\mathcal{D} $$在 SFT 中用，而 $$\mathcal{P} $$在 RLHF 中用 。
1.  训练数据的构建涉及两个步骤:协作数据标注和自动数据合成。首先，我们从大规模指令语料库中提取数据本体论，从而获得广泛和多样化的高质量指令集。这些指令经过系统增强，以纳入更大的复杂性。通过人工标注，我们获得目标响应 $$y_i$$ 及其正反例 $$(y_i^+, y_i^-)$$ 。随后，通过多种自动化方法对齐策略被用来合成大量跨代码、数学、指令跟随、创建、角色扮演和安全领域的人工注释数据。简单来说，`就是通过人工和自动化的方法，创建出大量高质量的训练数据，用于训练模型`。

### 4.1.1 协同数据标注

1.  **自动本体提取。** 该过程从应用 InsTag 开始，InsTag 是一种开放集细粒度标记器用于从大规模指令数据集中提取底层本体。随后的手动精修确保了提取的本体的准确性。
1.  **指令选择。** 每个带有标签注释的指令都会根据标签多样性、语义丰富性、复杂性和意图完整性进行评估。基于这些标准，我们选择一组代表性的指令。
1.  **指令进化。** 为了丰富指令数据集，采用了一种自我演化策略，促使 Qwen 模型为现有指令添加约束或要求，从而增加它们的复杂性，并确保数据集内存在各种难度级别。
1.  **人类注释。** 通过使用不同的生成策略和不同规模的 Qwen 模型，对指令获得多个响应。注释者根据他们的偏好对这些响应进行排名，确保最佳响应符合既定标准，从而产生演示和偏好数据。

### 4.1.2 自动化数据合成

1. 保持指令响应的注释质量在大规模上重大挑战，特别需要专业知识、经验、谨慎或耐心的工作。为了解决这些挑战，我们设计了各种自动对齐策略，以规模化地合成数据。

3. **拒绝采样。** 对于具有确定最终答案的数学或类似任务，应用拒绝采样来提高解决方案的质量。大型语言模型的任务是为每条指令生成多个响应，即数学问题推理路径。`模型认为会导致准确结论且被认为是合理的路径被保留，作为演示数据。通过对比正确和错误的路径来生成偏好数据`。

5. **执行反馈。** 对于编程任务，LLMs 会生成代码和测试用例，然后针对测试用例编译运行这些代码来验证其有效性，从而创建示范和偏好数据。这种方法也可以用来评估模型是否按照指令执行任务。

7. **数据再利用。** 为文学写作任务创建响应对标注员来说具有挑战性。为了解决这个问题，我们从公共领域收集高质量的文学作品， LLMs 会利用高质量的文学作品和详细的人物等资料来生成指令和响应，这些数据被用作示范数据。这样可以确保生成的响应既生动又符合人物设定。

9. **合法反馈。** 合法人工智能是指引导 LLMs 根据预定义的原则集生成响应的过程。为了确保遵守安全和价值观等指导原则，汇编了合法的数据集。该数据集`界定了应遵循和应避免的原则`。它用于指导 LLMs 生成与这些指导原则一致或偏离的指导原则，作为演示和偏好数据的参考。

## 4.2 SFT

我们已经收集了一个广泛的教学数据集，其中包含了超过` 500000 个示例`，涵盖了指令跟踪、编码、数学、逻辑推理、角色扮演、多种语言和安全等技能。我们的模型对两个时代进行了微调，序列长度为 32768 。为了优化学习，学习率从 $$7 \times 10^{-6}$$ 逐渐下降到 $$7 \times 10^{-7}$$ 。为了解决过拟合，我们应用了 0.1 的权重衰减和梯度被限制最大值 1.0 。

## 4.3 RLHF

我们为 RLHF 设计的训练机制包括两个连续的阶段:`离线训练和在线训练`。在离线训练阶段，我们使用预编译的偏好数据集 $$\mathcal{P} $$，通过直接偏好优化 DPO 最大化 $$y_i^+, y_i^-$$ 之间的似然差异。在线训练阶段，模型通过利用奖励模型进行实时反馈，迭代地改进其性能。具体来说，我们从当前策略模型中采样多个响应，奖励模型选择最偏好和最不偏好的响应，形成偏好对，用于每个序列中的 DPO 。此外，我们使用在线合并优化器来缓解对齐 tax ，即对齐模型生成与人类偏好相关的性能下降。

# 5 评估

我们为Qwen2 基础模型和指令调优模型实施了全面的评估，包括一般知识理解、语言理解、生成、编码、数学、推理和额外的专业知识领域。

## 5.1 基础语言模型

我们在知识和基本能力的基准数据集上评估这些模型，并应用多语言基准数据集来评估它们对语言的支持。

### 5.1.1核心能力

**基准和评估协议。** `评估基础语言模型核心功能的常见做法是使用少样本或零样本提示实施基准数据集评估`。此次评测主要关注自然语言理解、一般问答、编码、数学、科学知识、推理等模型性能。

  


**Qwen2-72B。** 结果如表2所示。Qwen2-72B 在 MMLU 和 MMLU-Pro 的一般知识理解方面均优于Llama-3-70 B 。在科学评估中 Qwen2-72B 表现出优于 Llama-3-70 的优越性。在编码中 Qwen2-72B 超过 Qwen1.5-72B 。在数学方面 Qwen2-72B 优于 Qwen1.5-72B 。Qwen2-72B 显示的推理能力相当于 Llama3-70B，这归功于其改进的编码和数学数据。在评估中文语言理解方面，Qwen2-72B 显著优 于Mixtral-8x22B 、Llama-3-70B 和 Qwen1.5-72B。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/87b00c5307d940dcad5be4b0d292dee0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=0yce50kJmaGnpi%2Fhpcp1fni9zCY%3D)

**Qwen2-57B-A14B。** 为了评估 MoE 模型，Qwen2-57B-A14B 与类似大小的基线进行了比较。这些基线包括其他MoE模型以及密集模型。和 Qwen1.5-32B 两者都约有300亿个参数。结果如表3所示我们预计激活140亿个参数的 Qwen2-57B-A14B 将匹配 300 亿个参数密集等效 Qwen2 模型的性能。我们的评估显示，Qwen2-57B 在自然语言理解任务中与Yi-1.5-34B表现相当。此外，它在编码和数学任务中优于基线模型。此外，Qwen2-57B-A14B展示了强大的中文语言理解能力，堪比更大的Qwen2-72B模型。本质上Qwen2-57B-A14B 是一个高效的模型，在每次前向传递中仅激活 140 亿个参数，同时保持 300 亿个参数密集模型的性能水平。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6d301bc5468e4fda9f13afc33c2dcbf0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=jd75pyn%2Bg7km5PGfR%2BeWUqSaSJ8%3D)

**Qwen2-7B。** 7B 模型被广泛使用，因为它能够在配备 16GB 内存的加速器上执行 16 位浮点运算。我们的重点是将此模型与其他领先的7B模型进行比较。结果可以在表4中找到。Qwen2-7B在大多数数据集上的表现优于其他模型，特别是在编码任务、数学和中文语言任务方面表现出色。它在多语言理解和考试方面也表现出色。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/91f59c207a38402ea91ff33271c96710~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462741&x-orig-sign=BMHBSNwRmLQqVC2R1TYi2kfjzJc%3D)

  


**Qwen2-1.5B & Qwen2-0.5B。** 为了评估我们较小模型的性能，特别是 Qwen2-1.5B 和 Qwen2-0.5B，我们将其与已建立的基准模型进行比较。结果如表5所示。在语言理解方面，Qwen2-1.5B 优于在类似教科书的数据集上训练的模型 Phi-2。在编码任务中，Qwen2-0.5B 与 Gemma-2B 和 Qwen1.5-1.8B 的表现相当，而 Qwen2-1.5B 除了 Phi-2外，都超过了这些基准模型。与竞争对手相比，Qwen2 的两个模型在数学方面都表现出卓越的性能。在一般推理方面，我们发现 Phi-2 通常优于所有其他模型，这在一定程度上反映了教科书数据对推理能力的重要性。在 TruthfulQA中，Qwen2-1.5B 表现最好，这表明较小的模型不一定会出现幻觉。在中文理解方面，两个Qwen2模型都优于所有其他模型，这一趋势与在各自比较中较大的模型一致。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/abc45c8131524cd7b1272f76b8cfac0a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=CxT2tTYsGRX6ZEyrXxSvhX%2FUJfk%3D)

总的来说，Qwen2 系列在不同模型大小上展示了优于基线的性能。值得注意的是，在所有Qwen2模型中，Qwen2-72B表现出最高性能，这强调了型大小扩展的有效性。

  


## 5.2 指令微调模型

使用公开数据集和基准对基础技能和人类偏好进行评估。特别关注评估长期上下文能力。安全措施包括多语言安全评估和红队演习。

### 5.2.1开放基准评价

为了全面评估指令调优模型的训练质量，我们汇集了自动和人工评估以评估能力和人类偏好。对于基本能力的评估，我们使用和预训练模型评估类似的数据集，这些数据集针对自然语言理解、编码、数学和推理。具体来说，我们评估了MMLU、MMLU-Pro、GPQA和Theorem QA的语言理解和知识，HumanEval、MBPP、MultiPL-E和LiveCodeBenchv1 的编码，GSM8K 和 MATH 的数学。此外，我们通过基准来评估人类偏好对齐和指令遵循的性能，这些基准包括 MT-Bench 、Arena-Hard、AlignBench 、MixEval ，其结果与 Chatbot Arena 近似，以及 IFEval 用于指令遵循。

**Qwen2-72B-Instruct。** 结果如表6所示。强大的基础语言模型有助于提升指令调优模型的下游性能。具体来说，Qwen2-72B 在语言理解、编码和数学等领域表现突出，除了GPQA和 MBPP ，Qwen2-72B相对于基线模型具有显著优势。我们认为这一成就既归功于高质量的预训练模型，也归功于训练后训练的数据和技术的改进。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ec6ae12436ab4867a570a54cd227f431~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=nWY9R9Q%2BuCz6c%2ByNdCAqpfl0ojE%3D)

**Qwen2-57B-A14B-Instruct。** 对于中型模型结果如表7所示。Qwen2-57B-A14B-Instruct 在几乎所有基准测试中表现优异，Qwen2-57B-A14B-Instruct 除了在数学评估中之外，在大多数评估中都有优势。在对齐性评估方面，Qwen2-57B-A14B-Instruct的优势显而易见。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ed7a680470304ec3bb3b5e922728fd3c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=830fO1i%2BclJ0vp8HGeTBRSflvLo%3D)

**Qwen2-7B-Instruct 。** 结果可在表8中 Qwen2-7B-Instruct 在综合评估方面取得了重大进步，尤其是在编码和数学相关任务中取得了更高的分数。与最近的 SOTA模型 Llama-3-8B-Instruct 相比，Qwen2-7B-Instruct 表现出有竞争力的性能，特别是它在编码方面取得了卓越的性能。然而在指令遵循方面，Qwen2-7B-Instruct远远落后于竞争对手为了解决这一限制，我们计划通过提高后训练数据的质量来增强7B模型在指令遵循方面的能力，确保更稳健地理解和执行复杂命令。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/29c45ac54752470baf73e3b750fc6b77~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=3VM1A7XqPnyhyfCNpeVPqHi38zQ%3D)

Qwen2-1.5B-Instruct & Qwen2-0.5B-Instruct。值得注意的是，为较大模型设计的某些数据集的复杂性超出了这些较小模型的能力，因此我们的分析侧重于选定的子集。如表9所示，Qwen2模型在核心能力和指令遵循任务方面都明显优于其前身。这一成就主要归因于预训练数据的扩展。数据扩展仍然是提高模型性能的有效策略，即使在十亿参数以下的模型领域也是如此。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d8900ca0709440289132403c3d445f51~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=wxe5166ja7SnT8e%2FwcCcKFa%2BaxY%3D)

### 5.2.2 内部自动化评估

尽管有多个公开的基准数据集用于评估，但我们认为这远不足以完全理解LLMs的能力。具体来说，我们制作了一系列内部数据集，评估模型的不同能力，例如知识理解、文本生成、编码等。评估是中文和英文的。结果分别收集在表10和表11中。

**中文评估。** 对于中文评估，我们主要比较 Qwen2 模型与Qwen1.5型的性能。对于小模型 Qwen2-1.5B-Instruct 在几乎所有评估中表现优于 Qwen1.5-1.8B-Chat ，即使参数更少。在 7B 模型的比较中，Qwen2 的优势更加明显。值得注意的是，尽管Qwen1.5-110B-Chat 的参数远多于Qwen2-72B ，但 Qwen2-72B 的表现优于Qwen1.5-110B-Chat。MoE模型在大多数领域都表现出优于Qwen1.5-32B-Chat的性能，但知识理解除外。这种差异可能归因于训练前token短缺。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/98bf3ef059ed43a69a4079baff7c115a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=ONNbFhk5g68grvmdHR2shIdNHJo%3D)

**英语评估** 。 Qwen2的小型模型显著优于Qwen1.5的对应模型。然而，与Llama-3-70B 相比，Qwen2-72B-Instruct 略微落后，尤其是在理解和编码方面。我们认为预训练中英语 token 的数量以及后训练中数据的数量和多样性导致了英语性能差距。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/31e633380035482a9fd5f477750d99b0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=OQhA6dD1rFhaSyKMIQtBkZwyU64%3D)

### 5.2.3 长上下文能力

采用了三种评估长上下文能力的方法:` Needle in a Haystack、NeedleBench 和LV-Eval` 。

**Needle in a Haystack。** 这个实验评估模型在大量文本中准确找出事实的能力。文本长度从 8K、16K、…128K 个标记不等，事实被有策略地定位在不同的深度 。每个深度间隔，例如从 0 %到 10 %，包含两个实例。对于超过 32K 的上下文，在本次评估中使用了 YARN 。如图1所示，Qwen2-72B-Instruct 在从整个1 28K 上下文中检索信息方面表现出色。结合其内在优势，该模型成为处理大量文本的最佳选择，前提是可获得足够的资源。此外，同一系列中的模型在不同上下文长度上展示了卓越的性能。具体来说，Qwen2-7B-Instruct 在处理最多 128K 个标记的上下文时达到了高水平的准确性。 Qwen2-57B-A14B-Instruct 能够熟练地管理最多 64K 个标记的上下文 ，而Qwen2系列中的两个较小模型可以支持最多32K个标记的上下文。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e6e7f6cfa3694a3fb1d9ec072837d08e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462741&x-orig-sign=qNj7emFs3Os58ChLmx29XAWi2HU%3D)

**NeedleBench。** NeedleBench 通过包含段落中的多个事实(两个到五个)来增加 NIAH 的挑战，这需要同时识别和多跳推理。表12显示，YARN 和 DCA 的集成显著提高了Qwen2模型的长语境能力。Qwen2-7B-Instruct 超过了声称具有1M 上下文长度的ChatGLM4-9B-1M 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/53f56d9e3c0c4272926d319e56a6113b~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=Mky%2FAlFowC8g6eQIVRscvRPbTnc%3D)

**LV-Eval。** LV-Eval 包含 11 个不同的 QA 数据集，需要同时理解多个证据。为了纠正其原始指标的缺点，该指标过于严格，导致假阴性率较高，我们采用关键字召回作为报告分数。如表12所示，集成 YARN 和 DCA 极大地增强了Qwen2 模型在 LV-Eval 上的长语境能力。Qwen2-7B-Instruct 与 ChatGLM4-9B-1M 相当，尽管在扩展语境下下降更为明显。此外，Qwen2-72B-Instruct在各个方面表现出强劲的性能，证实了其处理长语境任务的能力。

### 5.2.4 多语言评价

对于多语言评估，我们实施了一项全面的人工评估，以评估多语言能力。具体来说，我们设计了多种测试用例，评估大型语言模型的不同能力，并且我们有多种语言的测试用例。对于标注员，我们为每种语言邀请一位专业标注员，他们专门负责该语言的评估。对于每个测试用例，标注员会根据1 到 5 的分数对模型的响应进行评分。

从表13中可以看出， Qwen2-72B-Instruct 显著优于 GPT-3.5-Turbo ，与 GPT-4-Turbo 相当，略微落后于 Claude-3-Opus。这表明我们的多语言预训练和指令调优数据对 Qwen2-72B 的多种语言能力做出了贡献，并且与大多数最先进的专有 LLMs 相当。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6edb8f6022644d95b5b97df1a310f6db~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=feSwha3OZB1VEZGxbhT2c4VOtYY%3D)

### 5.2.5 安全与责任

我们实施了多语言安全评估，测试不同语言的LLMS。具体来说，我们在非法行为、欺诈、色情和隐私等主题上评估模型的安全性能。我们收集了容易越狱的提示，并用它们来测试模型是否可以通过拒绝提供安全的回应。结果如表14所示，其中显示了模型产生的有害响应比例，比例越低越好。可以看出，Qwen2-72B-Instruct的表现优于专有模型GPT-4，并且显著优于开源模型 Mixtral-8x22B-Instruct 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/f7c070f90ad941419d1ac48a580ebc1e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462740&x-orig-sign=cIj9G5OLBraG2iifEJGLmyfisT4%3D)

### 5.2.6 污染分析

对于大型语言模型，什么是污染以及如何运行污染分析仍然是一个活跃的研究领域 。以下我们首先介绍我们如何尝试清理训练语料库以对抗评估数据集，然后估计基准分数在多大程度上受到剩余污染的影响。

在构建预训练和后训练数据集时，我们使用 n-gram 匹配排除潜在的污染数据。然而，我们发现这种方法可能会导致很高的假阴性率，因为可能有常用的表达式，特别是在数学和编码资料，我们还应用了另一个基于最长公共子序列(LCS)的约束。具体来说，我们首先从测试和训练序列中删除所有符号和标点符号，然后进行标记化。对于训练序列 $$s_t$$，如果有一个测试序列 $$s_e$$，使得 $$|LCS(s_t,s_e) \geq 13|$$ 并且 $$|LCS(s_t,s_e) \geq 0.6 \times min(|s_t|,|s_e|)|$$，我们就去掉它。

为了评估数据泄露对测试性能的潜在影响，我们遵循 OpenAl 的方法，构建了一个严格的无污染测试集，以检查在严格去污染后是否会出现显著的性能下降。具体来说，我们通过排除任何样本与预训练或者后训练数据重叠 13-gram(不限制LCS)，然后在测试集上计算相应的指标。

结果如表15所示。虽然一些数据集在严格标准下显示出较高的污染百分比，但我们注意到，大多数被识别为污染样本都是假阳性，主要源于数学和编码数据集。可能因为某些代码片段和数学方程可能非常常见，因此在解决测试数据方面没有提供任何有意义的优势。此外，我们的分析表明，Qwen2 模型在原始和非污染测试数据之间的表现保持一致这表明数据污染的潜在问题没有显著影响模型的性能。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d8e83ad223f04a8791b634fd45d4123c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740462741&x-orig-sign=5atCH7IMGeFhtKE4brIBBwpDGvY%3D)