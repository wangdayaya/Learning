https://arxiv.org/pdf/2412.15115 2025年1月3日

# 摘要


1.  在预训练方面，我们将高质量预训练数据集从之前的 `7 万亿个令牌扩展到了 18 万亿个` token。
1.  在后训练方面，我们实施了复杂的 SFT，使用了超过` 100 万个样本`，以及多阶段的强化学习，`包括离线学习 DPO 和在线学习 GRPO` 。后训练技术显著增强了人类偏好，并显著改善了长文本生成、结构数据分析和教学跟踪。
1.  Qwen2.5 LLM 开放重量的版本包括 0.5B、1.5B、3B、7B、14B、32B 和 72B 参数的基本模型和指令调优模型。不仅提供了 bfloat16 精度的原始模型，还提供了不同精度的量化模型。
1.  目前专有模型`包括两种专家混合(MoE)变体:Qwen2.5-Turbo 和 Qwen2.5-Plus` 。Qwen2.5-Turbo 和 Qwen2.5-Plus 在成本效益方面表现出色，同时分别在与 GPT-4o 和 GPT-4o-mini 的竞争中表现出色。
1.  Qwen2.5 在各种评估语言理解、推理、数学、编码、人类偏好对齐等的基准测试中表现出顶级性能，开放重量旗舰 `Qwen2.5-72B-Instruct 的性能优于许多开源和闭源模型，并与最先进的开源模型 Llama-3-405B-Instruct 具有竞争力，后者规模大约5倍`。
1.  Qwen2.5 作为基础模型在训练专业模型方面发挥了重要作用，如 Qwen2.5-Math 、Qwen2.5-Coder 、QwQ 和多模态模型。

  


![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/59ef039cd7c84f7e8714e4884bdb5cfd~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=a81BoGSHEbWfJXUN5s1zdszXfXk%3D)

# 1 引言

1.  模型和数据扩展的持续进步，加上大规模预训练之后的高质量监督微调(SFT)和基于人类反馈的强化学习(RLHF)，使得大型语言模型(LLMS)能够在语言理解、生成和推理方面发展出新的能力。

1.  在这个基础上，最近在 inference time scaling 的突破， `通过逐步推理和反思增强了 LLMs 的深度思考能力`。这些发现提高了语言模型的潜力，表明它们可能在科学探索中取得重大突破，因为它们继续展示出更通用的人工智能的涌现能力。

1.  Qwen2.5 关键能力：

    1.  `**Better in Size**`: 与 Qwen2 相比，除了0.5B、1.5B、7B和72B模型外，Qwen2.5还带来了3B、14B和32B模型，对于资源有限的情况来说，这些模型更具成本效益。Qwen2.5-Turbo 和 Qwen2.5-Plus 在准确性、延迟和成本之间提供了很好的平衡
    1.  `**Better in Data**`: 预训练数据从7万亿个标记增加到18万亿个标记，重点放在知识、编码和数学上。预训练分阶段进行，允许在不同混合之间进行转换。后训练的数据达到100万个示例，包含监督微调(SFT)、直接偏好优化(DPO)和群体相对政策优化(GRPO )的阶段。
    1.  `**Better in Use**`:Qwen2 在使用中的几个关键限制已被消除，包括更大的生成长度(从2K令牌增加到8K令牌)、对结构化输入和输出的更好支持(例如，表格和JSON)以及更易于使用的工具。此外，Qwen2.5-Turbo 支持高达 100 万令牌的上下文长度。

# 2 架构与分词器

对于密集模型，我们保持基于Transformer的解码器架构作为Qwen2。该架构包括几个关键组件:`分组查询注意力(GQA)用于高效KV缓存利用，SwiGLU活函数用于非线性激活，旋转位置嵌入(ROPE)用于位置编码用于编码位置信息。注意机制中的 QKV 偏置 和 RMSNorm 作为 pre-normalization 以确保稳定训练`。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/f42cd76666ef4606847aba0bac2b1a9a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=J%2BI0sAywS82oyuy3uKgBNc3NuDs%3D)

在密集模型架构的基础上，我们将其扩展到 MoE 模型架构。这通过用`专门的MoE层替换标准的馈送前向网络(FFN)层来实现，其中每个层包含多个FFN专家和一个路由机制，该机制将 token 调度到 top-K 专家`。遵循 Qwen1.5-MoE 中展示的方法，我们实现了`细粒度专家分割和共享专家路由`。这些架构创新极大地改善了下游任务的模型性能。

我们使用了 Qwen 的分词器，它实现了`字节级别的字节对编码(BBPE)`，使用 `151643 个`常规标记的词汇表。与之前的Qwen版本相比，我们将`控制 token 从 3 扩展到了22`，为工具功能添加了 2 个新标记，并将剩余的标记分配给其他模型功能。这种扩展在所有 Qwen2.5 模型中建立了一个统一的词汇表，提高了一致性并减少了潜在的兼容性问题。

# 3 预训练

## 3.1 预训练数据

Qwen2.5 与其前身 Qwen2 相比，在预训练数据质量方面显著提高，我们开发了一个更大、更高质量的训练前数据集，从Qwen2 中使用的` 7 万亿 token 扩展到 18万亿 token `。这些改进源于几个关键方面:

(1) **更好的数据过滤**。我们利用 `Qwen2-Instruct 模型`作为数据质量过滤器，多维度的分析以评估和评分训练样本。过滤方法相对于我们之前有重大进步，因为它受益于 Qwen2 的强大能力，能够进行更细致的质量评估，并更有效地过滤低质量的样本。

(2)**更好的数学和代码数据**。在 Qwen2.5 的预训练阶段，我`们整合了来自 Qwen2.5-Math 和 Qwen2.5-Coder 的训练数据`。Qwen2.5 在数学推理和代码生成方面继承了强大的能力。

(3)**更好的合成数据**。为了生成高质量的合成数据，特别是在数学、代码和知识领域，我们`利用 Qwen2-72B-Instruct 和 Qwen2-Math-72B-Instruct` 。通过使用我们的`奖励模型和专门的 Qwen2-Math-RM-72B 模型`进行严格的过滤，进一步提高了这种合成数据的质量。

(4)**更好的数据混合**。为了优化预训练数据的分布，我们使用 Qwen2-Instruct 模型来分类和平衡不同领域的内容。电子商务、社交媒体和娱乐等领域的内容过度代表，通常包含重复的、基于模板的或机器生成的内容。技术、科学和学术研究等领域的内容虽然包含更高质量的信息，但代表性不足。`通过下采样过度代表数据和上采样高价值数据`，我们确保训练数据集更加平衡且信息丰富，从而更好地服务于模型的学习目标。

## 3.2 超参数的尺度定律

我们基于 Qwen2.5 的预训练数据开发了超参数的尺度定律 。我们的缩放定律有助于确定密集模型和不同大小的 MoE 模型的关键训练参数，如批量大小 $$B$$和学习率 $$\mu$$ 。

通过广泛的实验，我们系统地研究了模型架构与最佳训练超参数之间的关系。具体来说，我们分析了最佳学习率 $$\mu_{opt}$$ 和批量大小$$B_{opt}$$如何随模型大小 N 和预训练数据大小 D 而变化。我们的实验涵盖了广泛的架构，包括具有 44M 到 14B 参数的密集模型和具有 44M 到 1B 激活参数的 MoE 模型，在从 0.8B 到 600B 标记的数据集上进行训练。`使用这些最佳超参数预测，我们将最终损失建模为模型架构和训练数据规模的函数`。

我们`利用缩放定律来预测和比较具有不同参数数量的MoE模型与密集模型的性能`。这种分析指导我们为MoE模型进行超参数配置，使我们能够通过仔细调整激活参数和总参数，实现与特定密集模型变体的性能对标等。

## 3.3 长上下文预训练

为了优化训练效率，Qwen2.5 采用两阶段预训练方法`:第一阶段使用 4096 个 token 的上下文长度，第二阶段除了 Qwen2.5-Turbo 之外的所有模型，都扩展到 32768 个 token 以处理更长的序列`。同时我们使用 ABF 技术将 RoPE 的基础频率从 `10000 增加到 1000000` 。

对于 Qwen2.5-Turbo ，我们在训练过程中实施了渐进式上下文长度扩展策略，通过`四个阶段推进: 32768 、65536 、131072 ，最终达到 262144 个 token ，RoPE 基频率在每个阶段都为 10000000，在每个阶段我们精心筛选训练数据，包括当前最大长度 40% 的序列和 60% 较短的序列`。这种渐进式训练方法能够平滑适应不断增加的上下文长度，同时保持模型有效处理不同长度的序列并进行泛化的能力。

为了增强模型在推理过程中处理更长序列的能力，我们实施了两个关键策略:`YARN 和双块注意力(DCA)`。通过这些创新，我们实现了序列长度能力的四倍提升使 Qwen2.5-Turbo 能够处理多达 100 万个标记，而其他模型能够处理多达 131072 个标记。值得注意的是，这些方法不仅通过降低困惑度来改善长序列建模，还保持了模型在较短序列上的强大性能，确保了在不同输入长度上的质量一致性。

# 4 后训练

Qwen2.5 比 Qwen2 两个显著的进步:

(1) `**扩大的监督微调数据覆盖范围**`:监督微调过程利用包含数百万高质量示例的大规模数据集。这一扩展特别针对先前模型表现出局限性的关键领域，例如长序列生成、数学问题解决、编码、指令跟随、结构化数据支持、逻辑推理、跨语言传输和鲁棒的系统指令。

(2)`**两阶段强化学习**`:Qwen2.5中的强化学习(RL)过程分为两个不同的阶段:线下强化学习和线上强化学习。

-   `Offline RL`:这一阶段侧重于开发奖励模型难以评估的能力，如推理、事实性和指令遵循。通过精心构建和验证训练数据，我们确保离线RL信号既可学习又可靠，使模型能够有效获取这些复杂技能。
-   `Online RL`:这一阶段利用奖励模型检测输出质量细微差异的能力，包括真实性、有用性简洁性、相关性、无害性和去偏见。它使模型能够生成准确、连贯和有条理的响应，同时保持安全性和可读性。因此，模型的输出始终符合人类质量标准和期望。

## 4.1 SFT

我们构建了一个包含超过 `100 万个SFT实例的数据集`。该模型进行`两个阶段微调`，序列长度为 32768个 token 。为了优化学习，学习率从 $$7 \times 10^{-6}$$ 逐渐降低到 $$7 \times 10^{-7}$$ 。为了解决过拟合，我们应用 0.1 的权重衰减，梯度正则被限制在最大值 1.0 。

-   (1)**长序列生成**:Qwen2.5能够生成高质量的内容，`输出上下文长度可达 8192 个 token` ，相较于 2,000 个标记以下的典型 post-training 响应长度有了显著提升，为了解决这一差距，我们开发了长响应数据集。我们使用反向翻译技术从预训练语料库中生成长文本数据的查询，施加输出长度限制，并使用 Qwen2 过滤掉低质量配对数据。
-   (2)**数学**:我们`引入了 Qwen2.5-Math 的链式思维数据`，它涵盖了各种查询来源，包括公共数据集、K-12 问题集和合成问题。为了确保高质量的推理，我们采用使用奖励模型的拒绝抽样以及带注释的答案作为指导，产生循序渐进的推理过程。
-   (3)**编码**:为了增强编码能力，我们`整合了 Qwen2.5-Coder 的指令调优数据`。我们将多种特定语言的代理纳入协作框架，在近 40 种编程语言中生成多样化且高质量的指令对。我们通过从与代码相关的问答网站合成新示例并从 GitHub 收集算法代码片段来扩展我们的指令数据集。使用多语言沙箱进行静态代码检查，并通过自动化单元测试验证代码片段，确保代码质量和正确性。
-   (4)**指令遵循**:为了确保高质量的指令遵循数据，我们`实施了一个严格的基于代码的验证框架`。在这种方法中，LLMs生成指令和相应的验证代码，以及全面的单元测试进行交叉验证。通过基于执行反馈的拒绝抽样，我们仔细筛选用于监督微调的训练数据，从而确保模型忠实遵循预期的指令。
-   (5)**结构化数据理解**:我们开发了一个`全面的结构化理解数据集`，涵盖了传统任务，如表格问答、事实验证、错误修正和结构理解，以及涉及结构化数据和半结构化数据的复杂任务。通过将推理链纳入模型的响应中，我们显著提高了从结构化数据中推断信息的能力，从而提高了其在这些多样化任务中的表现。这种方法不仅拓宽了数据集的范围，还加深了模型推理的能力，并从复杂的数据结构中得出有意义的见解。
-   (6)**逻辑推理**:为了增强模型的逻辑推理能力，我们`引入了一组多样化的70,000个新查询`，涵盖了各种领域。这些查询包括选择题、真假题和开放式问题。模型被训练以系统性地解决问题，使用一系列推理方法，如演绎推理、归纳概括、类比推理、因果推理和统计推理。通过选代的完善，我们系统地过滤出包含错误答案或错误推理过程的数据。这个过程逐步增强模型逻辑和准确推理的能力，确保在不同类型的推理任务中表现出强大的性能。
-   (7)**跨语言迁移**:为了促进模型的一般能力在不同语言之间的迁移，我们`使用一个翻译模型将高资源语言中的指令转换为各种低资源语言，从而生成相应的响应候选`。为了确保这些响应的准确性和一致性，我们评估每个多语言响应与其原始对应项之间的语义对齐。这个过程保留了原始响应的逻辑结构和风格细微差别，从而在不同语言之间保持其完整性和连贯性。
-   (8)**稳健系统指令**:我们构建了`数百个通用系统提示，以在后训练提高系统提示的多样性`，确保系统提示和对话之间的连贯性。使用不同的系统提示表明，模型保持了良好的性能并降低了方差，这表明稳健性得到了提高。
-   (9)**响应过滤**:为了评估响应的质量，我们`采用多种自动评分方法，包括一个专门的批评模型和一个多代理协作评分系统`。响应经过严格的评估，只有所有评分系统都认为完美的响应才会被保留。这种全面的方法确保我们的输出保持最高质量标准。

## 4.2 离线强化学习

与在线强化学习(RL)相比，离线强化学习允许预先准备训练信号，`这对于存在标准答案但难以使用奖励模型进行评估的任务特别有利`。在本研究中，我们专注于客观查询领域，如数学、编码、指令遵循和逻辑推理，在这些领域获得准确的评估可能很复杂。在前一阶段，我们广泛采用执行反馈和答案匹配等策略，以确保响应质量。在当前阶段，我们重新使用该流水线，`采用SFT模型对一组新查询的响应进行重采样。通过我们质量检查的响应被用作正例，而失败的响应则被用作直接偏好优化(DPO)训练的负例` 。为了进一步提高训练信号的可靠性和准确性，我们使用了`人工和自动化审核`过程。这种双重方法确保了训练数据不仅可学习，而且与人类期望一致。最终，我们构建了一个`大约 150000 个`训练对组成的数据集。然后使用`在线合并优化器(Online Merging Optimizer)`训练模型一个 epoch ，学习率为 $$7 \times 10^{-7}$$ 。

## 4.3在线强化学习

为了开发一个强大的在线强化学习奖励模型，我们遵循一套精心定义的标签标准。这些标准确保模型生成的响应不仅质量高，而且符合伦理和用户中心的标准。数据标签的具体指导方针如下:

-   `真实性`:回答必须基于事实准确性，忠实地反映提供的上下文和指令。模型应避免生成虚假或不支持给定数据的信息。
-   `有用性`:模型的输出应该真正有用，有效地回答用户的查询，同时提供积极、引人入胜、教育性和相关的内容。它应该准确地遵循给定的指令，并为用户提供价值。
-   `简洁性`:回复应简明扼要，避免不必要的冗长。目标是清高效地传达信息，而不是用过多的细节压倒用户。
-   `相关性`:响应的所有部分都应与用户的查询、对话历史和助手的上下文直接相关模型应调整其输出，以确保完全符合用户的需求和期望。
-   `无害性`:模型必须优先考虑用户安全，避免任何可能导致非法、不道德或有害行为的内容。它应该始终促进道德行为和负责任的沟通。
-   `去偏见`:模型应生成无偏见的响应，包括但不限于性别、种族、国籍和政治。它应平等公正地对待所有话题，遵守广泛接受的道德和伦理标准。

用于`训练奖励模型的查询来自两个不同的数据集:公开可用的开源数据和具有更高复杂性的专有查询集`。响应由 `Qwen 模型的检查点生成`，这些模型在不同训练阶段使用不同的方法(SFT、DPO 和 RL)进行了微调。为了引入多样性，这些响应在不同温度设置下进行采样。通过人工和自动标记创建偏好对，DPO 的训练数据也集成到该数据集。

在我们的在线强化学习(RL)框架中，我们采用` Group Relative Policy Optimization (GRPO )`。用于训练奖励模型的查询集与RL训练阶段使用的查询集相同。训练期间查询的处理顺序由奖励模型评估的响应分数的方差决定。具体来说，响应分数方差较高的查询将优先处理，以确保更有效的学习。我们为每个查询抽取 8 个响应。所有模型都使用 2048 全局批量大小和每个 episode 中 2048 个样本进行训练，将一对查询和响应视为一个样本。

## 4.4 长上下文微调

为了进一步扩展 Qwen2.5-Turbo 的上下文长度，我们在` post-training 引入了更长的 SFT 示例`，使其能够更好地适应长查询中的人类偏好。在` SFT 阶段，我们采用两阶段方法。在第一阶段，模型仅通过包含最多 32768 个 token 的短指令进行微调。这个阶段使用与其他 Qwen2.5 模型相同的数据和训练步骤，确保在短任务上表现强劲。在第二阶段微调过程结合了短指令(最多 32768 token)和长指令(最多 262144 token)`。这种混合方法有效地增强了模型在长期上下文任务中的指令遵循能力，同时保持其在短任务上的表现。

`在 RL 阶段，我们使用类似于其他 Qwen2.5 模型使用的训练策略，仅关注于短指令`。这个设计选择是基于三个主要考虑因素:`首先对于长上下文任务 RL 训练的计算成本很高;其次目前缺乏为长上下文任务提供合适奖励信号的奖励模型。最后我们发现仅对短指令采用 RL 仍然可以显著提高模型在长上下文任务中与人类偏好的对齐`。

## 5 评价

为了防止测试数据泄露，我们在构建预训练和后训练数据集时使用n-gram匹配来排除可能受到污染的数据。遵循Qwen2中使用的标准，对于训练序列 $$s_t$$，如果有一个测试序列 $$s_e$$，使得 $$|LCS(s_t,s_e) \geq 13|$$ 并且 $$|LCS(s_t,s_e) \geq 0.6 \times min(|s_t|,|s_e|)|$$，我们就从训练数据中去掉它。

## 5.1 基础模型

我们对Qwen2.5系列的基语言模型进行了全面评估。对基模型的评估主要强调其在自然语言理解、一般问题回答、编码、数学、科学知识、推理和多种语言能力方面的表现。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/878bb505910c4d02843313ef3f87d3fb~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=RbWNAnOinPfqV9gsaYGLK0wkeYw%3D)

**Qwen2.5-72B & Qwen2.5-Plus** Qwen2.5-72的基础模型在广泛的任务中显著优于同类模型。它在仅使用五分之一的参数的情况下，达到了与 Llama-3-405B 相当的结果。此外，与前身 Qwen2-72B相比，Qwen2.5-72在几乎所有基准评估中都显示出显著改进，特别是在通用任务、数学和编码挑战方面表现出色。

**Qwen2.5-14B/32B & Qwen2.5-Turbo** 结果如表3所示。Qwen2.5-14B 模型在各种任务中表现出色。Qwen2.5-32B 尤其展示了卓越的能力，经常超越类似模型的较大模型。值得注意的是，它在数学和编码等具有挑战性的领域表现尤为出色，

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/2661768eb8554a8fb06e20bd7932a51f~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476126&x-orig-sign=Qn7IV8kbGkwnOHJdRvrUM1qKMSY%3D)

  


**Qwen2.5-7B** 结果可以在表4中找到 Qwen2-7B 和 Qwen2.5-7B 的未嵌入参数仅为 6.5B ，而Gemma2-9B的未嵌入参数为8.2B。尽管 Qwen2.5-7B 模型的未嵌入参数较少，但它在许多基准测试中超过了其前身和同类模型。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/97ecc8497a7b4dc39b59774f6c9a5c7d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=XGZJZuZNwozSxUTqwQ0Qw9Bgt5k%3D)

**Qwen2.5-0.5B/1.5B/3B** 结果如表5所示。Qwen2.5-0.5B、1.5B 和3B 在几乎所有基准上都保持了强大的性能。值得注意的是，Qwen2.5-0.5B模型在各种数学和编码任务上优于Gemma2-2.6B。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/aab43c13bbae48a581a8518ccd872f11~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=xVxBQtJRj4%2Fb0nPTxialZSYdgno%3D)

## 5.2 指令微调模型

使用公开数据集和基准来评估基础技能和人类的偏好。

### 5.2.1 开放基准评价

为了全面评估调优模型的指令微调质量，我们汇集了了自动和人工评估，以评估模型的能力和人类的偏好。

**Qwen2.5-72B-Instruct & Qwen2.5-Plus** 如表6所示 Qwen2.5-72B-Instruct 模型提供了卓越的性能，甚至在多个关键基准测试中超过了更大的 Llama-3.1-405-B-Instruct 模型。此外，Qwen2.5-Plus 在13个基准中的9个上表现优于 Qwen2.5-72-B-Instruct 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0689c0c7a8834c0cb764a009bdacbbca~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=b6DWEwFqsUchVyhwMIofeo4wj4k%3D)

  


**Qwen2.5-14B/32B-Instruct & Qwen2.5-Turbo** 结果总结在表7中。与类似大小的其他模型相比，Qwen2.5-32 B-Instruct 模型在大多数任务中表现出卓越的性能。值得注意的是，我们的开源的 Qwen2.5-14B-Instruct 模型在所有基准上都能提供有竞争力的结果，与 GPT-4o-mini 相当。尽管 Qwen2.5-Turbo 模型训练和推理成本显著较低，但在十个基准中的八个上优于 Qwen2.5-14B-Instruct 。这表明 Qwen2.5-TurBo 实现了显著的效率和效果，使其成为资源受限环境的绝佳选择。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/05a7b337e53b4ba6bc85cdc70e3b43db~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=BhJ42zWAmij2RGcLbAtOAbg7pUs%3D)

**Other Instruction-tuned Models** 如表8所示，Qwen2.5-7B-Instruct 模型在除IFEval 之外的所有任务中都明显尤于其竞争对手。对于边缘侧指令模型，如表9所示Qwen2.5-3B-Instruct 模型但在数学和编码任务中超过了它们竞争对手。 Qwen2.5-1.5B-Instruct 和 Qwen2.5-0.5B-Instruct 模型在性能上也比之前的版本有了显著提升，如表 10 所示。这些改进使它们特别适合在资源受限环境中的边缘侧应用。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b5a92ca7a303468aa41afb15eb5fe45c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=UptJcUfr1kdmfeBKOg33iBT64zI%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d8b02b299d26499688cd45c2353a7b6a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=bqpqhGrkQDS%2FoHKhNDjgng4cDb0%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/39e6c6a28a784d45bad6a37b0f4c58d0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476126&x-orig-sign=kxiz2XPhOto3yY%2B%2B3YXK7UQwuOg%3D)

### 5.2.2 内部自动评估

我们开发了一系列内部数据集，旨在评估模型性能的各个方面，包括知识理解、文本生成、编码等。这些评估既包括中文也包括英文。此外，我们还专门评估了指令调优模型的多语言性能。

**English & Chinese Evaluation** 表11是英文，表 12 是中文。 对于较小的模型，我们观察到 Qwen2.5-0.5B 模型的性能与 Qwen2-1.5B 模型相当甚至超过。 Qwen2.5-3B 模型的性能与 Qwen2-7B模型相当。值得注意的是，Qwen2.5-32B 模型的表现显著优于Qwen2-72B模型。我们的旗舰模型Qwen2.5-72B进一步缩小了Qwen与GPT-4 等先进模型之间的差距。特别是，Qwen2.5-72B 在所有指标上都达到或超过了 Llama-31-405B 的性能，除了指令遵循之外。这一成就突显了Qwen2.5-72B 在各种语言处理任务中的竞争力 。Qwen2.5-Plus 解决了之前中文指令学习中的不足，并进一步增强了其在其他领域的优势。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5b9ed845639b481ea755cd1c6ed28bb1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=QeI78rVyQCcXzWroh1ynBtN5Ke4%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8b55abb885d543bcbd34bbef13687233~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476126&x-orig-sign=l5AIkV6lX%2BIp5gwgsz4te3jqETQ%3D)

**Multilingual Evaluation** 表13显示 Qwen2.5在指令遵循、多语言知识和数学推理方面表现出竞争性性能，与可比规模的模型相一致。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e69a6c344dc24dc5a70be2512a068500~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476126&x-orig-sign=ixtNqpI4reoFw%2FLjkJeq9wbBtoM%3D)

  


![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/be42c2a6e7a44b98b7414ebf569aaa99~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=SbphwE05jpi8H6hEdt2cp%2BRhULk%3D)

### 5.2.3 奖励模型

奖励模型是指导 RL 过程的基石，因此我们对 Qwen2.5 系列中使用的奖励模型进行了单独的评估。我们的评估基准包括Reward Bench 、RMB 和PPE ，以及一个内部收集的跨领域中文人类偏好基准(Human-Preference-Chinese)，以提供全面的分析。 总体而言，我们的研究发现 Qwen2.5-RM-72B 在 PPE 和 Human-Preference-Chinese 评估中都排名第一，在 RMB 基准上仅次于Athene-RM-70B。由于缺乏对奖励模型的评估方法，当前的奖励模型通常使用Reward Bench进行评价。然而，我们从多个 RM 基准的评估结果中发现，`在特定基准上进行过度优化可能导致在其他基准上的性能下降，并可能影响下游对齐性能`。这突显了在多种基准上全面评估奖励模型的必要性，而不仅仅是依赖一个基准。

更重要的是，通过反复实验发现:`RM基准上的高分并不一定与所得到的RL模型的性能优越相关`。需要进一步研究更准确的奖励模型评估方法。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/11f106de30c54105bf40648b9ad7578a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476126&x-orig-sign=S1d%2F0f%2FekhfUpXT%2B1q%2FO1vvteSw%3D)

  


### 5.2.4 长上下文能力

我们使用三个基准来评估 Qwen2.5 模型的长上下文能力:RULER 、LV-Eval 和 Longbench-Chat 。在 LV-Eval 中，我们采用关键词召回率作为报告分数，以减少原始指标中存在的假阴性率。结果如表16和表17所示。我们可以看到，在配备了长度外推技术(即DCA+YARN)后，Qwen2.5 模型在三个数据集上展示了强大的长上下文处理能力。其中，Qwen2.5-72B-Instruct 在所有上下文长度上都表现出最强的性能，显著优于现有的开放权重长上下文模型以及像 GPT-4o-mini 和 GPT-4 这样的专有模型。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3536bb0dda7f4fa7a4b8b42b95f3d629~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476125&x-orig-sign=9nyFNUSCs%2BFWOR0QNgV0QD%2BscIg%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c73a3c11e9404e1d87eeaf79feead316~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=%2BnnW51rpo1Mk5IigHxkM8%2FfZ2%2FI%3D)

此外，如图2所示，Qwen2.5-Turbo 在 1M-token 密钥检索任务中实现了 100% 的准确性，展示了其从超长上下文中捕捉详细信息的卓越能力。我`们基于 Minference 开发了一种稀疏注意力机制，显著提高了推理速度，这对于处理长上下文时用户体验至关重要`。对于 1M 个 token 的序列，该方法将注意力机制的计算负载减少了 12.5 倍。图3展示了 Qwen2.5-Turbo 在各种硬件配置下的第一个令牌生成时间(Time To First Token,TTFT)，我们的方法实现了3.2到4.3倍的加速。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/bdce26762bde4470b4aff14f6940ed81~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=oZha8Mgl7Rxyf%2BTKzBDDIEG%2FXG0%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6154f78fafbb48d6b6916bec789b14ec~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740476124&x-orig-sign=GEemTJ2oSh4kFtESfhdxmvm7eSE%3D)