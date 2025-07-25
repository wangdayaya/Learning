

## 背景问题

1.  Adapter Tuning 增加了模型层数，引入了额外的推理延迟
1.  Prefix-Tuning 难于训练，且预留给 Prompt 的序列挤占了下游任务的输入序列空间，影响模型性能
1.  P-tuning v2 很容易导致旧知识遗忘，微调后表现明显变差

## LORA

https://arxiv.org/pdf/2106.09685

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/358abd80cfc24e859e3ed5401e331f24~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1749698325&x-orig-sign=eUe4P8khZdNbCHFFCYHnPrddixM%3D)

1.  训练的时候固定 PLM 的参数，在原始 PLM (Pre-trained Language Model) 旁边增加一个旁路，做一个降维操作 A 矩阵再升维操作 B 矩阵的操作，用随机高斯分布初始化 A ，用 0 矩阵初始化 B ，只训练降维矩阵 A 与升维矩阵 B ，训练完成后可以将参数 merge 到原模型即可，所以在推理过程中，LoRA 也不影响推理速度。

1.  这种思想有点类似于残差连接，通过增加旁路，达到四两拨千斤的效果。同时使用这个旁路的更新来模拟 Full Fine-Tuning的过程。并且，Full Fine-Tuning可以被看做是 LoRA 的特例。

1.  LoRA 一般应用于线性层。

1.  可训练参数最少，所需显存大大减少，训练耗时较少，效果相比其他的参数有效微调方法更好，并且整体上和全量微调相当甚至更好。

1.  在实际计算时，lora 部分的计算结果并非直接和主干网络相加，而是有个影响系数。具体计算公式如下：

    1.    1. 前向传播 $$ h = W_0 x + \frac{\alpha a}{r} \Delta W x$$
    1.    2. 反向传播 $$ weight = weight + \frac{\alpha a}{r} lora\_weight $$

缺点：

1.  如果追求最佳性能且不受限于计算资源，全参数微调方法可能更合适。因为可以充分挖掘模型的潜力，实现更好的性能。
1.  LoRA微调方法可能不适用于无法设置低秩矩阵的模型，
1.  LORA微调不适用于如记忆新知识的任务场景或者复杂高精度任务场景。

## QLORA

https://arxiv.org/pdf/2305.14314

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/cc62ddb371884da185d6c1845818c871~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1749698325&x-orig-sign=rNf%2BsgoE5FRcAs4sDQMMpwsyFQM%3D)

1.  QLORA 通过将冻结的预训练模型量化为低精度格式（如4位），并在训练过程中使用高精度格式（如bf16）进行反量化，从而在保持模型精度的同时，进一步减少存储需求和计算复杂度。能够在单个 48GB GPU上微调 65B 参数模型，同时保持完整的 16 位微调任务性能。
1.  QLORA 引入了一些创新，以节省内存而不牺牲性能:(a) 4 位 NormalFloat(NF4) ，一种新的数据类型，理论上是正态分布权重的最优信息；(b)双量化通过两次量化来减少平均内存占用；(c)分页优化器来管理内存尖峰，从而避免导致内存不足的错误，这些错误在过去使得大型模型难以在单台机器上进行微调。
1.  除了低秩矩阵，还通过量化减少内存占用，适用于资源有限的环境。

缺点：

1.  建模能力略有下降，量化过程可能存在性能损失。
1.  实现复杂度较高：QLORA 的实现相对 LoRA 更为复杂。

## MORA

https://arxiv.org/pdf/2405.12130

1.  低秩适应(LORA)是一种流行的参数高效微调(PEFT)方法，LORA 中低秩更新机制可能限制了LLMS有效学习和记忆新知识的能力。
1.  受此启发提出 MORA，它使用一个方阵来实现高秩更新，同时保持相同的可训练参数数量。为了实现这一目标我们引入了相应的非参数算子来减少输入维度并增加方矩阵的输出维度。此外，这些算子确保权重可以合并回LLMS，这使得我们的方法可以像LoRA一样部署。 A 和 B 是 LoRA 中的可训练低秩矩阵。M 是 MoRA 方法中的可训练矩阵。灰色部分是非参数运算符，用于减少输入维数并增加输出维数，r 表示两种方法中的秩。
1.  MoRA 在记忆密集型任务上优于 LoRA，并在其他任务上取得了和 LoRA 相当的性能。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b7504c4833ef4e12990182e4764d9a74~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1749698325&x-orig-sign=HmqNLgVLwcnTgCipt4kbWovyMkY%3D)

缺点：

实现复杂度更高：MoRA的实现相对LoRA更为复杂，需要引入额外的非参数化算子来处理输入和输出维度。

## LORI

https://arxiv.org/pdf/2504.07448

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/eb17c7e317d64739b8340953f2761b1e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1749698326&x-orig-sign=FI%2B196JKoY74QWhsX1acoNkVKvI%3D)

1.  LORA 在多任务场景下仍存在显著开销过多和多适配器参数互相干扰问题。
1.  我们提出 LORI ，一种简单而有效的策略，将投影矩阵 A 冻结为随机投影，并使用经过校准之后的特定任务掩码对选中的少量的矩阵 B 参数进行稀疏化训练。这种设计大大减少了可训练参数的数量，比 LoRA少多达95%，降低训练成本，同时保持了强大的任务性能。
1.  此外，LoRI通过利用适配子空间之间的正交性，最小化适配器在合并中的跨任务干扰，并通过稀疏性来缓解灾难性遗忘，支持持续学习。
1.  (a)固定A、稀疏B的单任务适配，大大减少可训练参数量；(b)利用正交性合并多任务适配器；(c)通过稀疏隔离实现持续学习

  


缺点：

1.  实现复杂：需要引入特定任务的稀疏掩码和冻结投影矩阵 *A*，这可能增加了实现和调试的难度
1.  虽然 LoRI 在适配器合并时减少了跨任务干扰，但其合并策略（如合并拼接和线性拼接）的选择和实现需要更多的考虑和实验验证，需要更多的实验和调整。
1.  在某些需要高秩更新的任务中，LoRI 的性能可能不如 MoRA 。
1.  稀疏掩码校准需要消耗额外的计算资源，增加了训练时间。
1.  对稀疏化超参数设置非常敏感，影响模型性能。