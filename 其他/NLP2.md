 # NLP2

自然语言处理（英语：Natural Language Process，简称NLP）是计算机科学、信息工程以及人工智能的子领域，专注于人机语言交互，探讨如何处理和运用自然语言。


# bf16 和 bfloat16

`TrainingArguments` 中的 `bf16=True` 参数和 `AutoModelForCausalLM.from_pretrained` 方法中的 `torch_dtype="bfloat16"` 参数都与模型训练时的数据类型有关，但它们的作用和使用场景有所不同：

1. **`TrainingArguments` 中的 `bf16=True` 参数**：
   - 这个参数是 `TrainingArguments` 类的一部分，用于配置训练过程中的优化器和模型的数据类型。当设置为 `True` 时，它指示训练过程中使用 bfloat16（Brain Floating Point 16）精度，这是一种16位的浮点数格式，由Google开发，旨在提供与32位浮点数相似的动态范围，同时减少内存占用和加速计算。
   - `bf16` 参数主要用于优化器状态和梯度的存储，以减少内存消耗并可能加速训练，特别是在支持bfloat16的硬件上。

2. **`AutoModelForCausalLM.from_pretrained` 方法中的 `torch_dtype="bfloat16"` 参数**：
   - 这个参数用于指定加载模型时模型权重的数据类型。当设置为 `"bfloat16"` 时，模型的权重将被加载为 bfloat16 格式，这同样可以减少模型的内存占用并可能加速计算。
   - `torch_dtype` 参数直接影响模型权重的数据类型，适用于需要在模型推理或训练时使用特定数据类型的场合。

**区别**：
- `bf16=True` 参数主要影响训练过程中优化器状态和梯度的数据类型，通常在训练开始前设置，而 `torch_dtype="bfloat16"` 参数则影响模型权重的数据类型，一般在加载模型时指定的。
- 使用 `bf16=True` 可能会影响训练过程中的梯度更新和优化器的行为，而 `torch_dtype="bfloat16"` 主要影响模型的存储和计算效率。

在实际应用中，这两个参数可以根据训练环境和硬件支持来设置，以达到内存优化和计算加速的目的。需要注意的是，并非所有的硬件都支持 bfloat16 精度，因此在设置这些参数之前，需要确认你的硬件和深度学习框架是否支持这种数据类型。


举例，我在某次训练中使用如下关键参数训练消耗大约 38 分钟。


    TrainingArguments（bf16=True）和 AutoModelForCausalLM.from_pretrained( torch_dtype="bfloat16")

但是用如下关键参数，预计消耗 46 多个小时。

    TrainingArguments（bf16=True）和 AutoModelForCausalLM.from_pretrained( torch_dtype="float16")


# accelerate、trainer、peft
- [从 PyTorch DDP 到 Accelerate 到 Trainer，轻松掌握分布式训练](https://huggingface.co/blog/zh/pytorch-ddp-accelerate-transformers)
- [PEFT：在低资源硬件上对十亿规模模型进行参数高效微调](https://huggingface.co/blog/zh/peft)

# 量化技术和混合精度


量化技术和混合精度计算都是用于提高深度学习模型效率的技术，但它们的目标和实现方式不同。

### 量化技术
- **目的**：通过减少模型参数的位数来降低模型的存储需求和计算开销。**量化**主要关注于减小模型的存储和计算需求，通常在推理阶段使用，适合部署在资源受限的环境中。
- **实现方式**：
  - 将浮点数（通常是32位）参数转换为低位数（如8位整数）。
  - 常见的方法有权重量化、激活量化和后训练量化。
  - 量化可以在训练后进行（Post-Training Quantization），也可以在训练过程中进行（Quantization-Aware Training）。
- **优点**：
  - 显著减少模型大小，适合在内存和存储有限的设备上运行。
  - 提高推理速度，特别是在专用硬件（如TPU、FPGA）上。

### 混合精度计算
- **目的**：在训练过程中使用不同的数值精度，以提高计算效率和模型性能。**混合精度计算**主要针对训练过程，通过使用不同精度来加速训练，提高性能，同时保持模型的准确性。
- **实现方式**：
  - 使用16位浮点数（FP16）进行大部分计算，而保持关键操作（如梯度计算和权重更新）使用32位浮点数（FP32）。
  - 使用动态损失缩放以避免在使用低精度时出现数值稳定性问题。
- **优点**：
  - 加速训练过程，减少内存带宽需求。
  - 在支持混合精度计算的硬件（如NVIDIA的Ampere架构GPU）上，可以实现更快的计算和更低的内存使用。

 


# BLEU、ROUGE 评估模型

BLEU、ROUGE等指标通常用于评估自然语言处理模型（尤其是机器翻译和文本摘要）的性能。以下是使用这些指标的常规步骤：

### 1. 数据准备
- **参考文本（Reference Texts）**：为每个输入文本准备一个或多个标准答案（ground truth），用于与模型生成的文本进行比较。
- **生成文本（Generated Texts）**：使用模型生成的文本，通常是在给定输入后生成的预测结果。

### 2. 计算BLEU分数
- **n-gram匹配**：计算生成文本和参考文本中n-gram的重叠程度（通常n取1到4）。
- **精确度计算**：计算每个n-gram的精确度（即生成文本中有多少n-gram在参考文本中出现）。
- **惩罚机制**：使用惩罚机制（如brevity penalty），防止模型生成过短的文本。
- **最终得分**：将不同n-gram的精确度结合，得到最终的BLEU分数。

### 3. 计算ROUGE分数
- **召回率**：计算生成文本中与参考文本匹配的n-gram的比例，通常关注ROUGE-N（n-gram召回率）和ROUGE-L（最长公共子序列）。
- **精确度和F1分数**：除了召回率，ROUGE还可以计算精确度和F1分数，以便更全面地评估模型性能。
- ROUGE-1:**定义**：计算生成文本和参考文本之间一元组（unigram，单个词）的重叠情况。 **意义**：主要衡量生成文本中出现的单词与参考文本的相似性，反映了基本的内容覆盖。侧重于内容的覆盖。
- 2. ROUGE-2：**定义**：计算生成文本和参考文本之间二元组（bigram，两个相邻词）的重叠情况。 **意义**：评估生成文本的连贯性和短语结构的匹配，强调了词与词之间的关系。考虑短语的结构和关系。
- 3. ROUGE-L:**定义**：计算生成文本与参考文本之间的最长公共子序列（Longest Common Subsequence）。 **意义**：捕捉生成文本与参考文本在顺序上的相似性，强调生成文本的语法和结构的流畅性。关注文本的整体流畅性和顺序。

### 4. 实现工具

- 可以使用Python的`nltk`库、`sacrebleu`库或`rouge-score`库等工具来简化BLEU和ROUGE的计算。
- 将BLEU和ROUGE分数与其他模型进行比较，或与基线模型进行对比，以评估改进效果。
- 分析得分的意义，判断模型的优缺点，进一步优化模型。


 

 


# 显卡价格

- Volta 架构：Volta 架构是 NVIDIA GPU 的第六代架构，发布于 2017 。**V100 至少1万美元**。
- Turing 架构：Turing 架构是 NVIDIA GPU 的第七代架构，发布于 2018 年。
- Ampere 架构：Ampere 架构是 NVIDIA GPU 的第八代架构，2020 年发布。为了在遵守美国限制规则**A800主要是将NVLink的传输速率由A100的600GB/s降至了400GB/s，其他参数与A100基本一致。**，A800至少1.2万美元，A100至少1.5万美元。
- Hopper 架构：Hopper 架构是 NVIDIA GPU 的第九代架构，2022 年发布。为了在遵守美国限制规则**H800 的芯片间数据传输速度是 H100 的一半**。H100 大约4万美元，H800大约是其八折。




# model.train() 和model.eval()
model.eval() 这行代码将模型设置为评估模式。在评估模式下，模型的一些特定层（如Dropout和BatchNorm）会改变它们的行为，以适应非训练环境。


#  `BatchNorm` 和`LayerNorm` 

`BatchNorm`（批量归一化）和`LayerNorm`（层归一化）都是用于归一化神经网络层输出的技术，但它们在模型训练中的具体差异主要体现在归一化的数据范围、计算方式和适用场景上。以下是它们的一些具体差异：

1. **归一化的数据范围**：
   - `BatchNorm`：归一化是在单个层的每个特征（channel/neuron）上，针对每个批次（batch）的数据进行的。它计算每个批次中所有样本在该特征上的均值和方差，并使用这些统计数据来归一化该特征的输出。
   - `LayerNorm`：归一化是在单个样本的整个层输出上进行的。它计算单个样本在该层所有特征上的均值和方差，并使用这些统计数据来归一化该样本的输出。

2. **计算方式**：
   - `BatchNorm`：在训练时，它会计算并存储每个批次的均值和方差，用于归一化；在评估时，使用训练期间计算的全局均值和方差。
   - `LayerNorm`：无论在训练还是评估模式下，它都只使用当前样本的统计数据来进行归一化，不需要存储和使用全局统计数据。

3. **适用场景**：
   - `BatchNorm`：由于它依赖于批次统计数据，因此更适合用于具有稳定批次大小的场合。它在卷积神经网络（CNN）和全连接层中非常流行，尤其是在图像处理任务中。
   - `LayerNorm`：由于它基于单个样本的统计数据，因此更适合用于序列数据或批次大小可能变化的场合。它在循环神经网络（RNN）和变换器（Transformer）模型中更为常见，尤其是在自然语言处理（NLP）任务中。

4. **对模型性能的影响**：
   - `BatchNorm`：可以减少梯度的方差，有助于加速训练过程，但也可能导致模型对批次统计数据的依赖，从而在小批次或评估时性能下降。
   - `LayerNorm`：由于它不依赖于批次统计数据，因此在处理不同长度的序列或小批次时可能更为稳定，但可能不如`BatchNorm`在大批次训练时有效。

5. **参数数量**：
   - `BatchNorm`：除了归一化的参数外，还有两个可学习的参数（缩放因子和偏移量），每个特征一个。
   - `LayerNorm`：也有两个可学习的参数（缩放因子和偏移量），但它们是针对整个层的，而不是每个特征。

总的来说，`BatchNorm`和`LayerNorm`在模型训练中的具体差异体现在它们处理数据的方式、计算归一化的统计数据以及适用的场景上。选择哪种归一化技术取决于具体的任务、模型架构和训练数据的特点。


 
#  （NLP）任务中为什么`LayerNorm` 比`BatchNorm` 更受欢迎 

1. **处理变长序列**：
   - NLP任务常常涉及处理不同长度的文本序列，例如句子或文档。`LayerNorm`对单个样本的所有激活进行归一化，这意味着它不依赖于批次的大小或序列的长度，因此更适合处理变长序列。

2. **小批次尺寸的稳定性**：
   - 在训练深度学习模型时，尤其是在使用GPU时，通常会使用小批次来提高内存利用率。`BatchNorm`依赖于批次统计数据，当批次尺寸较小时，这些统计数据可能不够代表性，导致归一化效果不稳定。相比之下，`LayerNorm`使用单个样本的统计数据，因此在小批次尺寸下也能保持稳定性。

3. **减少批次内依赖**：
   - `BatchNorm`通过规范化批次内的数据来减少内部协变量偏移，这可能导致模型对批次内数据的分布产生依赖。`LayerNorm`通过规范化单个样本的数据来减少这种依赖，有助于模型学习到更加泛化的特征表示。

4. **适应稀疏激活模式**：
   - 在NLP任务中，尤其是在使用注意力机制的模型（如Transformer）中，激活模式可能非常稀疏。`LayerNorm`能够更好地适应这种稀疏性，因为它不会受到批次内其他样本的影响。

5. **简化模型并行**：
   - 在模型并行训练中，`LayerNorm`由于其独立于批次的特性，可以更容易地在多个设备上并行处理，而不需要同步批次统计数据。

6. **改善梯度流动**：
   - `LayerNorm`有助于更均匀地分布梯度，这可以改善深层网络中的梯度流动问题，尤其是在深层的RNN或Transformer模型中。

7. **更好的处理序列依赖性**：
   - 在序列模型中，如RNN或Transformer，`LayerNorm`可以在不同时间步之间提供更一致的归一化，有助于模型捕捉长期依赖关系。

8. **灵活性**：
   - `LayerNorm`可以更容易地与其他归一化技术（如`GroupNorm`）结合使用，以适应不同的模型架构和训练策略。

由于这些原因，`LayerNorm`在NLP任务中变得更加流行，尤其是在需要处理序列数据、小批次尺寸或深层网络结构的场景中。然而，选择哪种归一化技术仍然取决于具体的任务、模型架构和训练数据的特点。




 # transformer 相关工程的数学估算
 
-  https://blog.eleuther.ai/transformer-math/
-  [The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)


## 计算量估计举例

### 例子
**问题**：GPT-3 的 82B 参数模型使用 1024 个 Nvidia A100 GPU 集群对 150B 个 token 进行训练。这需要多长时间？

**解决方案**：[A100 的峰值 float16 FLOPs 吞吐量](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf)为𝜏 = 312 teraFLOPs = 3.12e14 FLOPs。总计算量为*C* = 6 ∙ 8.2e10 ∙ 1.5e11 = 7.38e22。训练必须至少花费*T* = *C* / 1024𝜏 / 86400 = 2.67 天。

**答案验证**：根据[白皮书](https://arxiv.org/abs/2109.04650)，训练耗时 13.4 天。我们的估算有 5 倍偏差，但我们确实得到了正确的数量级。

错误是由于我们天真地插入了理论峰值吞吐量𝜏，而这是分布式训练无法实现的。另一个原因是，对于最大的模型来说，checkpointing 是必须的，所需的计算*C*上升到大约≈ 8ND。
### 解释
要理解为什么是 6ND ，需要理解下面三部分内容：

第一，我们有一个很重要的假设，权重 FLOP 是 Transformer FLOP 的绝大部分，这意味着我们可以将偏置向量添加、层归一化、残差连接、非线性、softmax 甚至注意力所需的 FLOP 放在一边，尽管他们也会占用一定的计算量，但是和主要的计算相比不在一个量级。

第二，要知道在每个 batch 中，每个权重**w**在前向和后向传递中恰好生成 6 个 FLOP ，详细过程如下，其实就是完整的前向传播、反向传播、梯度更新涉及到的 6 次操作：

- **FLOP 1 multiply**：将 i 的激活值 h(i) 乘以 w 送往 j 
- **FLOP 2 add**：将 i 所有维度送往 j 的值在 j 的每个维度上进行加法求和得到 a(j)
- **FLOP 3 multiply**：j 将传入的损失梯度 dL/da(j) 乘以 w 返还给 i
- **FLOP 4 add**：i 的每个维度将 j 返还的梯度进行累加得到 dL/dh(i) ，表示 h(i) 对损失 L 的梯度，可以供算法继续反向传播梯度
- **FLOP 5 multiply**：dL/da(j) 乘以激活值 h(i) 表示针对该样本的权重梯度 dL/dw
- **FLOP 6 add**：将所有样本根据第  5 步计算出来的梯度进行累加，用于更新梯度
 
第三， 不论一个 batch 中的序列有多长，Transformer 为每个输入的 token 执行1次权重矩阵乘法，因此每个 token 的总 FLOP 数等于模型参数量 N \* 6 。
 
### Weight FLOPs(WFLOPs)

- A100 GPU 处理不同精度，不同维度大小的操作，以及不同类型的操作，算力有明显的不同。
- GPT-2 吞吐量仅为 68 teraWFLOP/s。可能的解释是，内存密集型计算（例如残差连接、激活、层规范化、注意力掩蔽和注意力 softmax）结合在一起时确实会成本较高。

### 回到例子

**最新解决方案**：A100 的实际 teraWFLOP 吞吐量为𝜏 = 68 teraFLOPs = 6.8e13 FLOPs。总计算量为C = 8 ∙ 8.2e10 ∙ 1.5e11 = 7.38e22。训练必须至少花费T = C / 1024𝜏 / 86400 = 16.355 天，很接近真实的时间消耗。

 

## 算力要求


 

![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/237cdab3027d4382aff52d35a7c95e47~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=IhLGqPVAd6k6KlyTUpaIWm2kF%2Bc%3D)

 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/ce5bd209cc2e4c258b3674b6c4718a0a~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=b4oMhXkgOIahVSdZWOEIjK1gHxM%3D)

- 这些方程是在[OpenAI 的缩放定律论文](https://arxiv.org/abs/2001.08361)和[DeepMind 的缩放定律论文](https://arxiv.org/abs/2203.15556)中提出并通过实验验证的。
-  Actual FLOPs 是个重要概念。因为虽然 GPU 加速器白皮书通常会宣传其理论 FLOP，但这些在实践中从未达到过（特别是在分布式环境中！）




## 参数与数据集大小的关系

- 训练的 token 数量会极大地影响计算成本和最终模型性能
- 近似满足 D=20P 这一公式
- 不建议使用少于 200B 个 token 来训练 LLM ，但生成的模型通常很差

## 计算成本

- Transformer 的计算成本通常以 GPU 小时或 FLOP 秒为单位列出。
- GPT-NeoX （遵循 GPT-3 框架的一款开源的大型语言模型）在普通注意力机制下可达到 150 TFLOP/s/A100，在 Flash Attention 机制下可达到 180 TFLOP/s/A100。这与其他大规模高度优化的库一致
- 一般来说，您应该始终能够实现大约 120 TFLOP/s/A100。如果您看到低于 115 TFLOP/s/A100，则您的模型或硬件配置可能有问题。
- 借助 InfiniBand （是一种高速、低延迟、低CPU开销、高效且可扩展的服务器和存储互连技术，广泛应用于高性能计算和数据中心环境）等高质量互连，您可以在数据并行维度上实现线性或亚线性扩展，即增加数据并行度应该会几乎线性地增加整体吞吐量。测试 GPT-NeoX 库的图表如下。
 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/081e9115f9a147da94a0ae34ae44420c~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=tdzTDWjVXxY%2Btvhohg%2BCDXbNe84%3D)


## 推理所需内存

除了存储模型权重所需的内存外，实际前向传递期间还会产生少量额外开销。根据我们的经验，此开销不超过 20%，并且通常与确定适合您的 GPU 的最大模型无关。这个模型在推理时候所需的总内存为：

 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/ef0b41b0dc4248aca6284fc5ea54edb0~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=jCS41Fv6dizyio6O8gwkfhYM%2BPc%3D)

## 训练时内存需求计算

### 训练时候存储模型权重所需的内存

 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/fa30907fb6e0408a99060e82609bd1e9~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=nej1In9plzMMBh7YxjQninXwUE8%3D)

- 训练总是需要比推理更多的内存，通常要多得多！
- **大多数 Transformer 都以混合精度**进行训练，即 fp16 + fp32 或 bf16 + fp32。这减少了训练模型所需的内存量，也减少了运行推理所需的内存量。我们可以将语言模型从 fp32 转换为 fp16 甚至 int8，而不会对性能造成重大影响。
- 模型可以在纯 fp32 或 fp16 的精度下进行训练，所需内存如下：
 
 
 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/e097c4962e174a7ebb7f0ca4c502c503~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=G%2FiGShM5Xm0qulE%2BFlOAB04PVWw%3D)


- 训练领域经常使用混合精度训练，如 AMP 。混合精度要求将 fp16/bf16 和 fp32 版本的模型存储在内存中，

### 存储优化器状态所需内存

 
Adam 很神奇，但它的内存效率非常低，对于普通版 AdamW ，所需的内存如下：

 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/79736e67eb7b4c2390e7062fa61b49f6~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=55LFU%2FGKDehYLnFgeOZ7ctj0OHA%3D)

-  32 位精度的副本模型参数: 4 bytes/param
-   动量 Momentum: 4 bytes/param
-   方差 Variance: 4 bytes/param

对于其他优化器，存在内容或者精度的不同。

### 存储梯度所需内存


有多少模型权重就有多少梯度，梯度可以存储在 fp32 或 fp16 中，需要请注意，梯度数据类型通常与模型数据类型匹配。我们看到，对于 fp16 混合精度训练，它存储在 fp16 中，内存开销如下：


 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/a673f907d21d4f41ac0b027e4ec7ded2~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=X8c7zuiZHMdNt2tA2DHcd8IupRw%3D)


### 存储激活值所需内存

https://arxiv.org/pdf/2205.05198

- 论文中`激活`指的是在前向传播中创建的任何张量，并且在反向传播期间需要用于梯度计算的张量。
- 没有并行技术下只考虑每个Transformer层（block）
- 有并行技术的情况下，要分别考虑张量并行、序列并行、管道并行等情况下的激活值内存占用情况


![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/9300f89ef2e647fcb1c39f30f3299bf8~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=TUcvtV%2FL0mIYG3c%2FraqB%2FWH0kIg%3D)


- 训练 LLM 的 GPU 的瓶颈通常是显存，而不是 FLOP。
- activation recomputation/checkpointing 是一种非常流行的方法，它用降低的显存成本换取额外的计算。工作原理是重新计算某些层的激活值，而不是将它们存储在显存中。减少内存取决于我们选择性地清除哪些层， Megatron （Megatron是一个由NVIDIA研究团队开发的开源深度学习训练框架，专门用于训练超大型的Transformer语言模型。Megatron通过模型并行和数据并行技术，使得在多个GPU上高效训练具有数千亿参数的模型成为可能。）的选择性重新计算方案如下图所示。


 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/74a10e5a83cf435e910b963c0851ddff~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=VkKZTv9I4Sj1Sr52gEwr1THJnUM%3D)

- 存储 transorfmer 模型的激活值所需内存的基本方程，让我逐一解释这三个公式的含义：

1. **No Recomputation (无重计算)** ：在无重计算策略下，所有的激活值都需要存储，整个网络的前向传播过程会记录每层的激活值，以便反向传播时使用。因此，内存需求较大。
2. **Selective Recomputation (选择性重计算)** ：在选择性重计算的策略下，只有部分激活值会被存储，其他激活值在反向传播时被重计算。这减少了部分存储需求，但会增加计算负担。 
3. **Full Recomputation (全重计算)** ：在全重计算策略下，所有激活值在反向传播时都被重新计算，从而可以在前向传播时不存储任何激活值。这大幅降低了内存需求，但增加了计算负担。此时的内存需求仅为前向传播的计算结果加上反向传播的结果，因而公式非常简洁，仅包含了模型规模 `sbhL` 的两倍。
  
 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/70bba9b5631b4dd497cda2f24b493d6c~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=fcABBNRcJ5%2FcaoJBiytFCcWSNJc%3D)

参数解释如下：

 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/32a46f413826439d873f20e92fdff691~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=mGVOzN570FbTpjk%2FA9ddLTSjzgE%3D)

### 总共的训练内存

 
`Total Memory Training = Model Memory + Optimiser Memory + Activation Memory + Gradient Memory`


 
 # 大模型训练优化
 
- https://blog.csdn.net/qq_36426650/article/details/130764843
- https://blog.csdn.net/Shirelle_/article/details/137869167
- https://blog.csdn.net/qq_36426650/article/details/130764843
- https://blog.csdn.net/cjnewstar111/article/details/128593120
- https://huggingface.co/docs/transformers/v4.27.2/en/perf_train_gpu_one#anatomy-of-models-operations
- https://huggingface.co/docs/transformers/v4.27.2/en/perf_train_gpu_many 

下面介绍的方法可以在单个 GPU 或者多个 GPU 上高效训练大型模型，有的可以通用，可以根据需求自行组合。

![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/bba82b425fbe43e086c2b97bed170c90~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=H5E%2FhKKU2uQqWCGHiZ3oM%2BTjkaU%3D)


 ## 分布式训练


数据并行、模型并行和ZeRO优化等技术是大模型训练中有效的优化策略。在实际应用中，建议根据模型大小、硬件资源和训练需求等因素，灵活选择和组合这些策略，可以显著提升训练效率。
 


### 数据并行 DP


Data Parallelism（数据并行）是最常用的并行方式之一，它的核心思想是将相同的模型复制到多个设备（如 GPU）上，每个设备处理不同部分的数据，并在设备间同步更新模型参数。

- **机制**: 
  1. 将输入数据划分为多个小批次，并分配到多个设备上。
  2. 每个设备上都有一个完整的模型副本，它们各自计算自己分配到的数据批次的损失和梯度。
  3. 在每个设备完成前向和反向计算后，将各设备计算出的梯度同步到一个主节点，主节点将这些梯度求平均，并更新所有设备上的模型参数。

- **常见使用场景**: 
  - 数据并行通常用于具有大批量输入数据的任务，例如图像分类、自然语言处理等，它可以有效利用多个 GPU 来加速训练。
  
- **优点**: 
  - 简单易实现，扩展性强。
  - 每个 GPU 只需计算一部分数据，因此可以充分利用大批量数据来提高训练速度。

- **缺点**: 
  - 模型需要在每个 GPU 上完整复制，因此对于非常大的模型，单个 GPU 内存可能无法容纳整个模型。
  - 设备之间需要频繁同步参数，通信开销较大。

#### DP（Data Parallelism） 

数据并行适用于单机多卡训练。在单个机器上安装多个GPU，每个GPU运行模型的一个副本，并通过PCIe或Nvlink等高速通信接口进行数据交换。[PyTorch](https://cloud.baidu.com/product/wenxinworkshop)中的`torch.nn.DataParallel`即为此类实现。
#### DDP（Distributed Data Parallelism）

分布式数据并行适用于多机多卡训练。在多个机器上部署GPU，通过[网络](https://cloud.baidu.com/product/et.html)进行通信，实现更大规模的并行训练。PyTorch中的`torch.nn.DistributedDataParallel`支持这一模式，并采用Ring-AllReduce算法优化通信效率。

优缺点：
-   **优点**：实现简单，易于理解；可以充分利用硬件资源，加速训练过程。
-   **缺点**：随着GPU数量的增加，通信开销也会增大，可能成为训练速度的瓶颈。

#### ZeRO DP

- https://www.cvmart.net/community/detail/7638
 

[优化器的巨大内存开销是ZeRO](https://arxiv.org/abs/1910.02054)和[FSDP](https://engineering.fb.com/2021/07/15/open-source/fsdp/)等  Sharded Optimizers 出现的原因。此类策略可将优化器开销降低，这就是为什么给定的模型配置可能适合大规模，但在小规模下会 OOM 。

ZeRO（Zero Redundancy Optimizer）是一种旨在减少内存使用并加速大规模模型训练的技术。它通过跨多个计算设备分散存储和计算模型的状态（如梯度、参数和优化器状态），从而减少每个设备上的冗余数据。

 
优缺点：

-   **优点**：显著减少内存占用，支持训练更大规模的模型；通过优化通信机制，提高训练速度。
-   **缺点**：实现复杂，需要深入理解模型结构和并行计算原理；对通信带宽和延迟要求较高。


 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/0e3be7be539b4a769cfbbc9a24e965dd~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=%2F433iUxrsu0A9IhUfNNldib7GR8%3D)
  
-  `Baseline`:传统的数据并行策略，每张GPU上存储全部模型的权重、梯度和优化器等参数，每张卡上并行训练不同的数据，并实现参数汇聚更新。该情况下，每张卡依然要加载120G参数，显然是无法在一般机器上实现的:
-  `ZeRo stage1--优化器并行`:在训练过程中，优化器状态参数占用的显存空间是非常大的，因此将优化器状态参数分发到不同的GPU上，此时单张卡上的显存占用会大大降低，假设使用混合精度和 Adam 优化器，内存需要：`Total Memory Training = Model Memory + Optimiser Memory/(No. GPUs) + Activation Memory + Gradient Memory`
- `ZeRo stage2--梯度+优化器并行`:在ZeR0 Stage1的基础上，额外对梯度进行了分布式存储，可以发现120G的显存占用直接降低到16G;假设使用混合精度和 Adam 优化器，内存需要：`Total Memory Training =   Activation Memory + Model Memory + (Optimiser Memory+Gradient Memory)/(No. GPUs)  `
- `ZeRo stage3--权重+梯度+优化器并行`:模型的所有参数都进行分布式存储，此时一张卡上只有1.9G占用。假设使用混合精度和 Adam 优化器，内存需要：`Total Memory Training = Activation Memory + (Model Memory + Optimiser Memory + Gradient Memory)/(No. GPUs)  `
 
 
 
请注意，ZeRO-3 引入了一组实时参数（***stage3_max_live_parameters、stage3_max_reuse_distance、stage3_prefetch_bucket_size、stage3_param_persistence_threshold***），用于控制 GPU 内存中一次有多少个参数（值越大，占用的内存越多，但需要的通信越少）。这些参数会对总 GPU 内存产生重大影响。

#### 用 ZeRO 机制如何计算节省的训练时间



使用 **DeepSpeed** 中的 **ZeRO (Zero Redundancy Optimizer)** 技术进行大模型训练时，时间的节省主要来源于以下几个方面：

1. **显存效率提升**：
   
   通过减少显存占用，更多的显存可以用于增大 batch size，从而降低梯度更新的频率，这可以显著减少训练时间。

2. **更大的批次训练**：
   - 由于 ZeRO 节省了显存，允许训练更大的模型或使用更大的 batch size。这意味着在一次训练迭代中可以处理更多的数据，从而减少了总的训练时间。
   - 较大的 batch size 通常也会减少训练过程中同步通信的次数，提升并行计算的效率。

3. **通信优化**：
   - ZeRO 通过减少跨 GPU 的通信量，降低了通信带来的开销，尤其在大规模集群上表现明显。这是因为 ZeRO 分片了优化器状态和梯度，不再需要每次进行完整状态的同步。



计算公式，假设：
- $ T_{base} $为没有使用 ZeRO 时训练的总时间。
- $ f_{bs}  $为 batch size 的倍增因子（即通过 ZeRO 提升的显存效率所能增加的 batch size 倍数）。
- $f_{comm}  $为通信开销减少的比例（例如，通信时间减少了 50%）。

训练时间节省可以通过以下公式粗略估算：



$$
T_{new} = \frac{T_{base}}{f_{bs}} \times f_{comm}
$$

    

示例：假设原始训练时间为 100 小时 ，通过 ZeRO 提升显存利用率使 batch size 增加 2 倍 ，同时通信开销减少 50% 。则新的训练时间为：

 
$$
T_{new} = \frac{100}{2} \times 0.5 = 25 \text{ 小时}
$$
在此情况下，训练时间节省了 75 小时。
 

####   ZeRO VS 模型并行


-  模型并行，是指在forward和backward的过程中，我只需要用自己维护的那块W来计算就行。即**同样的输入X，每块GPU上各算模型的一部分，最后通过某些方式聚合结果**。
-  ZeRO是模型并行的形式，数据并行的实质。
-  对ZeRO来说，它做forward和backward的时候，是需要把各GPU上维护的W聚合起来的，即本质上还是用完整的W进行计算。**它是不同的输入X，完整的参数W，最终再做聚合**。


#### ZeRO-Offload与ZeRO-Infinity


它的核心思想是：显存不够，内存来凑。ZeRO-Offload的做法是：
-   forward和backward计算量高，因此和它们相关的部分，例如参数W（fp16），activation，就全放入GPU。
-   update的部分计算量低，因此和它相关的部分，全部放入CPU中。例如W(fp32)，optimizer states（fp32）和gradients(fp16)等。

![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/37b3f5dc951e4f2a8204def0e062659b~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=0N9U3fhxWK4f0gGBvyAGk%2B6%2Fh%2FY%3D)
ZeRO-infinity也是同理，它们在解决的事情都是：找个除GPU之外的地方，存数据。

### Tensor Parallelism（张量并行,TP）


**Tensor Parallelism（张量并行）**是将模型的权重或张量切分后，在多个设备上并行进行计算。这种方法的重点是对模型的内部结构进行拆分，以减少每个设备的内存压力。

- **机制**: 
  1. 模型的权重或参数张量被切分成多个部分，每个设备只负责计算一部分张量。
  2. 各个设备并行计算它们负责的张量部分，计算结果在设备之间相互传递，最终组合成完整的输出。

- **常见使用场景**: 
  - 当单个神经网络层非常庞大，无法放入单个设备的显存时，可以通过张量并行将层的计算分散到多个设备上进行。
  
- **优点**: 
  - 能处理特别大的模型，尤其是那些具有大规模层参数的网络（如 GPT、BERT 等）。
  - 减少单个设备的显存需求。

- **缺点**: 
  - 因为需要跨设备传递中间计算结果，存在较高的通信开销。
  - 实现复杂度较高，适用于计算密集型的任务。

###   **Pipeline Parallelism (流水线并行，PP)**

**Pipeline Parallelism（流水线并行）**是将模型的不同层分配到不同的设备上。每个设备负责处理模型的某一部分层，数据像流水线一样，逐层通过这些设备进行处理。

- **机制**: 
  1. 模型被拆分为多个部分（按层次划分），并分配给不同的设备。
  2. 输入数据依次通过每个设备上的模型部分进行处理（类似流水线），每个设备处理完它的部分后，将结果传递给下一个设备。
  
- **常见使用场景**: 
  - 当模型特别深（如深层的 Transformer 模型），并且整个模型无法放入单个设备时，可以通过流水线并行将模型分层分配到不同设备。
  
- **优点**: 
  - 能处理超大规模的深层模型，尤其是层数较多、每层计算复杂的网络。
  - 降低每个设备的显存负担，因为每个设备只需存储部分模型层。

- **缺点**: 
  - 因为数据是串行通过设备的，可能存在空闲的等待周期，设备的计算资源可能未能充分利用。
  - 需要跨设备传递数据，存在通信延迟。



---

### **三种并行策略的对比**

| 特性                 | Data Parallelism (DP)                  | Tensor Parallelism (TP)                    | Pipeline Parallelism (PP)                 |
|----------------------|----------------------------------------|--------------------------------------------|------------------------------------------|
| **并行维度**         | 数据（输入数据划分给多个设备）           | 模型的张量（权重或计算）切分                | 模型的层次划分（每个设备计算不同层）     |
| **模型副本**         | 每个设备都有完整模型副本                | 每个设备只计算部分权重/张量                | 每个设备只计算部分模型层                 |
| **数据流动**         | 数据并行处理在每个设备上进行             | 张量在设备间流动（中间结果交换）            | 数据像流水线一样依次经过不同设备         |
| **通信需求**         | 同步所有设备的梯度                     | 各设备间传递中间计算结果，通信频繁          | 设备间传递数据，通信较少，但延迟高       |
| **内存需求**         | 每个设备需要完整模型副本，显存压力较大   | 每个设备只需存储一部分权重，降低显存压力    | 每个设备只需存储部分层，降低显存压力     |
| **适用场景**         | 适合大批量数据训练                      | 适合处理巨型模型中的单层                   | 适合深层神经网络，模型特别大的情况       |
| **计算效率**         | 计算资源利用率高，但有频繁通信           | 减少了每个设备的计算量，但通信开销较大      | 可能存在等待周期，设备利用率不高         |
| **扩展性**           | 扩展性较好，易于实现                    | 需要复杂的设备间通信，扩展性较差            | 需要精确划分模型层，扩展性复杂         

- **Data Parallelism (DP)**：在拥有大数据批量训练时效果好，简单且易扩展，但主要开销是在于梯度的计算和同步，同时对于超大模型，模型副本带来的显存限制是个挑战。
- **Tensor Parallelism (TP)**：适合单层参数非常大的模型，将计算负载分散到多个设备上减少计算量，但通信成本增加，适合计算密集型任务。
- **Pipeline Parallelism (PP)**：适合特别深的模型，模型每层的计算在不同设备上串行进行，但每个设备必须等待前一个设备的数据，因此在深层网络中可能会导致设备空闲周期，降低效率，需要通过批次流水线并行化来优化。

### DP+PP

由于每个维度至少需要 2 个 GPU，因此这里至少需要 4 个 GPU。

![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/18578efe65154d8db0c569e755c2e262~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=zlAuYXsqSAH7OBthWZIF%2BGpmukI%3D)

### 3D 并行， DP+PP+TP

为了实现更高效的训练，我们使用了 3D 并行，将 PP 与 TP 和 DP 相结合。如下图所示。由于每个维度至少需要 2 个 GPU，因此这里至少需要 8 个 GPU。


![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/26c4c30443ac4d1f992312353bb3a1ae~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=NetZnR%2B2ozJZJHyP7o1I%2BR0zWEg%3D)

### ZeRO DP + PP + TP

DeepSpeed 的主要功能之一是 ZeRO，它是 DP 的超扩展。通常它是一个独立的功能，不需要 PP 或 TP。但它仍然可以与 PP 和 TP 结合使用。

- 当 ZeRO-DP 与 PP（和可选的 TP）结合时，它通常只启用 ZeRO 第 1 阶段（优化器分片）。
- 虽然理论上可以将 ZeRO 第 2 阶段（梯度分片）与 PP 结合使用，但这会对性能产生不利影响。
- 出于同样的原因，ZeRO 第 3 阶段也不是一个好的选择——需要更多的节点间通信。



**Data Parallelism (DP)**、**Tensor Parallelism (TP)** 和 **Pipeline Parallelism (PP)** 是在分布式深度学习中常用的三种并行策略。它们的主要目的是为了加速大规模神经网络的训练，并解决模型规模过大、单台设备内存无法承受等问题。它们各自并行的维度不同，因此在具体的应用场景和实现细节上也有明显的区别。



结合这三种方式，可以根据任务需求进行选择或混合使用，以提高计算效率和降低内存需求。



## 混合精度训练
 
混合精度训练是一种在深度学习中提高训练速度和减少内存占用的技术。在 PyTorch 中，通过使用半精度浮点数 FP16 和单精度浮点数 FP32 的组合。简单的讲就是使用fp16进行乘法和存储，只使用fp32进行加法操作，避免累加误差;

### FP16 和 FP32

 
![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/04e410a41269456587e274f49a1c6f7b~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=Vo%2ByhnpFBgOLPA9z74dhsUtVClw%3D)

FP16（16位浮点数）：

- FP16 是一种半精度浮点数格式，它使用16位（2字节）来表示一个浮点数。
- 它的格式通常包括1位符号位、5位指数位和10位尾数位存储小数。
- 由于指数位较少，FP16能够表示的数值范围比FP32小，但它需要的内存和计算资源也更少。
- FP16在深度学习中被用于加速计算和节省内存，尤其是在支持FP16运算的硬件上。


FP32（32位浮点数）：

- FP32 是一种单精度浮点数格式，它使用32位（4字节）来表示一个浮点数。
- 它的格式包括1位符号位、8位指数位和23位尾数位存储小数。
- 相比于FP16，FP32能够表示更大范围的数值，具有更高的精度，但也需要更多的内存和计算资源。
- FP32是最常用的浮点数类型，适用于广泛的科学计算和工程应用。

### 典型代表 AMP
Automatic Mixed Precision（自动混合精度，简称AMP）是一种在深度学习训练中使用的技术，它结合了单精度（FP32）和半精度（FP16）来加速训练过程，同时减少内存使用并提高计算效率。应用非常广泛。在PyTorch 1.6及以上版本中，通过`torch.cuda.amp`模块可以很容易地实现AMP，而不需要依赖于第三方库如NVIDIA的Apex。
 

硬件要求：

为了充分利用AMP的优势，需要支持FP16计算的硬件，如NVIDIA的Volta及以后架构的GPU，这些GPU配备了Tensor Cores，可以高效执行FP16运算。



### 优点
- **减少内存占用**：由于FP16的数据类型占用的空间是FP32的一半，使用FP16可以减少模型参数和中间激活值的内存占用，允许更大的模型或更大的批次大小。
- **加快计算速度**：在支持Tensor Cores的NVIDIA GPU上，FP16的矩阵运算可以显著加速，因为这些核心专门为低精度运算优化。
- **避免精度损失**：尽管FP16的动态范围比FP32窄，可能导致溢出或舍入误差，但通过混合精度训练，即在需要时使用FP32（例如在权重更新和归一化层），可以在保持精度的同时加速训练。
- **主流框架支持自动调整精度**：使用AMP时，深度学习框架（如PyTorch）可以自动决定在何处使用FP16和FP32，以优化性能。
 

### 例子

下面是一个例子，`autocast()`自动应用合适的精度到不同的操作。如果特定操作的前向传递有`float16`输入，则该操作的反向传播也将产生`float16`梯度。由于FP16的数值范围较小，可能会导致梯度下溢（underflow），也就是可能表示为 0 。`GradScaler` 通过在反向传播之前将损失乘以缩放因子自动放大损失来计算梯度。然后在优化器更新参数之前，每个参数的梯度除以相同的缩放因子回到原来的大小，从而避免了数值下溢的问题。 

在混合精度训练中，保存模型时通常将权重转换回 FP32 格式，即使训练时时 FP16。这是因为 FP32 提供了更高的数值精度和更广泛的硬件支持，这使得模型在不同环境中的兼容性和可靠性更好。同理，在加载模型时候也会自动转换为 FP32 ，即使模型是 FP16 。


```
   model = SimpleMLP().cuda()
   model.train()
   scaler = GradScaler()
   for epoch in range(num_epochs):
       for batch in data_loader:
           x, y = batch
           x, y = x.cuda(), y.cuda()
           with autocast():
               outputs = model(x)
               loss = criterion(outputs, y)
           # 反向传播和权重更新
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update() 
 ```

 ### 缺点


- **数值稳定性问题**：使用FP16可能会导致数值下溢，即非常小的数值在FP16格式中无法有效表示变成零。同时由于FP16的精度较低，可能会在训练过程中引入舍入误差，影响模型的收敛和最终性能。
- **硬件兼容性**：在不完全支持 FP16 的 GPU 上运行可能不会带来预期的性能提升，甚至无法在这些硬件上使用。
- **软件和库的支持**：一些深度学习框架和库可能没有完全支持混合精度训练，或者对FP16的支持不够成熟。
- **模型和数据类型的转换**：在混合精度训练中，需要在FP32和FP16之间转换数据类型，这可能需要仔细管理以避免精度损失。
- **调试和分析困难**：需要跟踪哪些操作是在FP16下执行的，哪些是在FP32下执行的。
- **模型泛化能力**：混合精度训练可能会影响高精度对精度敏感的模型的泛化能力

## 改变优化器

用于训练 Transformer 模型的最常见优化器是 Adam 或 AdamW（带权重衰减的 Adam）。Adam 通过存储先前梯度的滚动平均值实现了良好的收敛，但这会增加与模型参数数量相当的内存占用。一种补救措施是使用替代优化器（例如 Adafactor），它对某些模型效果很好，但通常存在不稳定问题。

## 量化

## 梯度累积
- https://huggingface.co/docs/transformers/v4.27.2/en/perf_train_gpu_one#gradient-accumulation

- 对于模型训练时，扩大Batch size是提高模型训练效果的重要因素，降低Batch size可能会降低模型的效果。为了不降低 Batch size ，可以采用梯度累积的方法。梯度累积是指在前向传播之后所计算梯度并不立刻用于参数更新，而是接着继续下一轮的前向传播，每次计算的梯度会暂时存储下来，待在若干次前向传播之后，一并对所有梯度进行参数更新。因此梯度累积相当于是拿时间换空间。
- 可以直接添加参数 per_device_train_batch_size 和 gradient_accumulation_steps 来实现梯度累计，这两个参数的组合使用，可以在有限的硬件资源下，有效地模拟更大的批次大小，从而可能提高模型训练的稳定性和性能。

```
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
```
原始结果是：

```
Time: 57.82
Samples/second: 8.86
GPU memory occupied: 14949 MB.
```
改进后结果是：
```
Time: 66.03
Samples/second: 7.75
GPU memory occupied: 8681 MB.
```

我们可以发现内存占用量大幅减少，但代价是运行速度仅略慢。




## 梯度检查点（Gradient Checkpointing）
- https://blog.csdn.net/Solo95/article/details/131606918

大模型的参数量巨大，即使将 batch_size=1 并使用梯度累积的方式更新，也可能会 OOM 。原因是通常在计算梯度时，我们需要将所有前向传播时的激活值保存下来，这消耗大量显存。还有另外一种延迟计算的思路，丢掉前向传播时的激活值，在计算梯度时需要哪部分的激活值就重新计算哪部分的激活值，这样做倒是解决了显存不足的问题，但加大了计算量同时也拖慢了训练。

在上述两种方式之间取了一个平衡，梯度检查点这种方法采用了一种策略选择了计算图上的一部分激活值保存下来，其余部分丢弃，这样被丢弃的那一部分激活值需要在计算梯度时重新计算。

一种简单策略：前向传播过程中计算节点的激活值并保存，计算下一个节点完成后丢弃中间节点的激活值，反向传播时如果有保存下来的梯度就直接使用，如果没有就使用保存下来的前一个节点的梯度重新计算当前节点的梯度再使用。 

使用很简单，只需要开启参数即可 gradient_checkpointing=True 即可。

```
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args) 
trainer = Trainer(model=model, args=training_args, train_dataset=ds) result = trainer.train()
```
原始结果是：

```
Time: 57.82
Samples/second: 8.86
GPU memory occupied: 14949 MB.
```
开启参数 gradient_checkpointing=True 后，这节省了一些内存，但同时训练变得有点慢。一般经验法则是梯度检查点会使训练速度减慢约 `20%` 。另一种可以恢复速度的方法：混合精度训练。
```
Time: 85.47
Samples/second: 5.99
GPU memory occupied: 6775 MB.
```

## flash-attention 算法优化

## peft 参数有效性学习
参数有效性学习(Parameter-Eficient Leamning，PEL)旨在训练过程中指定少量参数参与梯度的计算和更新，从而在梯度和优化器参数上降低显存占用。

参数有效性学习有很多经典的方法，比如 Adapter-tuning、Prefx-tuning、P-tuning、LoRA、BitFit 等。本部分主要介绍 LORA 方法，因为在很多类 ChatGPT 的训练中都采用 LORA 进行参数有效性训练。

### LORA

### QLORA

## 混合专家 (MoE)训练

- https://huggingface.co/blog/zh/moe
- https://huggingface.co/docs/transformers/v4.27.2/en/perf_train_gpu_one#mixture-of-experts




- 混合专家：对于一个决策问题，交给众多专家进行决策投票，根据投票的结果来进行加权求和实现最终决策。在预训练中，则采用了这种思想。
- MoE 层由两个核心部分组成: 一个门控网络（用于决定哪些 token 被发送到哪个专家）和若干数量的专家（其实最常见的还是以 FFN  网络形式出现，也可以用更复杂的网络）。下图中展示了 MoE 的单层结构，router 负责决定给每个 expert 的权重，并指定权重最高的 expert 作为当前数据进行前后向传播的路由。例如上图中的 FFN 有4个，，“More” 被发送到 FFN2 ，而“Parameters” 被发送到 FFN1 ，此时只会对 FFN2 和 FFN1 进行参数更新，而其余的参数则固定不变，
- token 的路由方式是 MoE 使用中的一个关键点，因为路由器由可学习参数组成，并且与网络的其他部分一同进行预训练。

![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/b763904385f1482fbedc13b0b5f6f080~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=DINCM74iqceN%2FBUfXu9GuRlSCfc%3D)

- 将混合专家 (MoE) 集成到 Transformer 模型中，训练速度提高了 4-5 倍，推理速度也比同参数量级模型更快，因为 MoE 模型虽然可能拥有大量参数，但在推理过程中只使用其中的一部分，这使得它们的推理速度快于具有相同数量参数的稠密模型。
- 参数数量增加一个数量级，而不会增加训练成本
- 每个其他 FFN 层都被由许多专家组成的 MoE 层所取代，并具有一个门控函数，可根据输入标记在序列中的位置以平衡的方式训练每个专家。


![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/f9c172b8a6c64738b603c1237a175db3~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=Z5RKxj0%2FT%2BAYBe5ZQ%2BUaeRaHY1c%3D)

 
因此可以发现，MOE是一种变相的参数有效性训练方法，只不过不同于LRA等方法，MOE所引入的参数只是控制路由的，且在推理阶段不再使用router，因此对具体的模型推理能力并不起作用。



缺点：
- 主要缺点是它需要几乎比其他方法大一个数量级的 GPU 内存。尽管同时提出了各种提炼和方法来克服更高的内存要求。直接简答的权衡，只使用几个专家和一个 2-3 倍小的基础模型，从而适度提高训练速度，同时也适度增加内存需求。
- 微调 MOE 模型很容易过拟合，缓解办法有：更强的正则；仅冻结 MoE 层的参数，结果显示这种方法几乎与更新所有参数的效果相当；稀疏模型往往更适合使用较小的批量大小和较高的学习率，这样可以获得更好的训练效果。

## Accelerate

- https://huggingface.co/docs/accelerate/index

使用 🤗 Accelerate，您可以关注于使用 PyTorch 完全控制训练循环， 只需进行一些小修改。并且允许轻松地跨不同的基础设施（例如 CPU、GPU、TPU 或分布式多 GPU 设置）进行扩展，而无需更改任何代码。


  

# 大模型的推理加速

- https://www.cnblogs.com/sasasatori/p/18337693
- https://www.cvmart.net/community/detail/7638


模型并发推理设置为 n ，通常意味着系统能够同时处理 n 个推理请求，而不是将模型复制 n 份。在实际应用中，这可以通过以下几种方式实现：

## 多线程或多进程

在CPU或GPU上使用多线程或多进程技术，可以同时执行多个推理任务。每个线程或进程可以独立地处理一个推理请求，从而实现并发。

##

## Prefill & Decode
- Prefill，即预填充（大模型计算并存储原始输入token的KV Cache，并生成第一个输出token）
- Decode，即解码（大模型利用KV Cache逐个输出token，并用新生成的token的K，V（键-值）对进行KV Cache更新）


![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/7b5295648c7c47a09c21bd6476f0b2bc~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=lueFwpEWBHWetMppJWPdIiUnK6o%3D)
## KV-cache

由于大模型本身的自回归结构，在每次计算一个新token时，都需要用到前文的Key和Value向量，但由于token是一个一个生成的，所以实际上我们并不需要每一次都计算所有token的向量，而是可以对已经生成好的Key和Value做复用。这样可以减少大量的计算开销。

这种缓存要复用的Key和Value值的技术就是KV Cache，KV Cache仅在每次算一个新的token时做更新（当然qq因为本身就不涉及记忆，所以也不存在Q Cache的概念）。


![image.png](https://p9-juejin-sign.byteimg.com/tos-cn-i-k3u1fbpfcp/cd653595d020403fafaffb92b8330e87~tplv-k3u1fbpfcp-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6:q75.awebp?rk3s=bb34011d&x-expires=1730958302&x-signature=JsxGhkqo1sOzoW8GyqsL8XJryEg%3D)
## 并行推理

并行推理用于实现在多个设备节点上并行运行大模型推理以提供加速，现在可以分类为三种经典方法：数据并行[[3]](https://www.cnblogs.com/sasasatori/p/18337693#fn3)，模型并行[[4]](https://www.cnblogs.com/sasasatori/p/18337693#fn4)和流水线并行[[5]](https://www.cnblogs.com/sasasatori/p/18337693#fn5)。

**数据并行**方法的思路是:将大的数据集切分为小的batch，分散到独立运行模型的多个节点上，再将结果汇总。其特点是结构简单，每个GPU上的模型实例可以是相同的，但是限制了模型的参数量不能够超过单个节点的容量上限。
 
**模型并行**方法的思路是:对于一个大模型，将其中无法容纳在单个节点上的大型模型层或模块分散到多个GPU上运行，每个GPU负责模型的一部分，其特点是在用户看来仍然在使用完整的模型，可以显著减少单个节点的容量/计算需求，但实现上相对复杂。

**流水线并行**方法的思路:是将同一个模型的不同层次分散到不同的节点，以流水线的形式进行并行工作，这样可以在高效通信的同时处理大规模模型，但是流水级之间的计算时间和通信时间需要精确控制。

在实际应用中三种并行方法往往相互灵活组合。

## 模型优化

-   **有效结构设计**：设计高效的模型结构，如优化前馈网络(FFN)和注意力机制，以减少模型参数和计算复杂度 。
-   **模型压缩**：通过模型量化、剪枝、低秩分解等技术减少模型大小，以减少内存占用和加速推理过程 。
-   **Flash Attention技术**：这是一种优化自注意力机制的方法，通过减少中间结果的存储，将内存复杂度从O(n²)降低到O(logn)或O(n)，从而减少内存占用并提高并发量

## 硬件资源的充分利用

## 先进的成熟框架

vLLM、DeepSpeed-MII、TensorRT-LLM 等

