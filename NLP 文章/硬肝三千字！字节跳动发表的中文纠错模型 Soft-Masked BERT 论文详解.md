
本文主要介绍的是由 ByteDance AI Lab 发表的一篇关于中文纠错模型 Soft-Masked BERT 的论文，收录于 2020 年 ACL 。论文地址：https://arxiv.org/pdf/2005.07421.pdf

## 中文拼写错误纠正 (CSC) 

给定 n 个字符的序列 X = (x<sub>1</sub>, x<sub>2</sub>, · · · , x<sub>n</sub>)，目标是将其转换为另一个相同长度的字符序列 Y =(y<sub>1</sub>,y<sub>2</sub>,...,y<sub>n</sub>)，其中 X 不正确的字符被替换为正确的字符以获得 Y 。该任务可以看作是一个序列标记问题，其中模型是一个映射函数 f : X → Y 。然而这项任务相对来说更容易，因为通常只需要替换几个字符即可。

## 摘要

* 提出了一种新的神经架构来解决中文拼写错误纠正（CSC），被称为 Soft-Masked BERT ，它由一个基于 Bi-GRU 的错误检测网络和一个基于 BERT 的错误纠正网络组成，错误检测网络用于预测字符在每个位置出错的概率，然后利用该概率对该位置的字符嵌入进行 soft-masking 操作，然后将每个位置的 soft-masking 嵌入输入到纠正网络中。
* 本文提出的方法在端到端联合训练期间，可以使模型在检测网络的帮助下学习正确的上下文以进行纠错
* soft-masking 是传统 “hard-masking” 的扩展，当错误概率等于 1 时，前者退化为后者
* 方法的性能明显优于基于 BERT 在内的其他模型


## 贡献

* 针对 CSC 问题提出新的神经架构 Soft-Masked BERT
* Soft-Masked BERT 有效性的验证
* 使用 SIGHAN 和 News Title 两份数据集，Soft-Masked BERT 在准确度测量方面明显优于两个数据集上的其他比较模型

## 模型框架

### Problem and Motivation

CSC 最先进的方法是基于 BERT 来完成任务。现在本文发现，如果能找出更多错误的字符，则可以进一步提高该方法的性能。一般来说，基于 BERT 的方法倾向于不进行更正，或只是复制原始字符。可能的解释是，在 BERT 的预训练中，只有 15% 的字符被屏蔽以进行预测，导致模型的学习不具备足够的错误检测能力。这促使我们设计一个新模型。


### Model

 ![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f8548d9c6ac343c5b2c823df7079456b~tplv-k3u1fbpfcp-zoom-1.image)

本文提出了一种称为 Soft-Masked BERT 的新型神经网络模型，如图所示。Soft-Masked BERT 由基于 Bi-GRU 的检测网络和基于 BERT 的校正网络组成。检测网络预测错误的概率，纠正网络预测错误纠正的概率，而前者使用 soft masking 将其预测结果传递给后者。更具体地说：

* 我们的方法首先为输入句子中的每个字符创建一个嵌入，称为输入嵌入。
* 将嵌入序列作为输入，并使用检测网络输出字符序列的错误概率。
* 以错误概率为权重，它计算  input embedding 和 [MASK] 嵌入的加权和。计算出的嵌入以一种软方式掩盖了序列中可能的错误。
* 然后将软掩码嵌入序列作为输入，并使用校正网络输出纠错概率，这是一个 BERT 模型
* 最后一层由所有字符的 softmax 函数组成，输入嵌入和最后一层的嵌入之间也存在残差连接。

### Detection Network


检测网络是一个二分类模型，由经典的 Bi-GRU 实现。公式如下：
![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/324bf96ad4c248e4810948e5462cecce~tplv-k3u1fbpfcp-zoom-1.image)

输入是嵌入序列 E = (e<sub>1</sub>, e<sub>2</sub>, · · · , e<sub>n</sub>)，其中 e<sub>i</sub> 表示字符 x<sub>i</sub> 的embedding ，是 word embedding 、position embedding 和 segment embedding 的总和。输出是一个标签序列 G = (g<sub>1</sub>, g<sub>2</sub> , ... , g<sub>n</sub>)，其中 g<sub>i</sub> 表示第 i 个字符的标签，1 表示字符不正确，0 表示正确。对于每个字符，都有一个概率 p<sub>i</sub> 表示其为 1 的可能性。p<sub>i</sub> 越高，该字符错误的可能性越大。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/833f5af0d8c847dbb1aa9972c2e4d7b3~tplv-k3u1fbpfcp-zoom-1.image)

软掩码相当于以错误概率作为权重，输入嵌入和的掩码嵌入的加权总和。如果出错概率高，则 soft-masked embedding e′<sub>i</sub> 接近 mask embedding e<sub>mask</sub> ；否则它接近输入嵌入 e<sub>i</sub>  。第 i 个字符的 soft-masked embedding  e′<sub>i</sub> ,公式如下：

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ef0a42a6dfd743eb89cb307cdb361af6~tplv-k3u1fbpfcp-zoom-1.image)


其中 e<sub>i</sub> 是输入嵌入，e<sub>mask</sub> 是掩码嵌入。

### Correction Network


校正网络是基于 BERT 的顺序多分类模型。输入是 soft-masked 嵌入序列 E' = (e'<sub>1</sub>, e'<sub>2</sub>, · · · , e'<sub>n</sub>) ，输出是字符序列 Y =(y<sub>1</sub>, y<sub>2</sub>, ... , y<sub>n</sub>) 。

BERT 由 12 个相同块组成，将整个序列作为输入。每个块包含一个多头自注意力操作，然后跟一个前馈神经网络。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/df23110ae85d4db9a1e6b53e735476d7~tplv-k3u1fbpfcp-zoom-1.image)

将 BERT 最后一层的隐状态序列表示为 H<sub>c</sub> = (h<sup>c</sup><sub>1</sub>,h<sup>c</sup><sub>2</sub>,··· ,h<sup>c</sup><sub>n</sub>) ，对于序列的每个字符，错误修正的概率定义为：

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c35ef7b723884135a22f6059b32cb6d3~tplv-k3u1fbpfcp-zoom-1.image)

其中 P<sub>c</sub>(y<sub>i</sub> = j | X)   表示字符 x<sub>i</sub> 被更正为候选列表中字符 j 的条件概率，softmax 是 softmax 函数，从候选列表中选择概率最大的字符作为字符 x<sub>i</sub>  的输出，h′<sub>i</sub> 表示隐藏状态，公式为：

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/588ac748bcc2419baa53dd8171a7aaeb~tplv-k3u1fbpfcp-zoom-1.image)


其中  h<sup>c</sup><sub>i</sub>  是最后一层的隐藏状态，e<sub>i</sub> 是字符 x<sub>i</sub> 的输入嵌入。

### Learning


Soft-Masked BERT 的学习是端到端进行的，前提是对 BERT 进行了预训练，并给出了由原始序列和校正序列对组成的训练数据，表示为 = {(X<sub>1</sub>,Y<sub>1</sub>),(X<sub>2</sub> ,Y<sub>2</sub>),...,(X<sub>N</sub>,Y<sub>N</sub>)}。 创建训练数据的一种方法是在给定一个没有错误的序列 Y<sub>i</sub>，使用混淆表重复生成包含错误的序列 X<sub>i</sub>。

学习过程由优化两个目标驱动，分别对应于错误检测和错误纠正。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/71d28e87728c435680886cf5f027f106~tplv-k3u1fbpfcp-zoom-1.image)

其中 L<sub>d</sub> 是检测网络的训练目标，L<sub>c</sub> 是校正网络的训练目标。 这两个函数线性组合为学习的总体目标。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6f164710a40a4be4a2d8ac1426a5e3e6~tplv-k3u1fbpfcp-zoom-1.image)


## 实验


### Datasets

本文使用了包含 1100 个文本和 461 种错误类型的 SIGHAN 数据集，按照标准分成训练集、开发集和测试集。

另外为测试和开发创建了一个更大的数据集，称为 News Title ，该数据集包含 15730 个文本，有 5423 篇文本包含错误，分为 3,441 种类型。我们将数据分为测试集和开发集，每个包含 7,865 个文本。

此外，遵循 CSC 中的常见做法来自动生成用于训练的数据集，抓取了大约 500 万条新闻标题，我们还创建了一个混淆表，其中每个字符都与一些作为潜在错误的同音字符相关联。接下来，我们将文本中 15% 的字符随机替换为其他字符来人为地产生错误，其中 80% 是混淆表中的同音字符，其中 20% 是随机字符。这是因为在实践中，人们使用基于拼音的输入法，大约 80% 的中文拼写错误是同音字。



### Experiment Setting

使用了句子级别的准确率、准确率、召回率和 F1 分数作为评估标准

实验中使用的预训练 BERT 模型是 https://github.com/huggingface/transformers 提供的模型。在BERT的微调中，我们保留了默认的超参数，只使用 Adam 微调了参数。为了减少训练技巧的影响，我们没有使用动态学习率策略并在微调中保持学习率 2e-5。 Bi-GRU 中隐藏单元的大小为 256，所有模型使用的批大小为 320。

在 SIGHAN 上的实验中，对于所有基于 BERT 的模型，我们首先用 500 万个训练样本对模型进行微调，然后在 SIGHAN 中继续对训练样本进行微调。我们删除了训练数据中没有变化的文本以提高效率。

在 News Title 的实验中，模型仅使用 500 万个训练示例进行了微调。

 SIGHAN 和 News Title  的开发集用于的超参数调整。为每个数据集选择了超参数 λ 的最佳值。

### Main Results

![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fupload-images.jianshu.io%2Fupload_images%2F1599653-99b7c63fcdca17d5.png&refer=http%3A%2F%2Fupload-images.jianshu.io&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1641545610&t=034fb051da21cc8d95c5774b38412aad)

表中展示了所有方法在两个测试数据集上的实验结果。从表中可以看出，所提出的模型 Soft-Masked BERT 在两个数据集上都明显优于基线方法。

Soft-Masked BERT、BERT-Finetune 和 FASPell 的三种方法的性能优于其他基线，而 BERT-Pretrain 的方法性能相当差。结果表明，没有微调（即 BERT-Pretrain）的 BERT 不起作用，而经过微调（即 BERT-Finetune 等）的 BERT 可以显着提高性能。在这里我们看到了 BERT 的另一个成功应用，它可以获得一定量的语言理解知识。此外，Soft-Masked BERT 可以在两个数据集上大幅击败 BERT-Finetune。结果表明，错误检测对于在 CSC 中使用 BERT 很重要，soft masking 确实是一种有效的手段。

### Effect of Hyper Parameter

![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fwww.pianshen.com%2Fimages%2F58%2Fc0f1c5d89fc3dd6e50a35f162023eaba.png&refer=http%3A%2F%2Fwww.pianshen.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1641546352&t=7197633d0c07c3ddd53602a5b4cbef96)

表中展示了 Soft-Masked BERT 在  News Title 的测试数据上的结果，以说明参数和数据大小的影响。

显示了 Soft-Masked BERT 以及 BERT-Finetune 使用不同大小的训练数据学习的结果。 可以发现，对于 Soft-Masked BERT，表明使用的训练数据越多，可以实现的性能越高。 还可以观察到 Soft-Masked BERT 始终优于 BERT-Finetune。


![Impact of Different Values of λ.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c839a42860d8477c99743cc5dc3ab5f0~tplv-k3u1fbpfcp-watermark.image?)

更大的 λ 值意味着更高的纠错权重。 错误检测比纠错更容易，因为本质上前者是一个二分类问题，而后者是一个多类分类问题。 经过实验发现当 λ 为 0.8 时，获得最高的 F1 分数，这意味着达到了检测和校正之间的良好折衷。

### Ablation Study

我们在两个数据集上对 Soft-Masked BERT 进行了消融研究。论文中只给出了在 News Title 的测试结果，因为 SIGHAN 的结果类似。 

![Soft-Masked BERT Ablation Study.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/81132ac1785846e5aa289b9e8b33cce7~tplv-k3u1fbpfcp-watermark.image?)

在 Soft-Masked BERT-R 中，模型中的残差连接被删除。在 Hard-Masked BERT 中，如果检测网络给出的错误概率超过阈值，则当前字符的嵌入设置为 [MASK] 令牌的嵌入，否则嵌入保持不变。在 Rand-Masked BERT 中，错误概率随机化为 0 到 1 之间的值。研究发现 Soft-Masked BERT 的所有主要组件都是实现高性能所必需的。


本文还尝试了 BERT-Finetune+Force 模型的性能可以看作是一个上限。在该方法中，我们让 BERT-Finetune 只在有错误的位置进行预测，并从候选列表的其余字符中选择一个字符。结果表明，Soft-Masked BERT 比 BERT-Finetune+Force 性能较差，仍有很大的改进空间。