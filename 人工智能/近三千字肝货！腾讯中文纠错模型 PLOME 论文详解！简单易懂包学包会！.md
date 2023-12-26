本文主要介绍腾讯 AI Platform Department 发表的一篇关于中文纠错模型的论文，收录在第 2021 年 ACL ，论文地址：https://aclanthology.org/2021.acl-long.233.pdf

## 功能

PLOME 是为中文拼写纠正（CSC） 任务设计的，旨在检测和纠正中文文本中的拼写错误。

## 摘要

* 本文为中文拼写纠正 (CSC)  提出了一种带有拼写错误知识的预训练掩码语言模型 PLOME ，可以在与训练期间学习语义和拼写错误知识
*  PLOME 根据混淆集用相似的字符屏蔽所选的标记，而不是像 BERT 那样使用固定标记 “[MASK]” 
*  除了字符预测，PLOME 还引入了字符的发音和笔画作为输入，通过恢复掩码位置的真实字符和语音来学习字符和语音级别的拼写错误知识
*  PLOME 利用 GRU 网络对语音和笔画进行建模


## 贡献

* 本文方法相对于最先进的方法取得了显着的优势
* 发布源代码和预训练模型供社区使用: https://github.com/liushulinle/PLOME
* PLOME 是第一个专为中文拼写纠正而设计的特定任务语言模型，且是第一个在字符和语音级别上对该任务进行建模的
* PLOME 引入了拼音和笔画，这使它能够对任意字符之间的相似性进行建模

## 模型框架

下面是 PLOME 的主要框架图。
  
  ![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Finotgo.com%2FimagesLocal%2F202106%2F30%2F20210630212556609h_2.png.jpg&refer=http%3A%2F%2Finotgo.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1641450783&t=b41188a089c8c83d44ec87da6cf0c92a)
  
### Confusion Set based Masking Strategy
 
 PLOME 只遮盖输入的 15% 的 token ，它不像 BERT 只是用 MASK 进行遮盖，而是对遮盖的 token 执行四种可能的不同的变化，具体例子看下图案例：
 
*  60% 的可能变成语音相似字符
*  15% 的可能变成视觉相似字符
*  15% 的可能不变
*  10% 的可能从词表中随机选取一个


![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Finotgo.com%2FimagesLocal%2F202106%2F30%2F20210630212556609h_3.png&refer=http%3A%2F%2Finotgo.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1641451417&t=007be1b2a2f609e36182c85a2c99c624)

### Embedding Layer

通过看上面的框架图知道每个字符的最终嵌入是 character embedding 、position embedding 、phonic embedding 和  shape embedding 的总和，前两者是通过查找表获得的，其中词表大小和嵌入维度与 BERT<sub>base</sub> 中的相同。

在本文中，使用 [Unihan Database<sup>3</sup>](http://www.unicode.org/charts/unihan.html) 来获取汉字-拼音映射（没有音调），将每个字的拼音字母输入到 1 层 GRU 网络以生成 Phonic Embedding ，期望相似的拼音有相似的嵌入。下图中间部分给出了一个示例。

![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fnimg.ws.126.net%2F%3Furl%3Dhttp%253A%252F%252Fdingyue.ws.126.net%252F2021%252F0914%252Fbe6562e5p00qzf853000zd200e500aag00e500aa.png%26thumbnail%3D650x2147483647%26quality%3D80%26type%3Djpg&refer=http%3A%2F%2Fnimg.ws.126.net&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1641452832&t=ea6156b10f555b505c5db7b46803b468)

使用 [Stroke Order<sup>4</sup>](https://en.wikipedia.org/wiki/Stroke\_order) 来表示一个汉字笔画序列。在本文中，笔画数据是通过 [Chaizi Database<sup>5</sup>](https://github.com/kfcd/chaizi) 获得的。为了建模字符之间的视觉关系，将每个汉字的笔画顺序输入另一个 1 层 GRU 网络以生成 Shape Embedding ，在上图的底部给出了一个示例。

### Transformer Encoder
 
PLOME 中用到的 Transformer 的架构与 BERT<sub>base</sub> 中的架构相同。 共有 12 层 ，每层隐单元大小为 768 ，注意力头数为 12 。
 
### Output Layer

PLMOE 对每个选定的字符进行两部分内容预测：

* Character Prediction： 与 BERT 类似，PLOME 根据最后一个 Transformer 层生成的嵌入来预测每个掩码标记的原始字符

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/925a034f595042dfa4ef3736d0db56fc~tplv-k3u1fbpfcp-zoom-1.image)


* Pronunciation Prediction：语音错误在汉语拼写错误中占主导地位。大约 80% 的拼写错误是语音错误。为了在语音层面学习拼写错误的知识，PLOME 还预测每个掩码标记的真实发音，其中发音由拼音呈现，没有音调。
 
![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e517d1713e4346e0839d697d21c46866~tplv-k3u1fbpfcp-zoom-1.image)
 

 
### Learning

上面预测有两部分，模型的学习过程自然也是有两部分目标驱动的，分别是 Character Prediction 和 Pronunciation Prediction 。
其中 L<sub>c</sub> 是字符预测的目标，l<sub>i</sub> 是 x<sub>i</sub> 的真实字符，L<sub>p</sub> 是发音预测的目标，r<sub>i</sub> 是真实拼音。 

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/96c1b1fa53544273a60e43f09318f1c0~tplv-k3u1fbpfcp-zoom-1.image)

总体目标定义为：

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d3d96886f84c4039b2a447148cbf6ac2~tplv-k3u1fbpfcp-zoom-1.image)




### Fine-tuning Procedure
 
PLOME 是为 CSC 任务设计的，旨在检测和纠正中文文本中的拼写错误。形式上，给定由 n 个字符组成的字符序列 X = {x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>}，该模型预期生成目标序列 Y = {y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>n</sub>}，其中的错误能得到纠正。

Training：学习目标与预训练过程中的完全相同。此过程类似于预训练，不同之处在于： (1) 混淆集掩码策略被去掉了 ；(2)所有输入字符都需要预测，而不是像预训练那样只选择 token 。

Inference：由于 PLOME 预训练时候预测每个遮蔽 token 的字符分布和发音分布，因此构建了联合分布为：

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ba966ed8ea41475c8b5751d9fb247cb7~tplv-k3u1fbpfcp-zoom-1.image)

其中 p<sub>j</sub> (y<sub>i</sub> = j |X ) 是结合字符和发音预测，将 x<sub>i</sub> 的原始字符预测为第 j 个字符的概率，p<sub>c</sub> 和 p<sub>p</sub> 在等式 1 和等式 2 中分别定义，j<sup>p</sup>是第 j 个字符的发音。为此构建了一个指标矩阵 I∈R<sup>n<sub>c</sub>×n<sub>p</sub></sup>，其中如果第 i 个字符的发音是第 j 个拼音，则 I<sub>i,j</sub> 设置为 1 ，否则设置为 0 。那么联合分布可以是计算方式：

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e807382d0ce6480e9e9c012ba30c5990~tplv-k3u1fbpfcp-zoom-1.image)

使用联合概率作为预测分布。对于每个输入的 token ，选择联合概率最高的字符作为最终输出。联合分布同时考虑了字符和发音预测，因此更准确。
 
## 实验

### Pre-training 

Dataset：使用 [wiki2019zh<sup>6</sup>](https://github.com/suzhoushr/nlp_chinese_corpus) 作为预训练语料库，它由 100 万个中文维基百科页面组成。 此外还有 300 万篇新闻文章。 将这些页面和文章拆分成句子，总共获得了 1.621 亿个句子。 然后连接连续的句子以获得最多 510 个字符的文本片段，用作训练样例。

Parameter Settings：略
 
### Fine-tuning 

Dataset：训练数据由来自 2013 年、2014 年、2015 年的 SIGHAN 数据集构成 10K 手动注释样本和 271K 自动生成的样本组成。

Evaluation Data：使用最新的 SIGHAN 测试数据集来评估所提出的模型，该模型包含 1100 篇文本和 461 种错误类型。
 
Evaluation Metrics：使用最常见的准确率、召回率和 F1 分数作为评估指标。

Parameter Settings ：略


### Main Results

![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Finotgo.com%2FimagesLocal%2F202106%2F30%2F20210630212556609h_5.png&refer=http%3A%2F%2Finotgo.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1641464126&t=6d713dcccf18e73d835af7f55dfcfc3d)

从上面的测试结果表中，可以得到以下一些结论：

* 在没有微调的情况下，中间组中的预训练模型取得了较好的结果，甚至以显着的收益优于监督方法 PN 。这表明基于混淆集的屏蔽策略使模型能够在预训练期间学习特定于任务的知识。
* cBERT-finetune 在所有指标上都优于 BERT-finetune，这表明所提出的掩码策略提供了必要的知识，并且无法通过微调来学习。
* 结合语音和形状嵌入，PLOME-Finetune 在句子级检测和纠正方面的绝对改进超过 cBERT-Finetune 2.3% 和 2.8%。这表明字符的拼音和笔画提供了有用的信息，并且很难从混淆集中学习。
* SpellGCN 和本文方法使用的相同混淆集。 但采用不同的策略来学习其中包含的知识。 SpellGCN 构建了一个 GCN 网络来对这些信息进行建模，而 PLOME 在预训练期间从大规模数据中学习它。 PLOME 在所有指标上都取得了更好的性能，这表明本文方法更有效地对此类知识进行建模。

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a35f26cf8a7a4b2fbeb1bab99f8a9bbb~tplv-k3u1fbpfcp-zoom-1.image)

* 从上面的图，使用整个测试集和 SIGHAN 进行比较， PLOME 在所有指标上几乎始终优于 BERT 和 SpellGCN。
* 另外还在 SIGHAN13 和 SIGHAN14 上评估了所提出的模型， PLOME 始终优于所有比较模型。


### Effects of Prediction Strategy



PLOME 预测每个字符的三个分布：字符分布 p<sub>c</sub>、发音分布 p<sub>p</sub> 和联合分布 p<sub>j</sub>。后两个分布与发音预测有关，CSC 任务需要字符预测，因此只比较字符预测 p<sub>c</sub> 和联合预测 p<sub>j</sub> 的效果。

从实验中观察到联合分布在所有评估指标上都优于字符分布。尤其是精度分数的差距更加明显。联合分布同时考虑了字符和发音预测，因此预测结果更加准确。

### Effects of Initialization Strategy


一般来说，初始化策略对深度模型的性能有很大影响。本文基于 cBERT 和 PLOME 实现了四个基线。从实验中观察到用 BERT 初始化的 cBERT 和 PLOME 都获得了更好的性能。特别是所有评估的召回分数都有显着提高。本文认为以下两个原因可以解释这种现象。 1） BERT 中丰富的语义信息可以有效提高泛化能力。 2）PLOME 由两个 1 层 GRU 网络和一个 12 层 transformer 编码器组成，总共包含超过 110M 的参数。从头开始训练如此大规模的模型时，很容易陷入局部优化。


### Phonic/Shape Embedding Visualization

通过 GRU 网络为每个字符生成语音和形状嵌入，然后将它们可视化。根据 GRU 网络生成的 768 维嵌入的余弦相似度说明了最接近“锭”的 30 个字符，该相似度通过 t-SNE 进行可视化。 一方面，几乎所有的“锭”、“绽放”等类似“锭”的汉字都比较接近。 另一方面，彼此相似的字符彼此非常接近。 这些现象表明学习的形状嵌入很好地模拟了形状相似性。 另外图中还显示了和 “ding” 相似的拼音都聚集在一块的情况。

### Converging Speed of Various Models

由于基于混淆集的掩蔽策略，cBERT 和 PLOME 在预训练过程中学习了特定于任务的知识，因此它们在训练开始时取得了比 BERT 更好的性能。 此外，本文所提出的模型需要更少的训练步骤来实现相对较好的性能。 PLOME 只需要 7k 步就可以达到 80% 的分数，而 BERT 需要 47k 步。


## 尾记

- 本文撰写不易，转载标明出处
- 希望各位同学能学到知识，如果能点赞支持一下当然就更加完美🙏
- 欢迎评论区交流讨论