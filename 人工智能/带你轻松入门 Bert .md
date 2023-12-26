这是我参与更文挑战的第22天，活动详情查看： [更文挑战](https://juejin.cn/post/6967194882926444557)

## 什么是 BERT
BERT 的全称是 Bidirectional Encoder Representations from Transformers ，其实 BERT 的目的就是预训练 [Transformers](https://arxiv.org/pdf/1706.03762.pdf) 模型的 Encoder 网络，从而大幅度提高性能。本文没有讲具体的技术细节，只介绍主要想法。 具体内容可以看论文: https://arxiv.org/pdf/1810.04805.pdf
## 第一个任务
BERT 的第一个训练模型的任务就是，随机地遮挡一个或者多个单词，然后让模型预测遮挡的单词。具体过程如图所示。



![bert-1.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1b1217b1227a4e48bad63cbee0fe9b6e~tplv-k3u1fbpfcp-watermark.image)

* 图中的输入的第二个单词原本为 cat ，但是被 MASK 符号代替。

* MASK 会被 Embedding 层转换为词向量 x<sub>M</sub> 。

* 该时刻的输出即为  u<sub>M</sub> ，而  u<sub>M</sub> 不仅依赖于 x<sub>M</sub> ，而且依赖于所有的  x<sub>1</sub> 到 x<sub>6</sub> 的所有输入向量。也就是说  u<sub>M</sub> 知道所有的输入信息。正因为 u<sub>M</sub> 知道上下文的所有信息，所以可以用来预测被遮挡的单词 cat 。

* 将   u<sub>M</sub>  作为特征向量输入到 Softmax ，得到一个概率分布 p ，可以通过字典得到最大概率所对应的单词。这里被遮挡的是 cat 单词，所以要训练模型，使得模型的输出概率分布 p 中 cat 的概率值尽量最大。


## 第一个任务有什么用
BERT 在训练时候不需要人工标注数据集，可以大大节省成本和时间，训练数据也很好获得，可以用任何的书籍，文章等作为训练数据，它可以自动生成标签，轻松进行模型的预训练。


## 第二个任务

BERT 的第二个训练模型的任务就是，给出两个句子，判断这两个句子是不是相邻的。

首先准备训练数据，训练数据中 50% 的样本用真实相邻的两个句子，剩下的 50% 的样本用随机抽样的方式选取任意两个不相邻的句子。

选用真实相邻的句子的处理方式如下图所示拼接起来，拼接的时候用到了符号 CLS 和 SEP ，CLS 是表示“分类”的符号，SEP 是表示“分割”两个句子的符号。因为这两句话确实是相邻的句子，所以他们的标签为 true 。



![bert-3.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/089c9cb3f74c4af5a95d4fc246c76f05~tplv-k3u1fbpfcp-watermark.image)

选用不相邻的句子的处理方式如下图所示拼接起来，但因为这两句话不是相邻的句子，所以他们的标签为 false 。


![bert-4.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c07a5e9affd04adabe92352aa2ffc828~tplv-k3u1fbpfcp-watermark.image)

处理好训练数据，然后我们训练模型，用模型判断两个句子是否是上下相邻的。具体过程如下图所示。


![bert-2.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/06bedd9332ff4fbca0ecddde4fa3610a~tplv-k3u1fbpfcp-watermark.image)


* 将 [CLS][第一个句子][SEP][第二个句子] 拼接的字符序列输入到模型中。

* 通过 Embedding 层转换成词向量。

* [CLS] 位置的最终输出为向量 c ，由于向量 c 中经过模型提取得到了整个输入的两句话的信息，所以靠向量  c 可以判断两句话是否是真实相邻。

* 把向量 c 输入一个二分类器，输出的值是 0 或者 1 ，0 代表 false ，1 代表 true 。可以训练模型使得两个句子的预测标签可以尽可能接近它们的真实标签。

## 第二个任务有什么用
相邻的两句话通常有关联，通过做二分类判断，可以强化这种关联，训练 Embedding 的词向量强化这种内在关联。

 Transformer 的 Encoder 层中有 Self-Attention 机制，而 Self-Attention 的作用就是去找输入之间的相关性，而这个任务也可以加强以寻找输入之间的正确的相关性。

## 第三个任务
第一个任务就是预测遮挡单词，第二个任务就是判断两句话是否相邻。BERT 还能将这两个任务结合起来预训练 Transformer 的 Encoder 结构。

我们需要准备数据，如下图所示我们用到了真实相邻的两个句子作为训练数据，并且随机遮挡了 15% （这里是两个）的单词，这里一共有三个目标，因为是真实相邻的句子所以第一个目标为 true ，第二个目标就是真实遮挡的单词 branch ，以及第三个目标是真实遮挡的单词 was 。


![bert-5.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a521b14fbe234bd693bcac68f2ec4915~tplv-k3u1fbpfcp-watermark.image)

另外需要找不是真实相邻的句子作为训练数据，并且同样遮挡单词，这里只遮挡一个单词，所以有两个目标，第一个目标分别为 false ，第二个目标为单词 south 。


![bert-6.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/94fd6208c2a146049c4de05ce4273265~tplv-k3u1fbpfcp-watermark.image)
假如像上面有三个目标则有三个损失函数（如果有两个目标则有两个损失函数），第一个目标是二分类任务，第二个和第三个目标是多分类任务。目标函数是三个损失函数的和，然后关于模型参数求梯度，然后通过梯度下降来更新模型参数。

## BERT 优点
BERT 可以自动生成标签，不需要人工标注数据，这是个很耗时耗力的工作，而且很昂贵。

BERT 可以用各种文本数据，书，网页，新闻等

BERT 表现出来的成绩确实很优异

## BERT 缺点
BERT 的想法简单，模型也有效，但是代价很大，普通人难以有时间和精力去训练 BERT ，幸好已经公开了，可以自行[下载](https://github.com/google-research/bert.git)。

## 参考

[1] Devlin J ,  Chang M W ,  Lee K , et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[J].  2018.

[2] Vaswani A ,  Shazeer N ,  Parmar N , et al. Attention Is All You Need[J]. arXiv, 2017.