这是我参与更文挑战的第23天，活动详情查看： [更文挑战](https://juejin.cn/post/6967194882926444557)


## 引言

Transformer 模型内部细节很多，本文只主要讲解 Attention 部分和 Self-Attention 部分，如果感兴趣可以查看论文。

## 什么是 Transformer
* Transformer 是一个 Seq2Seq 模型，很适合机器翻译任务。不了解 Seq2Seq 模型的，可以看我之前的文章 [《Seq2Seq 训练和预测详解以及优化技巧》](https://juejin.cn/post/6973930281728213006)

* 它不是循环神经网络结构，而是单纯靠 Attention 、Self-Attention 和全连接层拼接而成的网络结构。

* Transformer 的测评性能完全碾压最好的 RNN + Attention 结构，目前业内已经没有人用 RNN ，而是用 BERT + Transformer 模型组合。

## 回顾 RNN + Attention 结构

![transformer-rnn-attention.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d0e56bc5147b421d90b57cb5c5535605~tplv-k3u1fbpfcp-watermark.image)

如图所示是一个 RNN + Attention 组合而成的模型，在 Decoder 的过程中计算 c<sub>j</sub> 的过程如下：

a）将 Decoder 的第 j 时刻的输出向量 s<sub>j</sub> 与 W<sub>Q</sub> 相乘得到一个 q<sub>:j</sub> 

b）将每个 Encoder 的隐层输出 h<sub>i</sub> 与 W<sub>K</sub> 相乘得到 k<sub>:i</sub> ，因为有 m 个输入，所以有 m 个 k<sub>:i</sub> 向量，用 K 表示。

c）用 K<sup>T</sup> 与 q<sub>:j</sub> 相乘，可以得到一个 m 维的向量，然后经过 Softmax 就可以得到 m 个权重 a<sub>ij</sub> 。

【解释】q<sub>:j</sub> 被成为 Query ，k<sub>:i</sub> 被称为 Key，Query 的作用是用来匹配 Key ，Key 的作用是被 Query 匹配，经过计算得到的权重 a<sub>ij</sub> 表示的就是 Query 和每个 Key 的匹配程度，匹配程度越高 a<sub>ij</sub> 越大。我认为可以这样理解， Query 捕获的是 Decoder 的  s<sub>j</sub> 特征，Key 捕获的是 Encoder 输出的 m 个   h<sub>i</sub>  的特征，a<sub>ij</sub> 表示的就是 s<sub>j</sub> 与每个 h<sub>i</sub> 的相关性。

d）将每个 Encoder 的隐层输出 h<sub>i</sub> 与 W<sub>V</sub> 相乘得到 v<sub>:i</sub> ，因为有 m 个输入，所以有 m 个 v<sub>:i</sub> 向量，用 V 表示。

e）经过以上的步骤，Decoder 的第 j 时刻的  c<sub>j</sub> 可以计算出来，也就是 m 个 a 和对应的 m 个 v 相乘，求加权平均得到的。

【注意】W<sub>V</sub>、W<sub>K</sub>、W<sub>Q</sub> 三个参数矩阵是需要从训练数据中学习。

## Transformer 中的 Attention 
在 Transformer 中移除了 RNN 结构，只保留了 Attention 结构，可以从下图中看出，使用 W<sub>K</sub> 和 W<sub>V</sub> 与 Encoder 的 m 个输入 x 进行计算分别得到 m 个 k<sub>:i</sub> 和 m 个 v<sub>:i</sub> 。使用 W<sub>Q</sub> 和 Decoder 的 t 个输入 x 进行计算得到 t 个 q<sub>:t</sub> 。

![transformer-attention-1.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/550eedb6e8064d639c7a2b65bd7a4da8~tplv-k3u1fbpfcp-watermark.image)

如下图，这里是计算 Decoder 第 1 个时刻的权重， 将 K<sup>T</sup> 与 q<sub>:1</sub> 相乘，经过 Softmax 转化得到 m 个权重 a ，记做 a <sub>:1</sub> 。

![transformer-attention-2.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fe2d45a617e049099e0c57aa3ddd5403~tplv-k3u1fbpfcp-watermark.image)

如下图，这里是计算 Decoder 第 1 个时刻的上下文特征 c<sub>:1</sub> ，将 m 个权重 a 与 m 个 v 分别相乘求和，得到加权平均结果即为 c<sub>:1</sub>。

![transformer-attention-3.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/37a26758a49543a880c5b3abc1e42f95~tplv-k3u1fbpfcp-watermark.image)

类似地，Decoder 的每个时刻的上下文特征都可以和上面一样计算出来。说白了 c<sub>:j</sub> 依赖于当前的 Decoder 输入 x<sup>'</sup><sub>j</sub> 以及所有的 Encoder 输入 [x<sub>1</sub>,...,x<sub>m</sub>] 。

![transformer-attention-4.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a3b74f9b38864b7b9088d4d7cca0a5c6~tplv-k3u1fbpfcp-watermark.image)

总结如下图，Encoder 的输入是序列 X ，Decoder 的输入是序列  X<sup>'</sup> ，上下文向量 C 是关于 X 和  X<sup>'</sup> 的函数结果，其中用到的三个参数矩阵 W<sub>Q</sub> 、W<sub>K</sub> 、W<sub>V</sub>  都是需要通过训练数据进行学习的。

![transformer-attention-5.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2ea83193853e44df9870311fed199ac8~tplv-k3u1fbpfcp-watermark.image)


下图所示是机器翻译的解码过程，Transformer 的这个过程和 RNN 的过程类似，RNN 是将状态向量  h<sub>:j</sub> 输入到 Softmax 分类器，只不过 Attention 是将上下文特征 c<sub>:j</sub> 输入到 Softmax 分类器，然后随机抽样可以预测到下一个单词的输入。


![transformer-attention-6.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/14d7726087ce4959a9b7cf5228a66305~tplv-k3u1fbpfcp-watermark.image)


## Transformer 中的 Self-Attention 

Self-Attention 的输入只需要一个 X 输入序列，这里分别用 W<sub>Q</sub> 、W<sub>K</sub> 、W<sub>V</sub> 与每个输入 x<sub>i</sub> 进行计算得到 m 个 q<sub>:i</sub>、 k<sub>:i</sub>、 v<sub>:i</sub> 三个向量。而第 j 时刻的权重和上面的计算方式一样，Softmax(K<sup>T</sup>  * q<sub>:j</sub>) 可以得到第 j 时刻的 x<sub>j</sub> 关于所有输入 X 的 m 个权重参数，最后将 m 个权重参数与 m 个 v<sub>:i</sub> 分别相乘求和，即可得到上下文向量 c<sub>:j</sub> 。



![transformer-self-attention-1.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/9567786bdfe8433f8ba4455fb7372980~tplv-k3u1fbpfcp-watermark.image)

类似的，所有时刻的 c<sub>:j</sub> 都可以用同样的方法求出来。

![transformer-self-attention-2.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8b6e3c10603849cfacc029766e0b3764~tplv-k3u1fbpfcp-watermark.image)

总结，输入是序列 X ，上下文向量 C 是关于 X 和 X 的函数结果，因为每次在计算 x<sub>j</sub> 的上下文向量 c<sub>j</sub>的时候，都是需要将 x<sub>j</sub> 与所有 X 一起考虑进去并进行计算。其中用到的三个参数矩阵 W<sub>Q</sub> 、W<sub>K</sub> 、W<sub>V</sub>  都是需要通过训练数据进行学习的。

![transformer-self-attention-3.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/45e93761445e4687b4b5ff6cc3b480a5~tplv-k3u1fbpfcp-watermark.image)

## 参考 
[1] Vaswani A ,  Shazeer N ,  Parmar N , et al. Attention Is All You Need[J]. arXiv, 2017.