这是我参与更文挑战的第16天，活动详情查看： [更文挑战](https://juejin.cn/post/6967194882926444557)

## Seq2Seq 的不足
Seq2Seq 虽然有不少改进效果的技巧，但是其本身还有一个很大的缺陷，当输入的序列太长的时候，最后输出的状态向量 h 很难记住最开始的内容，或者某些关键的内容。

![seq2seq_blue.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/785099a663574bacabbd383379657b6e~tplv-k3u1fbpfcp-watermark.image)


如果用 Seq2Seq 做机器翻译，当输入句子的单词在 20 个附近时的效果最佳，当输入句子的单词超过 20 个的时候，效果会持续下降，这是因为 Encoder 会遗忘某些信息。而在 Seq2Seq 的基础上加入了 Attention 机制之后，在输入单词超过 20 个的时候，效果也不会下降。



## 引入 Attention 的 Seq2Seq 

引入了 Attention 的 Seq2Seq 模型，可以大幅度提升 Seq2Seq 的性能，因为 Decoder 在每次解码的时候又回顾了 Encoder 对输入总结的所有特征，同时 Attention 还会告诉 Decoder 应该更加关注 Encoder 的哪些输入及其特征，这也是 Attention 名字的来源。这种机制对输入的关注方式和人类似，我们在读一句话的时候，也会直接抓住重点字词，而不是每个字符或者词都是重点。

Attention 尽管可以大幅度提升性能，唯一的缺点就是要进行大量的计算。


## Attention 原理



![attention.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f85dda55c8c84eefa25da4ae0cab81b7~tplv-k3u1fbpfcp-watermark.image)


如图所示，左边的是 Encoder 过程，， 右边的是 Decoder过程，两部分都可以用 RNN 及其变体组成的结构，这里借用 SimpleRNN 来介绍 Attention 原理。在 Encoder 照常捕获了输入的特征之后，输出了每个时刻的状态向量 h<sub>i</sub> ，并将最后一个时刻的 h<sub>m</sub> 当作 Decoder 的初始状态向量 s<sub>0</sub> ，此时的 Deocder 过程如下：

a）先计算第一次解码，首先我们计算 Decoder 中的 s<sub>0</sub> 与 Encoder 中的每个状态向量 h<sub>i</sub> 计算权重（权重计算方法在下面会介绍），每个状态向量  h<sub>i</sub> 对应一个权重  a<sub>i</sub> ，a<sub>i</sub> 表示 h<sub>i</sub> 与 s<sub>0</sub> 的相关性大小，然后对所有的 [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>] 做 Softmax 转化，变为权重  [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>] ，然后我们使用新的权重  [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>]  和对应的状态向量 [h<sub>1</sub>,h<sub>2</sub>,...,h<sub>m</sub>] 求加权平均得到 c<sub>0</sub> 。然后我们利用 x<sup>'</sup><sub>1</sub> 、c<sub>0</sub>  以及 s<sub>0</sub> 计算得到 s<sub>1</sub> ，公式如下：

**s<sub>1</sub> = tanh( A<sup>'</sup> * contact(x<sup>'</sup><sub>1</sub> ,c<sub>0</sub>, s<sub>0</sub>) + b)**

【关键解释】**因为 c<sub>0</sub> 是 Encoder 中所有时刻的状态向量加权和，所以它知道完整的 Encoder 输入信息，这就解决了 Seq2Seq 的遗忘问题。再加上当前的输入信息 x<sup>'</sup><sub>1</sub> 以及上一个时刻的状态信息 s<sub>0</sub> ，所以可以预测当前时刻的状态向量输出 s<sub>1</sub> 。**

b）第二次解码，和上面类似，我们计算 Decoder 中的 s<sub>1</sub> 与所有的 Encoder 中状态向量 h<sub>i</sub> 的权重，每个状态向量  h<sub>i</sub> 对应一个权重  a<sub>i</sub> ，a<sub>i</sub> 表示 h<sub>i</sub> 与 s<sub>1</sub> 的相关性，然后对所有的 [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>] 做 Softmax 转化，变为权重  [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>] ，然后我们使用新的 a<sub>i</sub> 和对应的 h<sub>i</sub> 求加权平均得到 c<sub>1</sub> 。然后我们利用 x<sup>'</sup><sub>2</sub> 、c<sub>1</sub>  以及 s<sub>1</sub> 得到 s<sub>2</sub> ，公式如下：

**s<sub>2</sub> = tanh( A<sup>'</sup> * contact(x<sup>'</sup><sub>2</sub>, c<sub>1</sub>, s<sub>1</sub>) + b)**

【关键解释】**因为 c<sub>1</sub> 是 Encoder 中所有时刻的状态向量加权和，所以它知道完整的 Encoder 输入信息，这就解决了 Seq2Seq 的遗忘问题。再加上当前的输入信息 x<sup>'</sup><sub>2</sub> 以及上一个时刻的状态信息 s<sub>1</sub> ，所以可以预测当前时刻的状态向量输出 s<sub>2</sub> 。**

c）类似重复上面的解码过程，直到结束。

## 权重计算的两种方法

一般情况下有两种计算 Decoder 中的 s<sub>i</sub> 与所有的 Encoder 中状态向量 h<sub>i</sub> 的权重大小。

第一种是原论文中的方法，如下图所示。图中以  s<sub>0</sub> 与所有的 Encoder 中状态向量 h<sub>i</sub> 计算权重为例。将 h<sub>i</sub> 和 s<sub>0</sub> 进行拼接，然后与参数矩阵 W 相乘后，经过了非线性函数 tanh 的转化，最后将得到的结果与参数矩阵 v<sup>T</sup> 相乘可以得到 a<sub>i</sub> ，因为有 m 个输入，所以 Encoder 有 m 个状态向量，因此需要计算出 m 个 a ，最后将  [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>]  经过 Softmax 变化得到新权重参数的  [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>]  。这里的 W 和 v<sup>T</sup> 都是需要训练的参数。


![similarity1.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/73b96bc161af4532908adc1f15b987c7~tplv-k3u1fbpfcp-watermark.image)

第二种是 Transformer 模型采用的方法，如下图。这里仍然以  s<sub>0</sub> 与所有的 Encoder 中状态向量 h<sub>i</sub> 计算权重为例，将  W<sub>K</sub> 与  h<sub>i</sub> 相乘得到  k<sub>i</sub> ，用  W<sub>Q</sub> 与  s<sub>0</sub> 相乘得到   q<sub>0</sub> ，然后把 k<sup>T</sup><sub>i</sub> 与 q<sub>0</sub> 的内积当作相似度  a<sub>i</sub> 。因为有 m 个输入，所以 Encoder 有 m 个状态向量，因此需要计算出 m 个 a ，最后将  [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>]  经过 Softmax 变化得到新的权重参数  [a<sub>1</sub>,a<sub>2</sub>,...,a<sub>m</sub>]  。这里的 W<sub>K</sub> 和 W<sub>Q</sub> 都是需要训练的参数。


![similarity2.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e2be7922fbea48b89b454f0b2e0c62c7~tplv-k3u1fbpfcp-watermark.image)

## 时间复杂度

假如输入长度为 m ，目标长度为 t 。

我们在引入 Attention 机制之后，在 Encoder 得到 m 个状态向量只哦呼，在接下来的 Decoder 过程中，每次解码都计算了 m 个 a ， Decoder 过程执行了 t 次，则最后一共计算了 m*t 个 a ，所以时间复杂度为 O(m+m\*t) 。所以在 Seq2Seq 中引入 Attention 虽然可以大幅度提升性能，避免遗忘问题，但是代价就是需要巨大的计算量。

而没有引入 Attention 机制的 Seq2Seq 的，因为 Encoder 只计算了 m 个状态向量，Decoder 解码了 t  次，所以时间复杂度仅为 O(m+t) 。

## 权重可视化



![weights_visualization.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/75c6f3d23cde42cf8c292a1e36b620b8~tplv-k3u1fbpfcp-watermark.image)

这里用英语翻译成法语为例，从可视化的角度直观解释权重参数表达的含义，图中紫色的线的粗细就表示了权重大小的程度。当在 Decoder 过程中翻译到单词 zone 的时候，它会与 Encoder 中的每个输入计算权重参数，我们可以看到 zone 虽然与所有的输入单词都有权重，但是与单词 Area 的权重值明显最大，表示翻译 zone 的的时候需要特别关注 Area 这个词，换句话说 Area 这个词对翻译 zone 的影响程度是最大的，而实际上法语中的 zone 和英语中 Area 的含义是相近的。这也是 Attention 名字的由来。又比如在翻译法语 Européenne 的时候，需要特别关注英语中 European ，道理同上。


## 案例

我自己用实现的一个有趣小案例，将胡乱写的字符串翻译成英文，有详细的注释，并且实现了两种权重计算方法，包教包会，肝文不易，觉好留赞。https://juejin.cn/post/6950602231905746975