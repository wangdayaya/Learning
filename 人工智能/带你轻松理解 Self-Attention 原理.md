这是我参与更文挑战的第21天，活动详情查看： [更文挑战](https://juejin.cn/post/6967194882926444557)

## 引言

之前我们上文介绍了用 Attention 来提升 Seq2Seq 的性能，将 Attention 共同作用于 Seq2Seq 的 Encoder 和 Decoder 两个部分。本文我们介绍 Self-Attention ，可以将 Attention 单独用到其中的一部分里面。

愿论文中 Self-Attention 作用于 LSTM ，这里我简化过程，用 SimpleRNN 代替 LSTM 介绍该思想。



## SimpleRNN  + Self-Attention 核心原理

**【SimpleRNN 求 h<sub>i</sub> 的方法】**

我们之前在 [SimpleRNN](https://juejin.cn/post/6972340784720773151) 中求 h<sub>i</sub> 的时候，是按照下面的这个公式的思路进行的：

**h<sub>i</sub> = tanh(A * concat(x<sub>i</sub>, h<sub>i-1</sub>)+b)**

说明当前时刻的隐层状态依赖于当前的输入 x<sub>i</sub> 和上一时刻的隐层状态输入 h<sub>i-1</sub> 。

**【SimpleRNN  + Self-Attention 求 h<sub>i</sub> 的方法】**



当引入 Self-Attention 之后，SimpleRNN 求 h<sub>i</sub> 的方式发生了变化，是按照下面的这个公式的思路进行的：

**h<sub>i</sub> = tanh(A * concat(x<sub>i</sub>, c<sub>i-1</sub>)+b)**


![self-attention-h.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5462a20a51dc4bef82ceacd28b89b473~tplv-k3u1fbpfcp-watermark.image)

图中例子说明 t<sub>3</sub> 时刻的隐层状态  h<sub>3</sub> 依赖于当前的输入 x<sub>3</sub> 和上一时刻的上下文向量 c<sub>2</sub> 。

其中 c<sub>i</sub> 就是将第 i 时刻的隐层输出  h<sub>i</sub>  与已有的   h<sub>1</sub>、... 、 h<sub>i</sub> 进行权重计算，得到权重列表 a<sub>1</sub>、... 、 a<sub>i</sub> ，最后将这些隐层输出与各自对应的权重参数进行加权平均求和得到 c<sub>i</sub> 。至于具体的权重计算方法和 [Attention](https://juejin.cn/post/6974298391874863141) 文章中提到的方法一样，这里不再赘述。


![self-attention-c.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/45a1df40f8094b4ebe06f27c282602e1~tplv-k3u1fbpfcp-watermark.image)

从图中的例子可以 c<sub>3</sub> 是 h<sub>1</sub>、 h<sub>2</sub> 、 h<sub>3</sub> 及各自对应权重 a<sub>1</sub>、 a<sub>2</sub> 、 a<sub>3</sub> 的加权平均和。

另外，可以考虑换更加复杂的计算思路，其他具体过程和上述一样：

**h<sub>i</sub> = tanh(A * concat(x<sub>i</sub>, c<sub>i-1</sub>, h<sub>i-1</sub>,)+b)**

## 总结

* Self-Attention 和 Attention 一样，都能解决 RNN 类模型的遗忘问题，每次在计算当前隐层输出  h<sub>i</sub> 的时候，都会用  c<sub>i-1</sub> 来回顾一下之前的信息，这样就能记住之前的信息。但是  Self-Attention 中的 c<sub>i</sub> 的计算在自身的 RNN 结构中即可计算，而不像 Seq2Seq 中的 Attention 那样横跨 Decoder 和 Encoder 两个 RNN 结构，即 Decoder 的 c<sub>i</sub> 依赖于 Encoder 的所有隐层输出。

* Self-Attention 可以作用于任何 RNN 类的模型来提升性能了，如 LSTM 等。

* Self-Attention 还能帮助 RNN 关注相关的信息，如下图所示，红色单词是当前的输入，蓝色单词表示与当前输入单词较相关的单词。
![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5048ba01ee8b41f39b4977fb196b3311~tplv-k3u1fbpfcp-zoom-1.image)

## 参考

Cheng J ,  Dong L ,  Lapata M . Long Short-Term Memory-Networks for Machine Reading[C]// Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. 2016.