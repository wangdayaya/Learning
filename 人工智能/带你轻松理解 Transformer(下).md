这是我参与更文挑战的第24天，活动详情查看： [更文挑战](https://juejin.cn/post/6967194882926444557)


## 承上启下

上文我们介绍了 Transformer 中主要的 Attention 和 Self-Attention 原理，本文将介绍如果使用它们搭建深度神经网络，具体技术细节可以查看论文。如果对基本知识不了解，可以复习上一篇文章[《带你轻松理解 Transformer(上)》](https://juejin.cn/post/6976906257462460424)。

## Single-Head Self-Attention

上文中我们介绍的 Self-Attention 输入一个序列 X ，经过三个参数矩阵 W<sub>Q</sub>、W<sub>K</sub>、W<sub>V</sub> 的计算，输出一个上下文状态特征序列 C ，这样的 Self-Attention 被称为 Single-Head Self-Attention 。结构如图。

![single-head-self-attention.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e92c0f10483f4900b610bd6b9c23a8a4~tplv-k3u1fbpfcp-watermark.image)

##  Multi-Head Self-Attention

Multi-Head Self-Attention 是由 l 个 Single-Head Self-Attention 组成的，每个 Single-Head Self-Attention 都独享自己的参数，不会互相共享参数。每个  Single-Head Self-Attention 有 3 个参数矩阵，所以 Multi-Head Self-Attention 有 3l 个参数矩阵。

如下图所示，Multi-Head Self-Attention 中的每个 Single-Head Self-Attention 都有相同的输入序列 X ，但是由于各自的参数矩阵各不相同，所以输出的 l 个上下文状态特征序列 C 也是各不相同。把 l 个 C 按照不同的时刻拼接起来，就是 Multi-Head Self-Attention  的输出。如果每个 Single-Head Self-Attention 输出大小为 【d，m】，d 是特征维度，m 是输入个数，此时 Multi-Head Self-Attention 的输出大小为  【l*d，m】。

![multi-head-self-attention.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8c72da653dbe4b108e84efac27b22f35~tplv-k3u1fbpfcp-watermark.image)

## Stacked Self-Attention Layers
现在我们已经知道如何构造 Multi-Head Self-Attention ，接下来我们将它们堆叠产生一个深度神经网络。如下图所示我们在 Multi-Head Self-Attention 之上再加入一个全连接层，将 Multi-Head Self-Attention 产生的上下文特征向量 C 经过非线性变化转化成网络的输出特征 U ，每个时刻的全连接层都是相同的，共享参数矩阵 W<sub>U</sub> 。此时的每一个时刻的特征向量输出  u<sub>i</sub>  都与任意一个输入的 x 有关。当然对 u<sub>i</sub> 影响最大的还是 x<sub>i</sub>  。

![self-attention-dense.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c1ef409bdf0b4bb48da8156062181a66~tplv-k3u1fbpfcp-watermark.image)

我们在使用一个 Multi-Head Self-Attention 和一个 Dense 层的组建而成的神经网络组合得到输出 U ，我们还可以将 U 当作输入，在 U 之上再用同样的方法搭建一个相同的神经网络组合。重复继续进行下去就可以搭建一个深度的神经网络，道理和多层 RNN 一样。如下图所示。

![stacked-self-attention.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/10e1207436d846c1ad667f9c9fa5868b~tplv-k3u1fbpfcp-watermark.image)



## Transformer Encoder

想要搭建 Transformer 的 Encoder 结构，需要了解一个名叫 Block 的概念，如下图右边所示，一个 Block 层就是上面介绍过的使用一个 Multi-Head Self-Attention 和一个 Dense 层的组建而成的神经网络。假如输入的大小为【d，m】，d 是维度大小，m 是输入个数。那么 Block 的输出大小也是【d，m】。

如图中左边所示，Transformer 的 Encoder 结构中有 6 个 Block 搭建而成，输入与输出的大小是一样的，都是【d，m】。


![transformer-encoder.jpg](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/61ddc5969b204155b34381317b804c83~tplv-k3u1fbpfcp-watermark.image)



## Multi-Head Attention
和上面组成 Multi-Head Self-Attention 方法类似，我们可以用 Single-Head Attention 组成 Multi-Head Attention 。每一个 Single-Head Attention 的输入都是序列 X 和序列 X<sup>'</sup> ，每个  Single-Head Attention 都有各自的参数矩阵，不会互相共享参数，所以会有各自的上下文状态特征序列 C 。将这些 C 堆叠起来，就是 Multi-Head Attention 的输出。


![multi-head-attention.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5f9d74caab7047479099579c251c8fc8~tplv-k3u1fbpfcp-watermark.image)

## Stacked Attention Layers

由于 Transformer 是 Seq2Seq 模型，所以在 Seq2Seq 结构中，Encoder 的输入为长度是 m 的序列 X ，大小是【d，m】，Encoder 的结构就是上面介绍过的 6 层 Block 拼接而成的网络结构，输出为 U ，大小为【d，m】。Decoder 的输入为长度是 t 的序列 X<sup>'</sup> ，大小是【d，t】，经过一个 Multi-Head Self-Attention 输出为 C 。然后以 U 作为 Encoder 的输入，C 作为 Decoder 的输入，在两者之上搭建一个 Multi-Head Attention ，Multi-Head Attention 的输出为 Z ，其大小为 【d，t】。在 Z 上面再搭建一个 Dense 层，将 Z 进行非线性转换成为 S 。Dense 层中的权重参数是同一个 W<sub>S</sub> 。

![stacked-attentions.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/fcb10995048b4bf38463a95f081b1004~tplv-k3u1fbpfcp-watermark.image)

## Transformer Decoder 

上面搭建的三层神经网络包括 Multi-Head Self-Attention Layer、Multi-Head Attention Layer 和 Dense Layer 就是 Transformer 的 Decoder 部分的一个 Block 。

Block 的输入是两个序列，一个序列是大小为【d，m】的 Encoder 输入，一个序列是大小为【d，t】的 Decoder 输入，d 是维度大小，m 是 Encoder 输入个数，t 是 Decoder 输入个数。这个 Block 的输出大小为 【d，t】。

![Transformer-Decoder.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/88c96aaaed1c422082a0b78a528bab55~tplv-k3u1fbpfcp-watermark.image)



## Put Everything Together

现在知道了 Transformer 模型中的 Encoder 和 Decoder 结构，可以将两者组合在一块形成真正的 Transformer 。

如下图所示，左边是靠 6 层 Block 搭建而成的 Encoder 网络，每个 Block 有 Multi-Head Self-Attention Layer 和 Dense Layer ，输入序列 X 的大小是【d，m】，输出是【d，m】。

图右边是靠 6 层 Block 搭建而成的 Decoder 网络，每个 Block 有 Multi-Head Self-Attention Layer、 Multi-Head Attention Layer 和 Dense Layer ，最底层 Block 的输入是两个，一个输入是 Encoder 的输出，大小是【d，m】，另一个输入是 Decoder 的输入序列  X<sup>'</sup> ，大小是【d，t】。上面其他的 Block 的输入也是两个，一个输入仍是 Encoder 的输出，大小是【d，m】，另一个输入是下面 Block 的输出，大小是【d，t】。

![transformer.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/29672ea4273a4e78b3311a357ef414c7~tplv-k3u1fbpfcp-watermark.image)

## Summary
Transformer 的 Encoder 网络就是靠 6 个 Block 搭建而成的，每个 Block 中有 Multi-Head Self-Attention Layer 和 Dense Layer ，输入序列大小是【d，m】，输出序列大小是【d，m】。

![encoder.jpg](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0cf42e3b0f64424da2c7c5aeeb656ec8~tplv-k3u1fbpfcp-watermark.image)

Transformer 的 Decoder 网络也是靠 6 个 Block 搭建而成的，每个 Block 中有 Multi-Head Self-Attention Layer、 Multi-Head Attention Layer 和 Dense Layer ，输入序列有两个，一个序列大小是【d，m】，另一个序列大小为【d，t】，输出序列大小是【d，t】。

![decoder.jpg](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/19b6341a618c491992e844b0ced68694~tplv-k3u1fbpfcp-watermark.image)

Transformer 是 Seq2Seq 模型，包括了一个 Encoder 和一个 Decoder 。Transformer 不是 RNN 模型，它没有循环结构。Transformer 完全基于 Attention 和 Self-Attention 搭建而成。Transformer 的性能在业内极其出色。

![transformer-model.jpg](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/131572266e184928b84a53b899cd0c9a~tplv-k3u1fbpfcp-watermark.image)


## 参考

[1] Vaswani A , Shazeer N , Parmar N , et al. Attention Is All You Need[J]. arXiv, 2017.
