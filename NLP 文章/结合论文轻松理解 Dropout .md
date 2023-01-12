这是我参与更文挑战的第19天，活动详情查看： [更文挑战](https://juejin.cn/post/6967194882926444557)

## Dropout 是玄学？
在大致拜读过 [《Improving neural networks by preventing co-adaptation of feature detectors》](https://arxiv.org/pdf/1207.0580.pdf) 这篇著名的论文的之后，我才知道原来 Hinton 大神写的论文也可以这么水。这篇论文只给出了 Dropout 经过不同实验的大量测试结果，但是究其原理却是给出了三种猜测，我也是醉了，所以说这个 Dropout 理解起来并不轻松【狗头】，是个比较玄学的东西，你得细品，只可意会不可言传。实际应用当中，Dropout 用的好能超神，用不好能拉胯，所以是一个训练模型技巧的可选项。但是从实际情况总体来说，确实能达到一定的性能提升的效果，吐槽结束，正文开始。

## 什么是 Dropout 
要想介绍 Dropout ，要先从机器学习或者深度学习常见的过拟合问题开始说起，当我们的训练数据集比较小的时候，模型参数多的时候，模型的过拟合现象经常发生，测试结果会表现地很差。而 Dropout 就是解决过拟合的一种常用的方法。至于有人说 Dropout 能解决耗时的问题，我不确定，理由下文会解释。

 Dropout 作为一种常用的解决过拟合的技巧，就是在训练过程中按照预先设置的概率，随机地丢弃一部分神经元，这种做法会大大降低过拟合现象，提升模型的泛化能力。其实论文中的摘要就已经对 Dropout 的产生原因和定义都解释了，上原文摘要：

When a large feedforward neural network is trained on a small training set, it typically performs poorly on held-out test data. This “overfitting” is greatly reduced by randomly omitting half of the feature detectors on each training case. 


![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fwww.sosoba.org%2Fdata%2Fattachment%2Fforum%2F201804%2F11%2F132923xgdpp3gtofzrizuu.jpg&refer=http%3A%2F%2Fwww.sosoba.org&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1626662377&t=a790415696b3ae76c0c6a5607665466b)


这句话其实已经把论文主要内容说完了，其他的长篇大论只是在介绍在没有用到 Dropout 和用到 Dropout 之后的测试效果对比。

## Dropout 具体流程
【训练阶段】

a）预先设置保留神经元的概率为 p ，丢弃神经元的概率为 1-p ，p 越高意味着更多的神经元被保留。从代码上表现出来的就是，让某个神经元的输出值以概率 1-p 变为 0 。

b）假如某层网络神经元的个数为 10 个，其激活函数输出值为 y<sub>1</sub>、y<sub>2</sub>、y<sub>3</sub>、...、y<sub>10</sub>，我们 p 设置为 0.6 ，那么这一层神经元经过 dropout 后，大约 4 个神经元的值被置为 0 。

c） 尽管可能丢弃掉了 4 个神经元，但是我们还需要对输出值 y1……y10 进行缩放，也就是乘以 **1/p**。原因可能是因为假如某个神经元的正常的输出是 **x** ，在加入 Dropout 之后期望值变为了 **E = p\*x + (1-p)\*0=p\*x** ，然后 **E\*1/p=x**，保证了输出期望值不会变。

【测试阶段】

在训练时由于丢弃了一部分神经元，因此在测试时需要在神经元输出结果乘 **p** 进行缩放。但是这样增加了测试的耗时。通常为了减少测试时的运算时间，可以将缩放的工作转移到训练阶段，正如上面的 c ）过程，这样测试时不用再缩小权重。但是在测试的时候需要将权重参数 **w** 乘 **p** ，假如某个神经元正常输出是 **x** ，那么 dropout 之后的期望值是 **E = p\*x + (1-p)\*0=p\*x** 。在测试时该神经元一直存在，正常输入为 **x** ，但是为了保持同样的输出期望值，需要调整 **x** 为 **px** 。  

![](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/91acb661840b450392facbd1dd338016~tplv-k3u1fbpfcp-zoom-1.image)

简单一句话总结就是，在训练时神经元有 **p** 大小的概率存在，所以每个神经元的输出都除 **p** ；在测试阶段，每个神经单元都是存在的，权重参数 **w** 要乘以 **p** 成为 **pw**。

## Dropout 为何能解决过拟合
 
论文中也没有给出明确的解释，可能作者也说不清楚，只是有三种说法可以参考，帮助理解内部原理，反正不管哪一种你自己细品吧，到底是对还是不对呢，你接着细细品。

【第一种说法】通过 Bagging 方式随机抽取不同的训练数据，训练不同的决策树以构成随机森林，采用投票方式完成决策任务。 而 Dropout 可以被视为一种极端形式的 Bagging ，设置丢弃神经元的概率之后，每次训练数据的时候都相当于是在训练一个新的模型，并且模型的每个参数都与所有其他模型中的相应参数共享来强化正则能力。

【第二种说法】Dropout 是“朴素贝叶斯”的极端情况，其中每个输入特征被单独训练以预测类别标签，然后在测试时将所有特征的预测分布相乘。当训练数据很少时，这通常比逻辑分类要好得多，后者训练每个输入特征在所有其他特征的上下文中都能很好地工作。

【第三种说法】物种为了生存往往会倾向于适应这种环境，环境突变则会导致物种难以做出及时反应，性别的出现可以繁衍出适应新环境的变种，有效的阻止过拟合，即避免环境改变时物种可能面临的灭绝。


上原文片段：

A popular alternative to Bayesian model averaging is “bagging” in which different models are trained on different random selections of cases from the training set and all models are given equal weight in the combination. Bagging is most often used with models such as decision trees because these are very quick to fit to data and very quick at test time . Dropout allows a similar approach to be applied to feedforward neural networks which are much more powerful models. Dropout can be seen as an extreme form of bagging in which each model is trained on a single case and each parameter of the model is very strongly regularized by sharing it with the corresponding parameter in all the other models. This is a much better regularizer than the standard method of shrinking parameters towards zero.

A familiar and extreme case of dropout is “naive bayes” in which each input feature is trained separately to predict the class label and then the predictive distributions of all the features are multiplied together at test time. When there is very little training data, this often works much better than logistic classification which trains each input feature to work well in the context of
all the other features.

Finally, there is an intriguing similarity between dropout and a recent theory of the role of sex in evolution. One possible interpretation of the theory of mixability articulated in is that sex breaks up sets of co-adapted genes and this means that achieving a function by using a large set of co-adapted genes is not nearly as robust as achieving the same function, perhaps less than optimally, in multiple alternative ways, each of which only uses a small number of co-adapted genes. This allows evolution to avoid dead-ends in which improvements in fitness require co- ordinated changes to a large number of co-adapted genes. It also reduces the probability that small changes in the environment will cause large decreases in fitness a phenomenon which is known as “overfitting” in the field of machine learning.



## Dropout 用到哪里
既可以对输入数据的 Dropout ，又能对隐层的 Dropout 。看原文描述：

Without using any of these tricks, the best published result for a standard feedforward neural network is 160 errors on the test set. This can be reduced to about 130 errors by using 50% dropout with separate L2 constraints on the incoming weights of each hidden unit and further reduced to about 110 errors by also dropping out a random 20% of the pixels 。

这里看出作者在实验过程中不仅将 Dropout 用到了隐层，还将输入的图片像素随机地进行了 20% 的 Dropout 。论文中的类似案例还有不少。
 


## Dropout 测试效果

这里截取了论文中使用 MNIST 数据集进行测试性能对比的图，可以看出对输入进行 20% 的 Dropout 和对隐层进行 50% 的 Dropout ，效果比单纯对隐层进行 50% 的 Dropout 要好。但是从图中我们也可以看出来训练的 Epoch 太多了，正常情况下不可能对数据集进行这么多的 Epoch ，所以我从这里觉得 Dropout 能够缩小训练时间是持怀疑态度的。而且在丢弃了部分神经元网络模型中，想要学习到合适的权重参数，不得不进行大量的训练。


![dropout 测试结果.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ed7718edf0b0444d8b5376952df1d84a~tplv-k3u1fbpfcp-watermark.image)
 

## 参考
* [Hinton G E ,  Srivastava N ,  Krizhevsky A , et al. Improving neural networks by preventing co-adaptation of feature detectors[J]. Computer Science, 2012, 3(4):págs. 212-223.](https://arxiv.org/pdf/1207.0580.pdf)
* https://blog.csdn.net/program_developer/article/details/80737724
* https://www.cnblogs.com/makefile/p/dropout.html
