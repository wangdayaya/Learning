# 前文介绍

本文构建了一个常见的深度学习模型，实现多标签文本分类，可以根据论文摘要的文本内容预测其所属的多个主题领域标签。

# 数据准备

原始数据中分为三列 `titles 、summaries 、terms` ，`titles` 是文章标题，`summaries` 是摘要内容，`terms` 是所属的标签列表，我们主要任务是通过判断 `summaries` 中的内容来预测所属的 `terms` 。所以数据处理的主要工作在  `summaries` 和 `terms` 两列。将每行的 summaries 和 terms 作为样本的（输入，标签）样本对。本次任务是多标签预测，每个摘要会有多个所属标签，所以我们要将标签转换为 multi-hot 的形式。

例如随机展示一个样本：

    Abstract: 'Graph convolutional networks produce good predictions of unlabeled samples\ndue to its transductive label propagation. Since samples have different\npredicted confidences, we take high-confidence predictions as pseudo labels to\nexpand the label set so that more samples are selected for updating models. We\npropose a new training method named as mutual teaching, i.e., we train dual\nmodels and let them teach each other during each batch. First, each network\nfeeds forward all samples and selects samples with high-confidence predictions.\nSecond, each model is updated by samples selected by its peer network. We view\nthe high-confidence predictions as useful knowledge, and the useful knowledge\nof one network teaches the peer network with model updating in each batch. In\nmutual teaching, the pseudo-label set of a network is from its peer network.\nSince we use the new strategy of network training, performance improves\nsignificantly. Extensive experimental results demonstrate that our method\nachieves superior performance over state-of-the-art methods under very low\nlabel rates.'
    Label: ['cs.CV' 'cs.LG' 'stat.ML']

经过处理，标签列表会变成一个数据集中所有标签集合大小的数组，将该样本出现的标签对应的索引位置变成 1 ，其余位置变成 0 ，具体处理过程见代码：

    Abstract: 'Visual saliency is a fundamental problem in both cognitive and computational\nsciences, including computer vision. In this CVPR 2015 paper, we discover that\na high-quality visual saliency model can be trained with multiscale features\nextracted using a popular deep learning architecture, convolutional neural\nnetworks (CNNs), which have had many successes in visual recognition tasks. For\nlearning such saliency models, we introduce a neural network architecture,\nwhich has fully connected layers on top of CNNs responsible for extracting\nfeatures at three different scales. We then propose a refinement method to\nenhance the spatial coherence of our saliency results. Finally, aggregating\nmultiple saliency maps computed for different levels of image segmentation can\nfurther boost the performance, yielding saliency maps better than those\ngenerated from a single segmentation. To promote further research and\nevaluation of visual saliency models, we also construct a new large database of\n4447 challenging images and their pixelwise saliency annotation. Experimental\nresults demonstrate that our proposed method is capable of achieving\nstate-of-the-art performance on all public benchmarks, improving the F-Measure\nby 5.0% and 13.2% respectively on the MSRA-B dataset and our new dataset\n(HKU-IS), and lowering the mean absolute error by 5.7% and 35.1% respectively\non these two datasets.'
    Label: [0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]







# 模型训练

模型结构主要由以下三部分组成：

1. `layers.Dense(512, activation="relu")` 将输入映射为 512 维的向量，并且使用 relu 激活函数进行非线性运算
2. `layers.Dense(256, activation="relu")` 将输入映射为 256 维的向量，并且使用 relu 激活函数进行非线性运算
3. `layers.Dense(lookup.vocabulary_size(), activation="sigmoid")` 输出词典大小维度的向量，并且使用 sigmoid 激活函数推断所属标签概率
4. 编译模型时候，使用 `binary_crossentropy` 作为损失函数，使用 `adam` 作为优化器，使用 `binary_accuracy` 作为观测指标


训练过程日志打印如下：
```
    Epoch 1/20
    258/258 [==============================] - 8s 25ms/step - loss: 0.0334 - binary_accuracy: 0.9890 - val_loss: 0.0190 - val_binary_accuracy: 0.9941
    Epoch 2/20
    258/258 [==============================] - 6s 25ms/step - loss: 0.0031 - binary_accuracy: 0.9991 - val_loss: 0.0262 - val_binary_accuracy: 0.9938
    ...
    Epoch 20/20
    258/258 [==============================] - 6s 24ms/step - loss: 7.4884e-04 - binary_accuracy: 0.9998 - val_loss: 0.0550 - val_binary_accuracy: 0.9931
    15/15 [==============================] - 1s 28ms/step - loss: 0.0552 - binary_accuracy: 0.9932
```

将训练过程产生的损失值和准确率进行了绘制，如下所示：
    
![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ce6f7bba2d474bec9b716533431e46a1~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=32868&e=png&b=fefefe)

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/022089ac780546518858eeaf22221e65~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=35720&e=png&b=fefefe)

# 测试效果

随机选取两个样本，使用训练好的模型进行标签预测，为每个样本最多预测 3 个概率最高的标签，并和原始标签进行对比，可以发现基本上所属的标签都会出现在预测结果的前几个。

```
    Abstract: b'Graph representation learning is a fundamental problem for modeling\nrelational data and benefits a number of downstream applications. ..., The source code is available at\nhttps://github.com/upperr/DLSM.'
    Label: ['cs.LG' 'stat.ML']
    Predicted Label(s): (cs.LG, stat.ML, cs.AI) 
    Abstract: b'In recent years, there has been a rapid progress in solving the binary\nproblems in computer vision, ..., The SEE algorithm is split into 2 parts, SEE-Pre for\npreprocessing and SEE-Post pour postprocessing.'
    Label: ['cs.CV']
    Predicted Label(s): (cs.CV, I.4.9, cs.LG) 
```

# 参考

- https://github.com/wangdayaya/DP_2023/blob/main/NLP%20%E6%96%87%E7%AB%A0/Large-scale%20multi-label%20text%20classification.py
- https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv
