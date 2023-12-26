# 文章介绍

本文主要介绍了使用 FNet 模型完成 IMDB 电影评论的情感分类任务，并且和传统的 Transformer 模型进行性能比较。

FNet 模型的出现是为了解决传统的 Transformer 模型计算时间复杂度过高的问题。它主要有以下特点：

1. **使用傅立叶变换代替自注意力机制**：相比于传统的自注意力机制，FNet 使用傅立叶变换来捕捉序列中的长距离依赖关系，从而提高了处理长序列的效率。

2. **轻量级设计**：FNet 具有相对轻量级的结构，使其在处理大规模序列数据时更加高效，同时也减少了计算成本，时间复杂度从 O(n^2) 降低到了 O(nlogn) 。

3. **模型训练速度提升**：FNet 作者声称 FNet 在 GPU 上训练速度比 BERT 快 80% ，在 TPU 上的训练速度快 70% 。

4. **损失较小的准确率**：FNet 作者声称为了提升训练速度，在准确率方面会有较小的牺牲，但是在 GLUE 基准测试中可以达到 BERT 准确率 92-97% 的效果。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/09f32118a8e94615a8d4d0db56cd9960~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=627&h=565&s=44429&e=png&b=fbf8f8)



# 数据处理

我们这里使用 IMDB 的数据，其中每个样本都有一个电影评论文本和情感标签，我们就是要把评论文本数据处理先进行分词，然后经过整数化变成模型输入需要的整数数组。如下举例所示，Sentence 是一条评论数据，Tokens 是经过分词和整数映射的结果，符合模型的输入要求，具体处理过程看最后的代码。

    Sentence:  tf.Tensor(b'this picture seemed way to slanted, it\'s almost as bad as the drum beating of the right wing kooks who say everything is rosy in iraq. it paints a picture so unredeemable that i can\'t help but wonder about it\'s legitimacy and bias. also it seemed to meander from being about the murderous carnage of our troops to the lack of health care in the states for ptsd. to me the subject matter seemed confused, it only cared about portraying the military in a bad light, as a) an organzation that uses mind control to turn ordinary peace loving civilians into baby killers and b) an organization that once having used and spent the bodies of it\'s soldiers then discards them to the despotic bureacracy of the v.a. this is a legitimate argument, but felt off topic for me, almost like a movie in and of itself. i felt that "the war tapes" and "blood of my brother" were much more fair and let the viewer draw some conclusions of their own rather than be beaten over the head with the film makers viewpoint. f-', shape=(), dtype=string)
    Tokens:,tf.Tensor(,[,145,576,608,228,140,58,13343,13,143,8,58,360,,148,209,148,137,9759,3681,139,137,344,3276,50,12092,,164,169,269,424,141,57,2093,292,144,5115,15,143,,7890,40,576,170,2970,2459,2412,10452,146,48,184,8,,59,478,152,733,177,143,8,58,4060,8069,13355,138,,8557,15,214,143,608,140,526,2121,171,247,177,137,,4726,7336,139,395,4985,140,137,711,139,3959,597,144,,137,1844,149,55,1175,288,15,140,203,137,1009,686,,608,1701,13,143,197,3979,177,2514,137,1442,144,40,,209,776,13,148,40,10,168,14198,13928,146,1260,470,,1300,140,604,2118,2836,1873,9991,217,1006,2318,138,41,,10,168,8469,146,422,400,480,138,1213,137,2541,139,,143,8,58,1487,227,4319,10720,229,140,137,6310,8532,,862,41,2215,6547,10768,139,137,61,15,40,15,145,,141,40,7738,4120,13,152,569,260,3297,149,203,13,,360,172,40,150,144,138,139,561,15,48,569,146,,3,137,466,6192,3,138,3,665,139,193,707,3,,204,207,185,1447,138,417,137,643,2731,182,8421,139,,199,342,385,206,161,3920,253,137,566,151,137,153,,1340,8845,15,45,14,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0,0,0,0,0,,0,0,0,0,0,0,0,0],shape=(512,),dtype=int32)




# 模型搭建




1. `input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")`：定义了输入层，用于接受整数类型的输入数据，形状为 (None, )，即任意长度的一维张量。

2. `keras_nlp.layers.TokenAndPositionEmbedding`：该层用于将输入的整数序列转换为其对应的嵌入向量序列。 

3. `keras_nlp.layers.FNetEncoder`：FNet 编码器层，它接受嵌入向量序列作为输入，并对序列进行编码处理。这里使用了 3 个相同的 FNet 编码器层。

4. `keras.layers.GlobalAveragePooling1D()`：对输入数据的时间维度进行全局平均池化操作，将每个样本的所有时间步上的特征进行平均。

5. `keras.layers.Dropout(0.1)`：对全局平均池化后的数据进行 Dropout 操作，以防止过拟合。

6. `keras.layers.Dense(1, activation="sigmoid")`：全连接层将池化后的数据映射到输出层，采用 Sigmoid 激活函数来处理二分类问题。

7. `fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")`：将定义的各层组合成一个完整的模型。输入为`input_ids`，输出为`outputs`。

 



# 编译测试

这里主要是定义优化器为 Adam ，损失函数为 binary_crossentropy ，检测指标为 accuracy 。

```
fnet_classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)
```

日志打印，验证集准确率能达到 0.8688 ，测试集准确率能达到 0.8567 ：

```
# Epoch 1/3
# 79/79 [==============================] - 18s 173ms/step - loss: 0.7111 - accuracy: 0.5031 - val_loss: 0.6890 - val_accuracy: 0.6062
# Epoch 2/3
# 79/79 [==============================] - 13s 165ms/step - loss: 0.5600 - accuracy: 0.6812 - val_loss: 0.3558 - val_accuracy: 0.8456
# Epoch 3/3
# 79/79 [==============================] - 13s 162ms/step - loss: 0.2791 - accuracy: 0.8856 - val_loss: 0.3220 - val_accuracy: 0.8688
# 98/98 [==============================] - 10s 73ms/step - loss: 0.3328 - accuracy: 0.8567
```

# 对比模型

将上面的模型结构中的三个  `keras_nlp.layers.FNetEncoder` 都换成 `keras_nlp.layers.TransformerEncoder` ，然后重新编译并进行训练，日志打印如下，，验证集准确率能达到 0.8756 ，测试集准确率能达到 0.8616 ：

```
# Epoch 1/3
# 79/79 [==============================] - 15s 139ms/step - loss: 0.7204 - accuracy: 0.5856 - val_loss: 0.4003 - val_accuracy: 0.8204
# Epoch 2/3
# 79/79 [==============================] - 10s 127ms/step - loss: 0.2679 - accuracy: 0.8896 - val_loss: 0.3165 - val_accuracy: 0.8804
# Epoch 3/3
# 79/79 [==============================] - 10s 127ms/step - loss: 0.1992 - accuracy: 0.9230 - val_loss: 0.3142 - val_accuracy: 0.8756
# 98/98 [==============================] - 8s 47ms/step - loss: 0.3437 - accuracy: 0.8616
```


# 存在问题

我们将两个模型的各项指标列举出来，如下所示：


| 指标 | FNet | Transformer|
| --- | --- | --- |
| 参数量 | 2,382,337  | 2,580,481|
| 测试集准确率 |  0.8567  |  0.8616  |
| 训练时间 | 44s | 35s |

我们可以看到 FNet 的参数量确实比 Transformer 要少，准确率也有 0.0049 的损失，这两个测试结果和论文中结论基本一致，但是在训练时间方面 FNet 没有比 Transformer 显著的减少，反而有大幅度的增加，这个是我无法理解的，我用的显卡是单机的 4090 ，实在不知道为啥，我猜测可能是由于模型结构太小了，或者是超参数选择有问题，或者是硬件环境和论文不同。

# 参考

https://github.com/wangdayaya/DP_2023/blob/main/NLP%20%E6%96%87%E7%AB%A0/Text%20Classification%20using%20FNet.py