# 前言

本文主要介绍了文本问答的实现过程，简单来说就是输入“文本+问题”，返回“答案在文本的起始索引和终止索引”。

# 数据处理

我们使用到的是经典的 SQuAD (Stanford Question-Answering Dataset) ，里面有很多问答数据，我们要做的就是把这里面的问答对转化成模型 BERT 需要的输入。每个样本的处理过程都一样，下面我通过一个样本介绍具体的处理步骤，具体实现过程详见文末的代码链接。

1. `问答对数据`：
    -  `context `：Architecturally, ...... , France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
    -  `question `：To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
    -  `answer `：Saint Bernadette Soubirous

分析分析
2. `分词`：
    - `context`：['[CLS]', 'architectural', '##ly', ',', ...... ,'saint', 'bern', '##ade', '##tte', 'so', '##ub', '##iro', '##us', 'in', '1858', '.', 'at', 'the', 'end', 'of', 'the', 'main', 'drive', '(', 'and', 'in', 'a', 'direct', 'line', 'that', 'connects', 'through', '3', 'statues', 'and', 'the', 'gold', 'dome', ')', ',', 'is', 'a', 'simple', ',', 'modern', 'stone', 'statue', 'of', 'mary', '.', '[SEP]']
    - `question`：['[CLS]', 'to', 'whom', 'did', 'the', 'virgin', 'mary', 'allegedly', 'appear', 'in', '1858', 'in', 'lou', '##rdes', 'france', '?', '[SEP]']
    - `answer`：['saint', 'bern', '##ade', '##tte', 'so', '##ub', '##iro', '##us']

3. `词典映射`：
    - `tokenized_context`：[101, 6549, 2135, 1010, ...... , 1997, 1996, 2364, 3298, 1006, 1998, 1999, 1037, 3622, 2240, 2008, 8539, 2083, 1017, 11342, 1998, 1996, 2751, 8514, 1007, 1010, 2003, 1037, 3722, 1010, 2715, 2962, 6231, 1997, 2984, 1012, 102]
    - `tokenized_question`：[101, 2000, 3183, 2106, 1996, 6261, 2984, 9382, 3711, 1999, 8517, 1999, 10223, 26371, 2605, 1029, 102]
    - `ans_token_idx`：[114, 115, 116, 117, 118, 119, 120, 121]

4. `BERT 输入`：
    - `input_ids` ： tokenized_context + tokenized_question[1:] + ([0] * padding_length)
    - `token_type_ids` ： [0] * len(tokenized_context) + [1] * len(tokenized_question[1:])   + ([0] * padding_length)
    - `attention_mask` ： [1] * len(input_ids)  + ([0] * padding_length)

5. `BERT 标签`：
    - `start_token_idx` : ans_token_idx[0]
    - `end_token_idx` : ans_token_idx[-1]



 

# 模型结构

这里定义了一个基于 BERT 模型的问答模型，该模型可以根据上下文和问题，预测出答案在上下文的起始和结束位置的概率分布，具体实现过程详见文末的代码链接。


1.  `定义模型的输入`：这里定义了三个输入张量，分别对应 BERT 模型的 input_ids 、token_type_ids 和 attention_mask ，并且得到了 embedding 。
   ```python
   encoder = TFBertModel.from_pretrained("bert-base-uncased")
   input_ids = Input(shape=(max_len,), dtype=tf.int32)
   token_type_ids = Input(shape=(max_len,), dtype=tf.int32)
   attention_mask = Input(shape=(max_len,), dtype=tf.int32)
   embedding = encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
   ```
 

2. `构建问题答案起始位置的输出层`：这里使用一个全连接层来预测答案的起始位置，并通过 `Flatten` 层将输出压成长度为 max_len 的 logits 。
   ```python
   start_logits = Dense(1, name="start_logit", use_bias=False)(embedding)
   start_logits = Flatten()(start_logits)
   ```
   

3. `构建问题答案结束位置的输出层`：这里使用一个全连接层来预测答案的结束位置，并通过 `Flatten` 层将输出压成长度为 max_len 的  logits 。
   ```python
   end_logits = Dense(1, name="end_logit", use_bias=False)(embedding)
   end_logits = Flatten()(end_logits)
   ```

4. `添加激活函数`：这里使用 softmax 激活函数对起始和结束位置的模型输出 logits 进行转换，得到最终的 0-1 之间的概率分布。
   ```python
   start_probs = Activation(keras.activations.softmax)(start_logits)
   end_probs = Activation(keras.activations.softmax)(end_logits)
   ```
   

5. `构建整体模型`：这里定义了整个模型，输入是 BERT 模型的输入，输出是答案的起始位置和结束位置的 softmax 概率分布。
   ```python
   model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=[start_probs, end_probs])
   ```

# 模型训练

整个模型的编译很简单，损失函数使用的是最常见的 SparseCategoricalCrossentropy ，优化器使用的是最常见的 Adam ，需要注意的是因为我们要预测两个向量分布，所以损失函数需要两个 loss 。

训练的时候我们这里定义了两个回调函数，ModelCheckpoint 用于在经过每次 epoch 之后保存最佳模型，ExactMath 用于计算验证集的准确率。
```
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=[loss, loss])
model.fit(x_train, y_train, epochs=1, verbose=1, batch_size=16, callbacks=[keras.callbacks.ModelCheckpoint('text_extraction', monitor='loss', save_weights_only=True), ExactMath(x_eval, y_eval, eval_squad_examples)])
```

日志打印，可以看到答案完全准确率达到 78% 左右：

    Epoch 1/5
    5384/5384 [==============================] - 1356s 250ms/step - loss: 2.4628 - activation_loss: 1.2933 - activation_1_loss: 1.1695
    323/323 [==============================] - 46s 136ms/step
    epoch=1, exact match score=0.78
    Epoch 2/5
    5384/5384 [==============================] - 1301s 242ms/step - loss: 1.5804 - activation_loss: 0.8393 - activation_1_loss: 0.7411
    323/323 [==============================] - 44s 136ms/step
    epoch=2, exact match score=0.77
    Epoch 3/5
    5384/5384 [==============================] - 1301s 242ms/step - loss: 1.1446 - activation_loss: 0.6128 - activation_1_loss: 0.5317
    323/323 [==============================] - 45s 137ms/step
    epoch=3, exact match score=0.77

# 效果展示
随便找了三个样本，放入模型中进行推理预测，并将结果进行处理，得到如下结果，可以看出来，预测结果符合预期，都存在于真实答案集合之中。

    预测答案 socialist realism , 真实答案集合 ['socialist realism', 'socialist realism', 'socialist realism']
    预测答案 warsaw citadel , 真实答案集合 ['warsaw citadel', 'warsaw citadel', 'warsaw citadel']
    预测答案 green , 真实答案集合 ['green', 'green', 'green']

# 参考
https://github.com/wangdayaya/DP_2023/blob/main/NLP%20%E6%96%87%E7%AB%A0/Text%20Extraction%20with%20BERT.py
