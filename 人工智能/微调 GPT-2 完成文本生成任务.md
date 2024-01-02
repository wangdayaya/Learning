

![GPT2.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5b49644f678b433c85edf4b36549b586~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1432&h=501&s=1325837&e=gif&f=53&b=fdfcfc)

# 原生 GPT-2 生成文本

我们使用 `kears_nlp` 中原生的 `GPT-2` 模型（`gpt2_base_en`），首先是指定我们的 `PROMPT` 是 `"My trip was"` ，也就是让 GPT-2 从这里开始文本生成，调用方式很简单，生成 `200 个` token 耗时 `22.81 s` ，速度大约 `8.77 token/s` 。
```
PROMPT = "My trip was"
ANSWER_START = "\nGPT-2 生成的文本如下："
preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset("gpt2_base_en", sequence_length=128)
gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en", preprocessor=preprocessor)
start = time.time()
output = gpt2_lm.generate(PROMPT, max_length=200)
print(f"{ANSWER_START}\n{output}\nTOTAL TIME ELAPSED: {time.time() - start:.2f}s")
```

日志打印生成的内容，我觉得效果差强人意吧，反正基本顺畅，但是你要是说内容有多好，其实也很一般：
```

GPT-2 生成的文本如下：
My trip was to a local bar, where I was greeted by a man in the front. He said that he had a friend who had a problem with drinking and had tried to stop him. I asked what he was drinking and he said, 'I'm a drunk man. I have been drinking for a week.' I asked if he had any questions or anything to say to me.

He was a man who was a bit of a dick. He said, 'If you want to get into a bar with me, just come to the bar and sit down. I'm going to have a good time.' He said I would have to leave the bar if I did that. He said that if I did it, he would be in a bar with me and I would get to see him.

I told him that he was a dick, but he told me to leave. He asked me if I wanted anything from me, and I said yes, I was going
```
另外我想尝试下中文的文本生成效果，但是找到了官方的 `kears_nlp` 可以调用的 `GPT-2` 模型，只有下面几种：

- gpt2_base_en
- gpt2_medium_en
- gpt2_large_en
- gpt2_extra_large_en
- gpt2_base_en_cnn_dailymail


全部都是英文数据训练出来的，我不太死心，还是想试试，所以调用 `gpt2_base_en` ，我也将 `PROMPT` 改成了 `我爱中国！` ，生成的结果简直就是乱七八糟，狗屁不通，自己把自己卡死掉了。如下：
```
# GPT-2 生成的文本如下：
# 我爱中国！拳探品的经和没有那与罗没有那格拳探品的品没有那格拳探品的品没有那格拳探品的经和没有那格拳探品的经和没有那格拳探品的那格拳探品的经和没有那格拳探品的
```



# 使用 Reddit 数据集进行微调

Reddit 数据集中每个样本中都包含一个 document 和 title ，document 是很长的文本记录了很多杂记，title 就是一个对应的标题。如下取了一个简略的样本：
```
b"me and a friend decided to go to the beach last sunday. we loaded up and headed out. we were about half way there when i decided that i was not leaving till i had seafood. \n\nnow i'm not talking about red lobster. no friends i'm talking about a low country boil. i found the restaurant and got directions. ...... , i got to the car and told him to get the hell out of there."
b'liking seafood'
```

微调代码如下，只选取了 500 条样本即可，因为 GPT-2 本身有 few-shot 甚至 zero-show 的功能，只需要少量的样本进行微调即可：

```
reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)
for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break
train_ds = reddit_ds.map(lambda d, _: d).batch(32).cache().prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.take(500)
num_epochs = 1
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(5e-5, decay_steps=train_ds.cardinality() * num_epochs, end_learning_rate=0.)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss, weighted_metrics=["accuracy"])
gpt2_lm.fit(train_ds, epochs=num_epochs)
```


微调训练过程如下：

    500/500 [==============================] - 128s 196ms/step - loss: 3.3047 - accuracy: 0.3263

微调之后再次使用同样的 PROMPT 进行文本生成，耗时 `9s` ，速度 `22 token/s`，相对于之前是有所提升的。另外很明显的变化就是生成的内容风格和 Reddit 数据集很像，说明微调的效果基本实现了：
```
    GPT-2 生成的文本如下：
    My trip was going fine. i had a few days off, but the trip was over. i was still on vacation with my family so i was pretty much in good shape. i decided to go out to the lake to see if there was any water. the lake was pretty cool, so i decided to head out and grab some ice cream. i was pretty hungry, so i went to grab some and i got the water. 

    the lake was a bit too cold for me to swim, so i was pretty hungry. i got a good chunk of ice cream and a little bit of ice cream
```

# 改变采样机制

我采用了 GreedySampler 采样器来进行 GPT-2 的文本生成，结果也是一塌糊涂，得到了一个无限循环的结果，可想而知选对采样器也是至关重要的。
```
greedy_sampler = keras_nlp.samplers.GreedySampler()
gpt2_lm.compile(sampler=greedy_sampler)
```
生成日志打印：
```
GPT-2 生成的文本如下：
My trip was to a local college, and i was in the process of getting my degree. i was in the process of getting my degree, and i was in the process of getting my license. i was in the process of getting my license, and i was in the process of getting my license. i was in the process of getting my license, and i was in the process of getting my license. i was in the process of getting my license, and i was in the process of getting my license. i was in the process of getting my license, and i was in the process of getting my license. i was in the process of getting my license, and i was in the process of getting my license. i was in the process of getting my license, and i was in the process of getting my license. i was in the process of getting my license, and i was in the process of getting my license.
```


# 用唐诗数据集微调

首先将诗词数据从 github 下载下来：

```
git clone https://github.com/chinese-poetry/chinese-poetry.git
```

使用 10000 条样本进行微调，训练日志如下：
```
    500/500 [==============================] - 157s 242ms/step - loss: 2.3250 - accuracy: 0.2879
```
最后使用 PROMPT 为 `春眠不觉晓` 进行诗词生成，耗时 `12.86s` 效果也算可以了，毕竟诗词的生成比普通文本的生成难多了，要对仗还要工整还要押韵：
```
GPT-2 生成的文本如下：
春眠不觉晓時暮，曾解風光滿處沙。白日黃沙深處渡，白霞濃淨暮晴沙。
```
总的来说，keras_nlp 中的可用的 GPT-2 模型列表都是基于英文数据训练的，想要在生成中文文本需要经过微调之后再使用，效果也仅仅是凑活用，想要真正的实现效果良好的中文内容生成，还是得要从底层开始使用大量的中文数据进行大模型的重新训练。

# 参考

- https://zhuanlan.zhihu.com/p/589980154