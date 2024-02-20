# 前文
本文展示了如何使用 LSTM 模型进行字符级别的文本生成过程，整个过程如果要达到文本通顺的程度至少需要 20 个 epoch ，以及至少 1M 字符的语料库，而且由于 RNN 网络计算量巨大所以建议在 GPU 上运行此脚本。

# 数据处理

- 下载 `nietzsche.txt` 文件，这个文件里面都是英文文章
- 对文本进行简单的处理，然后获取字符和字符索引之间的映射
- 接下来将文本转换为适合训练的张量表示。`x`是一个三维张量，形状为`(样本数, maxlen, 字符集大小)`，表示输入数据。`y`是一个二维张量，形状为`(样本数, 字符集大小)`，表示对应的标签。这里使用了one-hot编码，将字符表示为向量形式。
```
path = tf.keras.utils.get_file("nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt", )
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
x = np.zeros((len(sentences), maxlen, len(chars)), dtype="bool")
y = np.zeros((len(sentences), len(chars)), dtype="bool")
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
```

# 模型搭建


这段代码定义了一个基于 LSTM 的字符级别语言模型，并编译了该模型，这个模型的目标是接受固定长度的文本序列，然后预测下一个字符。在训练过程中，它会尝试最小化预测字符与实际字符之间的差距，以便能够准确地预测下一个字符。。

1.  我们定义了一个具有三个层的序列模型，第一层是是模型的输入层，它接受一个形状为`(maxlen, len(chars))`的输入张量，其中`maxlen`是之前定义的每个输入序列的长度，`len(chars)`是字符集的大小。第二层是一个 `LSTM`  层，具有 `128` 个隐藏单元。`LSTM` 是一种循环神经网络，适合处理序列数据。该层将输入序列转换为固定长度的向量表示。第三层是是一个全连接层，它的输出大小与字符集的大小相同。它使用 `softmax` 激活函数，将模型输出转换为每个字符的概率分布。
1.  这里使用了 `RMSprop` 优化器，并设置学习率为 `0.01`。
1.  损失函数使用了`CategoricalCrossentropy`，它是用于多类分类问题的常见损失函数。优化器使用了之前定义的 `RMSprop` 。编译模型后，模型就准备好进行训练了。



```
model = tf.keras.Sequential([tf.keras.Input(shape=(maxlen, len(chars))), tf.keras.layers.LSTM(128), tf.keras.layers.Dense(len(chars), activation="softmax")])
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss=tf.losses.CategoricalCrossentropy(), optimizer=optimizer)
```

# 训练
这段代码定义了一个用于从预测概率分布中采样下一个字符的函数。这在训练文本生成模型时非常有用，特别是当你想要增加一些随机性以生成更多样化的文本时。对预测概率进行调整，使用一个称为 `temperature` 的参数。这个温度参数可以控制从概率分布中进行采样时的“随机性”。较高的温度会导致更多的随机性，而较低的温度会导致更加确定性的采样。取对数是为了缩放预测概率值，这样就可以更好地控制温度的影响。
 
```
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
```

下面是训练过程，一共经历了 40 个 epoch ，并且使用了不同的 temperature 进行文本生成，每次生成的 seed 都是 `preface   supposing that truth is a woma` , 下面是训练过程中，每个 epoch
结束后的生成情况。
```
epochs = 40
batch_size = 1024
for epoch in range(epochs):
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print(f"epoch: {epoch}")
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(f"temperature: {temperature}")
        generated = ""
        sentence = text[0: 0 + maxlen]
        print(f"Generating with seed: {sentence}")
        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char
        print(f"GENERATED:{generated}")
```

    epoch: 0
    temperature: 0.2
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:n the soure the suseres the soustion of the sourtion the sulition the superity and the sourt and the seation and suselition suseres the sure the siching and suchilision the serestition the sustion the sulition the sulition the soustes the soust of the susted the selficion of the suching and the sulfor the sulfored and the sore the sour the sution the sulition the sourted the soustion of the superi
    temperature: 0.5
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:l his with susperes his evering the soust the restical soust, and with him pare the stered in the serfourt fus tome consertion--the suint, but seuture the seres the manke of the greamention the with as the soust of the stiticilisist of know enderally is the soreing not one sureaties in the one susticions as that in of the must of the with surality of the sustions, the suble pesition the maner his
    temperature: 1.0
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:l ty in wereademsophipile."--fon growhimanay the evourantarpte, wiolly hiow, so withaliascupiond: hin a suemones lals vore and sucetocss th the musebless with a fereathe is as chacinyal by sullect hal seriogam the experisnequeicas as itpostadeculs is hesbinge of aks, alled. hay the arssay yatity in hiscreatanesn alsopaticed spongrimann? prosey comunlly whe prilin suliutd canity but edversalicions
    temperature: 1.2
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:lityeves iqustill" korely, punty, wigh real thure wojkislly bo! its all whille ipjrlin of es, ho st othesm. ow s lilch, fur. dusens sor whithane rew, coupsnd: an they fuald=--ancthicbleman. whe itw. weld, hithe to gorcusss, is ra8ges.-maress.! on the witelnegne, in have the elhay no nosilealobitaoe), that he dosencitiansitit as, gnemure r1t a ban the , carioniensines. buing it n(turtisw, itsassh)
    196/196 [==============================] - 1s 6ms/step - loss: 1.7927

    ... 
    
    epoch: 20
    temperature: 0.2
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:n his present intellectual value in the same displies to the can no longer the sense of the soul of the sense of the sense of the sense of the sense of the present intellectual the sense of the sense of the world of the sense of the sense of the sense of the present and more process which it is the sense of the present into the sense and his self-dreadors and presents and according to the sense of
    temperature: 0.5
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:n in a condition of pleased by the consequently the souls the power agerest behinder to the present the tempt and have a condition of the latter and his own individual in a soul the existence of the very fell--who ward to the most delicate and the worst of the same dispositic facts, the consequently all the greater the secret and more personal erotess and love of the however have denings one exper
    temperature: 1.0
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:n, who had in the play even in a superiority. there even in the greek ambitatite contral of humpels must bassed, the only this point of persons with the world, here had to lines, essettingly ideas"--and interpocome in granes the alterative as to philosot rase and pure hands, the intellectual vest hard flounible had has be but with the being with allow antscoryory, and theereover_ who is not by fic
    temperature: 1.2
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:n, of life. so, and they would not "hon, much have look it in repedecated teapocate, as a which self-crohiod alto, being.=--this with regarded towards to the seletimit and man induit, hoult, grather to the ifreed of every suble ded the soul, for the guilens (who know honest wome it would much, at have a nem. it have beyoxhe "prenation?nersher's tendion--sase--nes=--adorts, however, i kafe in the s
    196/196 [==============================] - 1s 6ms/step - loss: 1.1712

    ...

    epoch: 39
    temperature: 0.2
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:n his similage the same with the proceed the probably are man in the same belief and probably believe in the same was not be the process in the probably more of the same with an instrument and probably the senses the same with the sense of the same with the same was not the same with the same with an innocent is the same with the same with the same belief and process in the senses to the same with
    temperature: 0.5
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:n his similation that it is a pleasure of the scientific share of him the temptation of the relation of the philosophers of the senses in the consequence to the man in the senses to the becomes and consequence and interrow who in the conditions of the more probably been and interpretation of a proceed of the same with any philosophy has not be that the conception of the "but that the word to the s
    temperature: 1.0
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:n his considered in a certain, until great retardes! in clayshers of nature; also, who has all this strength, belongiags does?  190. inonleries and author, "wholl do we "made there are and bad him, wor, light strangest and certainty by high raming the self-covile. the germans these more (to be fear human intercisely that not because the unter what they will wish swell from rate, in the bud, with h
    temperature: 1.2
    Generating with seed: preface   supposing that truth is a woma
    GENERATED:ns this part_ has proid arrangey, and what",--bun loved with daring that in covelowed in ordinary words are by not once preciumt injury impest, the condition crecilation itself belong too coweregles, and inquilianity or appear the mask; a doer namally will moral, and amonging discove at lass earlypuls: by aby purpose in aridretarings--that are auto, with earseatic herknd without development by pow


# 参考
https://github.com/wangdayaya/DP_2023/blob/main/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/Character-level%20text%20generation%20with%20LSTM.py
