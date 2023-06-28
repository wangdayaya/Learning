# 前文
本文主要展示了如何实现微型的 GPT 模型完成文本生成任务，该模型只由 1 个 Transformer 块组成。

# Data

这部分代码主要用于准备文本数据集进行语言模型训练，这里需要事先下载好[ aclImdb 数据](
https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)，并且解压到当前目录。

首先，定义了一个批量大小 `batch_size` 和一个存储文件名的列表 `filenames`，其中包含了要处理的文本文件的路径。接下来，通过随机打乱 `filenames` 列表的顺序，来增加数据集的随机性。接着使用 TensorFlow 的 `TextLineDataset` 创建一个文本数据集 `text_ds`，并通过 `shuffle` 方法对数据集进行洗牌。

定义了一个自定义的标准化函数 `custom_standardization`，用于对输入字符串进行标准化处理。函数将字符串转换为小写，并使用正则表达式去除 HTML 标签和标点符号。然后创建了一个 `TextVectorization` 层，通过自定义函数对文本数据进行处理，对 `text_ds` 中的文本进行矢量化处理，也就是将文本转换为整数序列。并从数据集中自动构建词汇表。

通过 `map` 方法将处理后的文本转换为模型的输入和标签，即将每个序列中的一句话去掉最后一个字作为输入，然后将对应的同样一个序列从第二个字开始到最后的序列作为标签。从局部来看也就是前一个字是输入，预测输出后一个字。

最后，使用 `prefetch` 方法对数据集进行预取操作，以便在模型训练过程中能够高效地加载数据。


    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers import TextVectorization
    import numpy as np
    import os
    import string
    import random
    
    batch_size = 128
    filenames = []
    directories = [ "aclImdb/train/pos",  "aclImdb/train/neg", "aclImdb/test/pos", "aclImdb/test/neg",]
    for dir in directories:
        for f in os.listdir(dir):
            filenames.append(os.path.join(dir, f))

    random.shuffle(filenames)
    text_ds = tf.data.TextLineDataset(filenames)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)

    def custom_standardization(input_string):
        lowercased = tf.strings.lower(input_string)
        stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
        return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")

    vectorize_layer = TextVectorization( standardize=custom_standardization, max_tokens=vocab_size - 1, output_mode="int", output_sequence_length=maxlen + 1, )
    vectorize_layer.adapt(text_ds)
    vocab = vectorize_layer.get_vocabulary()   

    def prepare_lm_inputs_labels(text):
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y

    text_ds = text_ds.map(prepare_lm_inputs_labels)
    text_ds = text_ds.prefetch(tf.data.AUTOTUNE)
    
# Miniature GPT

##  Transformer Block

这里主要是定义了一个 `TransformerBlock` 类，该类实现了 Transformer 模型中的一个 Transformer Block 。整个 `TransformerBlock` 类的作用是将输入序列经过自注意力计算和前馈网络变换，得到一个更丰富的表示。

`TransformerBlock` 类的构造函数 `__init__` 接受四个参数：`embed_dim` 表示嵌入维度，`num_heads` 表示注意力头数，`ff_dim` 表示前馈网络的维度，`rate` 表示 Dropout 的比例。

在 `call` 方法中，首先获取输入的形状信息，包括批大小和序列长度。然后调用 `causal_attention_mask` 函数生成一个注意力掩码，用于遮蔽 Transformer 中的未来信息，确保模型只能看到当前位置以及之前的输入信息。这个掩码是一个二维矩阵，维度为 (seq_len, seq_len)。

接下来，使用 `MultiHeadAttention` 层 `self.att` 对输入进行自注意力计算，并传入注意力掩码。然后应用第一个 Dropout 层 `self.dropout1` 对注意力输出进行随机失活。将输入和注意力输出相加，并通过 LayerNormalization 层 `self.layernorm1` 进行归一化处理，得到第一个子层的输出 `out1`。

接着，将第一个子层的输出 `out1` 传入前馈神经网络 `self.ffn` 进行非线性变换。再次应用 Dropout 层 `self.dropout2` 对前馈网络的输出进行随机失活。将第一个子层的输出 `out1` 和前馈网络的输出相加，并通过 LayerNormalization 层 `self.layernorm2` 进行归一化处理，得到 Transformer Block 的最终输出。


    def causal_attention_mask(batch_size, n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat( [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0 )
        return tf.tile(mask, mult)

    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super().__init__()
            self.att = layers.MultiHeadAttention(num_heads, embed_dim)
            self.ffn = keras.Sequential( [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),] )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)

        def call(self, inputs):
            input_shape = tf.shape(inputs)
            batch_size = input_shape[0]
            seq_len = input_shape[1]
            causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
            attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
            attention_output = self.dropout1(attention_output)
            out1 = self.layernorm1(inputs + attention_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output)
            return self.layernorm2(out1 + ffn_output)

## Token And Position Embedding

这里定义了一个 `TokenAndPositionEmbedding` 类，用于得到输入序列的 token 和位置信息的嵌入。

`TokenAndPositionEmbedding` 类的构造函数 `__init__` 接受三个参数：`maxlen` 表示序列的最大长度，`vocab_size` 表示词汇表的大小，`embed_dim` 表示嵌入维度。

在 `call` 方法中，首先获取输入序列 `x` 的长度 `maxlen`。然后使用 `tf.range` 函数生成一个从 0 到 `maxlen-1` 的位置向量 `positions`。接着将位置向量 `positions` 传入位置嵌入层 `self.pos_emb` 进行嵌入，得到位置嵌入张量。同时，将输入序列 `x` 传入标记嵌入层 `self.token_emb` 进行嵌入，得到 token 的嵌入张量。最后，将 token 嵌入张量和位置嵌入张量相加，得到融合了标记和位置信息的嵌入张量，并将其作为输出返回。

整个 `TokenAndPositionEmbedding` 类的作用是将输入序列的 token 和位置信息进行嵌入计算，为后续的 Transformer 模型提供丰富的输入表示。在 Transformer 模型中，token 嵌入用于表示每个输入 token 的语义信息，而位置嵌入用于表示每个输入 token 在序列中的位置信息。


    class TokenAndPositionEmbedding(layers.Layer):
        def __init__(self, maxlen, vocab_size, embed_dim):
            super().__init__()
            self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
            self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

        def call(self, x):
            maxlen = tf.shape(x)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            positions = self.pos_emb(positions)
            x = self.token_emb(x)
            return x + positions
            
## Create Model 

这里定义了一个 `create_model` 函数，用于创建一个 Transformer 模型。这个模型使用 Transformer 架构来处理输入序列，并在最后通过全连接层进行分类预测。它能够学习输入序列中的语义和上下文关系，用于生成预测的单词概率分布。

函数中首先创建了一个输入层 `inputs`，其形状为 `(maxlen,)`，数据类型为 `tf.int32`，用于接收输入序列。接下来定义了一个 `TokenAndPositionEmbedding` 层，传入参数 `maxlen`、`vocab_size` 和 `embed_dim`，用于将输入序列的标记和位置信息进行嵌入。将输入层 `inputs` 作为输入传递给嵌入层，得到嵌入后的输出张量 `x`。

然后创建了一个 `TransformerBlock` 层，传入参数 `embed_dim`、`num_heads` 和 `feed_forward_dim`，用于对嵌入后的序列进行 Transformer 操作。将嵌入后的张量 `x` 传递给 `TransformerBlock` 层，得到处理后的输出张量 `x`。

接下来通过一个全连接层 `layers.Dense` 对输出张量 `x` 进行预测，输出一个形状为 `(vocab_size,)` 的张量 `outputs`，也就是计算出来下一个预测的单词的概率分布。

最后，定义了损失函数 `loss_fn` 为稀疏分类交叉熵损失函数，并使用 `"adam"` 优化器进行模型的编译。




    vocab_size = 20000  
    maxlen = 80  
    embed_dim = 256  
    num_heads = 2  
    feed_forward_dim = 256 

    def create_model():
        inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
        x = transformer_block(x)
        outputs = layers.Dense(vocab_size)(x)
        model = keras.Model(inputs=inputs, outputs=[outputs, x])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile( "adam", loss=[loss_fn, None], )
        return model
## Text Generator

这里定义了一个 `TextGenerator` 类，用于在训练过程作为回调函数来生成文本，展示在不同训练 epoch 下面的文本生成效果。

构造函数 `__init__` 接收参数 `max_tokens`、`start_tokens`、`index_to_word`、`top_k` 和 `print_every`，用于配置文本生成的相关参数。

`sample_from` 方法用于从给定的 logits（对数概率）中进行采样，根据概率分布选择下一个预测的单词 。它首先使用 `tf.math.top_k` 选择概率最高的前 `k` 个单词，然后进行 softmax 归一化，得到概率分布。最后，使用 `np.random.choice` 方法根据概率分布进行采样，选择下一个预测的单词。

`on_epoch_end` 方法在每个训练周期结束时调用，用于生成文本。它通过循环生成文本的过程，从给定的起始文本开始，逐步生成下一个单词，直到达到指定的最大生成单词数。在每次生成单词后，将其添加到已生成的列表中，并更新起始文本。最后，将生成的文本转换为字符串，并打印输出。


接下来，定义了一个起始提示文本 `start_prompt` 为 `this movie is very good`，并根据词汇表和起始提示文本生成了起始标记 `start_tokens`。然后，创建了一个 `TextGenerator` 对象 `text_gen_callback`，传入生成文本所需的参数。


    class TextGenerator(keras.callbacks.Callback):
        def __init__( self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1 ):
            self.max_tokens = max_tokens
            self.start_tokens = start_tokens
            self.index_to_word = index_to_word
            self.print_every = print_every
            self.k = top_k

        def sample_from(self, logits):
            logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
            indices = np.asarray(indices).astype("int32")
            preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
            preds = np.asarray(preds).astype("float32")
            return np.random.choice(indices, p=preds)

        def detokenize(self, number):
            return self.index_to_word[number]

        def on_epoch_end(self, epoch, logs=None):
            start_tokens = [_ for _ in self.start_tokens]
            if (epoch + 1) % self.print_every != 0:
                return
            num_tokens_generated = 0
            tokens_generated = []
            while num_tokens_generated <= self.max_tokens:
                pad_len = maxlen - len(start_tokens)
                sample_index = len(start_tokens) - 1
                if pad_len < 0:
                    x = start_tokens[:maxlen]
                    sample_index = maxlen - 1
                elif pad_len > 0:
                    x = start_tokens + [0] * pad_len
                else:
                    x = start_tokens
                x = np.array([x])
                y, _ = self.model.predict(x)
                sample_token = self.sample_from(y[0][sample_index])
                tokens_generated.append(sample_token)
                start_tokens.append(sample_token)
                num_tokens_generated = len(tokens_generated)
            txt = " ".join(  [self.detokenize(_) for _ in self.start_tokens + tokens_generated]  )
            print(f"generated text:\n{txt}\n")

    word_to_index = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index

    start_prompt = "this movie is very good"
    start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
    num_tokens_generated = 40
    text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, vocab)
    
# Train 

该部分就是创建了一个文本生成模型，训练 30 个 epoch ，并且调用 text_gen_callback 对象，在每次 epoch 结束的时候进行文本的生成。

    model = create_model()
    model.fit(text_ds, epochs=30, callbacks=[text_gen_callback])

下面对部分结果进行打印展示。可以看出来生成的文本效果一般，可能和数据集质量以及模型的复杂度有关。

    Epoch 1/30
    0s 169ms/stepse_5_loss: 5.59
    generated text:
    this movie is very good movie . the worst movie is about the plot . the story of course of course the story line is a great story about it . it is so well . the plot is a great plot of the
    Epoch 2/30
    0s 17ms/step- loss: 4.7109 - dense_5_loss: 4.71
    generated text:
    this movie is a great movie . a wonderful movie about it , it was just the characters that they were not a movie but the way the acting was not a bad script that is bad . but the script was bad ,
    ...
    Epoch 12/30
    0s 18ms/step- loss: 3.6976 - dense_5_loss: 3.69
    generated text:
    this movie is one of the best movies i have ever seen and i have seen it on vhs uncut and i 've seen the first time . i watched this film for the first and was all of it . it was great
    Epoch 13/30
    0s 19ms/step- loss: 3.6531 - dense_5_loss: 3.65
    generated text:
    this movie is one of the worst actors i have ever seen . it is the worst bollywood movie i have ever seen . i have no idea it . the acting was terrible and the directing is bad , but it was bad
    ...
    Epoch 29/30
    0s 18ms/step- loss: 3.2507 - dense_5_loss: 3.25
    generated text:
    this movie is so [UNK] and the acting is awful , but the script is poor . the plot is laughable and the ending is terrible . there isn 't anything about this movie that was so bad it doesn 't make any sense
    Epoch 30/30
    0s 17ms/step- loss: 3.2359 - dense_5_loss: 3.23
    generated text:
    this movie is not a great time , but this movie is one of those actors that are [UNK] and that you can not take up to the screen . the plot is simple . it doesn 't matter what 's going on and

