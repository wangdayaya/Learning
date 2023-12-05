# 前言

问答是一个经典的 NLP 任务，有多种实际应用的形式。 如下面列举出来的两种常见形式：

- 每个问题都提供可能答案的列表，模型只需要返回`答案选项的概率分布`，这种一般比较简单。 
- 给定一个输入`文档`（俗称上下文）和一个`有关该文档的问题`，并且它必须提取文档中`包含答案的文本范围`。 在这种情况下，模型不是计算答案选项的概率分布，而是计算文档文本中标记的`两个概率分布`，对应表示包含答案的范围的`开始位置`和`结束位置`。这种问答称为“提取式问答”。

一般来说提取式问答的模型需要`非常庞大的数据`来从头训练，但是使用强大的`预训练基础模型`开始可以将数据集大小减少多个数量级，并且能取得令人满意的效果。本文介绍的是在轻量级 BERT 模型 `distilbert 模型`上进行微调来完成简单的问答任务。


# 数据

在将这些文本输入模型之前，我们需要对它们进行预处理。 这是由 `Transformers Tokenizer` 完成的，它将输入的文本转化为 `token id` ，并生成其他输入供 bert 模型使用。为此，我们使用 AutoTokenizer.from_pretrained 方法从 `distilbert-base-cased` 实例我们的分词器，这将确保我们得到一个与我们想要使用的预训练模型模型 distilbert 架构相对应的分词器。 

下面代码主要是数据处理过程，每个样本将 `context` 和 `question` 作为输入，将 `answer` 在输入的 `token 序列`中的起始位置 `start_positions` 和结束位置 `end_positions` 标记出来，如果 `answer` 在输入的 token 序列中不存在，则将起始位置 start_positions 和结束位置 end_positions 都标记为 ```cls_token_id``` 在序列中的索引位置。

```
def prepare_train_feature(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    examples["context"] = [c.lstrip() for c in examples["context"]]
    tokenized_examples = tokenizer(examples["question"], examples["context"], truncation="only_second",  max_length=max_length, stride=doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples
tokenized_datasets = datasets.map(prepare_train_feature, batched=True, remove_columns=datasets["train"].column_names)
train_set = tokenized_datasets["train"].with_format("numpy")[:]
validation_set = tokenized_datasets["validation"].with_format("numpy")[:]
```




# 模型

我们选择了轻量级的`distilbert`，它是著名的 BERT 语言模型的一个较小的蒸馏版本。 但是如果任务需要更高的准确率，并且有足够 GPU 来处理它，那么可以使用更大的模型，如 `roberta-large` 。


使用 `TFAutoModelForQuestionAnswering` 类从 Hugging Face Transformers 库中加载预训练的问答模型 ```distilbert-base-cased```。使用 TensorFlow 的混合精度 `mixed_float16` 进行训练，这有助于在相同计算资源下加速模型的训练。另外配置 `Adam` 优化器，设置学习率为 `5e-5`，然后使用 `compile` 方法将模型编译，最后将模型权重保存成 h5 模型，以供测试需要。

训练需要 16G 作用的显存，耗时总共 20 分钟左右。

```
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
keras.mixed_precision.set_global_policy("mixed_float16")
optimizer = keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer)
model.fit(train_set, validation_data=validation_set, epochs=2)
model.save_weights("QA.h5")
```

模型输入：
```
question: 'Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides. '
answer: What is Keras?
```

预测结果为：

    it offers consistent & simple APIs

可以看出，模型的问答效果初具效果。

# 发展

当前的问答任务，主要已经逐渐放弃了 BERT 这种“小模型”，而是使用参数量越来越大的大模型，其中以 ChatGPT 最为出色，但是究其架构细节，仍然是在 Transformer 基础上搭建而成的，只是额外加入了 SFT 和强化学习的步骤来使其更加符合人类的问答习惯而已。

# 参考

https://github.com/wangdayaya/DP_2023/blob/main/NLP%20%E6%96%87%E7%AB%A0/Question%20Answering%20with%20Hugging%20Face%20Transformers.py


