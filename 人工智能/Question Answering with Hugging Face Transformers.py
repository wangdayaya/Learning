from transformers import AutoTokenizer
from transformers import TFAutoModelForQuestionAnswering
from tensorflow import keras
from datasets import load_dataset
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
datasets = load_dataset("squad")
model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
max_length = 384
doc_stride = 128
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
model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
keras.mixed_precision.set_global_policy("mixed_float16")
optimizer = keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer)
model.fit(train_set, validation_data=validation_set, epochs=2)
model.save_weights("QA.h5")


# 加载模型时候使用
# model.load_weights("QA.h5")


def qa(context, question):
    inputs = tokenizer([context], [question], return_tensors="np")
    outputs = model(inputs)
    start_position = tf.argmax(outputs.start_logits, axis=1)
    end_position = tf.argmax(outputs.end_logits, axis=1)
    print(int(start_position), int(end_position))
    answer = inputs["input_ids"][0, int(start_position): int(end_position) + 1]
    print(answer)
    print(tokenizer.decode(answer))


qa('Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.', 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?')   # Saint Bernadette Soubirous
qa('Keras is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages. It also has extensive documentation and developer guides. ', 'What is Keras?')
qa('The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.', 'The College of Science began to offer civil engineering courses beginning at what time at Notre Dame?')   # the 1870s


