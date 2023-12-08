import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
from transformers import TFAutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
import keras_nlp
from transformers.keras_callbacks import KerasMetricCallback
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import pipeline

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
MIN_TARGET_LENGTH = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
MAX_EPOCHS = 5
MODEL_CHECKPOINT = "t5-small"

raw_datasets = load_dataset("xsum", split="train")
print(raw_datasets)
print(raw_datasets[0])
raw_datasets = raw_datasets.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")
train_dataset = tokenized_datasets["train"].to_tf_dataset(batch_size=BATCH_SIZE, columns=["input_ids", "attention_mask", "labels"], shuffle=True, collate_fn=data_collator)
test_dataset = tokenized_datasets["test"].to_tf_dataset(batch_size=BATCH_SIZE,  columns=["input_ids", "attention_mask", "labels"], shuffle=False, collate_fn=data_collator)
generation_dataset = tokenized_datasets["test"].shuffle().select(list(range(200))).to_tf_dataset(batch_size=BATCH_SIZE,   columns=["input_ids", "attention_mask", "labels"], shuffle=False, collate_fn=data_collator)
model.load_weights('summary.h5')
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer)

# rouge_l = keras_nlp.metrics.RougeL()
#
#
# def metric_fn(eval_predictions):
#     predictions, labels = eval_predictions
#     decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     for label in labels:
#         label[label < 0] = tokenizer.pad_token_id
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     result = rouge_l(decoded_labels, decoded_predictions)
#     result = {"RougeL": result["f1_score"]}
#     return result
#
#
# callbacks = [KerasMetricCallback(metric_fn, eval_dataset=generation_dataset, predict_with_generate=True),
#              ModelCheckpoint(filepath="summary.h5", monitor='val_loss', save_weights_only=True),]
# model.fit(train_dataset, validation_data=test_dataset, epochs=MAX_EPOCHS, callbacks=callbacks)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework="tf")
for i in range(5):
    print(summarizer(raw_datasets["test"][i]["document"], min_length=MIN_TARGET_LENGTH, max_length=MAX_TARGET_LENGTH))
    print(raw_datasets["test"][i]["summary"])


# 17.5G-0.5G
# Epoch 1/5
# 20405/20405 [==============================] - 3240s 158ms/step - loss: 2.6656 - val_loss: 2.3561 - RougeL: 0.2318
# Epoch 2/5
# 20405/20405 [==============================] - 3200s 157ms/step - loss: 2.5085 - val_loss: 2.2873 - RougeL: 0.2393
# Epoch 3/5
# 20405/20405 [==============================] - 3199s 157ms/step - loss: 2.4330 - val_loss: 2.2488 - RougeL: 0.2438
# Epoch 4/5
# 20405/20405 [==============================] - 3212s 157ms/step - loss: 2.3785 - val_loss: 2.2172 - RougeL: 0.2468
# Epoch 5/5
# 20405/20405 [==============================] - 3239s 159ms/step - loss: 2.3335 - val_loss: 2.1938 - RougeL: 0.2493

# [{'summary_text': 'The Australian government is trying to verify the deaths of two Lebanese fighters who left the country to fight with Islamic State militants.'}]
# The Australian government says new citizenship laws will help combat terrorism but experts worry too many people will be affected by the laws, including children.
# [{'summary_text': "It's been a week since the murder of Lucky the donkey, who was found dead at her home in Fermanagh, County Down."}]
# On the day of her burial, County Fermanagh murder victim Connie Leonard's legacy is already being felt.
# [{'summary_text': 'Insurance claims for catastrophic storms in Australia have risen by more than A$8.8m (£8.6m) in the past year, a council has said.'}]
# Severe storms that hit Australia during April and May have led to more than A$1.55bn ($1.18bn; Â£778m) in insurance losses so far.
# [{'summary_text': "It's been a long time since I was selected by BBC Sport to pick my own team of the week for BBC Sport."}]
# Manchester United did Chelsea's title rivals Tottenham a favour and kept up their own pursuit of the top four with a dominant win over the Premier League leaders.
# [{'summary_text': 'A Belarusian politician has been found guilty of tax evasion after he held bank accounts in Poland and Lithuania.'}]
# One of Belarus' most prominent human rights activists has been sentenced to four-and-a-half years in prison for tax evasion.