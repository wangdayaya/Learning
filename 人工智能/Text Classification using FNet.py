import keras_nlp
import tensorflow as tf
from tensorflow import keras


# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

SEED = 42
keras.utils.set_random_seed(SEED)
BATCH_SIZE = 64
EPOCHS = 3
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 15000
EMBED_DIM = 128
INTERMEDIATE_DIM = 512

train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=BATCH_SIZE, validation_split=0.2, subset="training", seed=SEED, )
val_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=BATCH_SIZE, validation_split=0.2, subset="validation", seed=SEED, )
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=BATCH_SIZE)

train_ds = train_ds.map(lambda x, y: (tf.strings.lower(x), y))
val_ds = val_ds.map(lambda x, y: (tf.strings.lower(x), y))
test_ds = test_ds.map(lambda x, y: (tf.strings.lower(x), y))

for text_batch, label_batch in train_ds.take(1):
    for i in range(2):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])


def train_word_piece(ds, vocab_size, reserved_tokens):
    word_piece_ds = ds.unbatch().map(lambda x, y: x)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(word_piece_ds.batch(1000).prefetch(2), vocabulary_size=vocab_size, reserved_tokens=reserved_tokens)
    return vocab


reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = [element[0] for element in train_ds]
vocab = train_word_piece(train_ds, VOCAB_SIZE, reserved_tokens)
print(f"Tokens length is {len(vocab)}")


tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocab, lowercase=False, sequence_length=MAX_SEQUENCE_LENGTH)
input_sentence_ex = train_ds.take(1).get_single_element()[0][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))


def format_dataset(sentence, label):
    sentence = tokenizer(sentence)
    return {"input_ids": sentence}, label


train_ds = train_ds.map(format_dataset).shuffle(512).prefetch(16).cache()
val_ds = val_ds.map(format_dataset).shuffle(512).prefetch(16).cache()
test_ds = test_ds.map(format_dataset).shuffle(512).prefetch(16).cache()


input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")
x = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=VOCAB_SIZE, sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBED_DIM, mask_zero=True)(input_ids)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)
fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")
fnet_classifier.summary()

fnet_classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=['accuracy'])
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)
# 5.6G-0.7G
# Epoch 1/3
# 79/79 [==============================] - 18s 172ms/step - loss: 0.7111 - accuracy: 0.5031 - val_loss: 0.6890 - val_accuracy: 0.6070
# Epoch 2/3
# 79/79 [==============================] - 13s 163ms/step - loss: 0.5597 - accuracy: 0.6817 - val_loss: 0.3584 - val_accuracy: 0.8450
# Epoch 3/3
# 79/79 [==============================] - 13s 164ms/step - loss: 0.2784 - accuracy: 0.8873 - val_loss: 0.3116 - val_accuracy: 0.8718
# Total params: 2,382,337
# Trainable params: 2,382,337
# Non-trainable params: 0
# Epoch 1/3
# 79/79 [==============================] - 18s 173ms/step - loss: 0.7111 - accuracy: 0.5031 - val_loss: 0.6890 - val_accuracy: 0.6062
# Epoch 2/3
# 79/79 [==============================] - 13s 165ms/step - loss: 0.5600 - accuracy: 0.6812 - val_loss: 0.3558 - val_accuracy: 0.8456
# Epoch 3/3
# 79/79 [==============================] - 13s 162ms/step - loss: 0.2791 - accuracy: 0.8856 - val_loss: 0.3220 - val_accuracy: 0.8688
# 98/98 [==============================] - 10s 73ms/step - loss: 0.3328 - accuracy: 0.8567



NUM_HEADS = 2
input_ids = keras.Input(shape=(None,), dtype='int64', name="input_ids")
x = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=VOCAB_SIZE, sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBED_DIM, mask_zero=True)(input_ids)
x = keras_nlp.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(x)
x = keras_nlp.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(x)
x = keras_nlp.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(x)
x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
transformer_classifier = keras.Model(input_ids, outputs, name="transformer_classifier")
transformer_classifier.compile(optimizer=keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=['accuracy'])
transformer_classifier.summary()
transformer_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
transformer_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)
# 9.6G-1.2G
# Epoch 1/3
# 79/79 [==============================] - 16s 137ms/step - loss: 0.7204 - accuracy: 0.5854 - val_loss: 0.3933 - val_accuracy: 0.8256
# Epoch 2/3
# 79/79 [==============================] - 10s 126ms/step - loss: 0.2667 - accuracy: 0.8908 - val_loss: 0.3273 - val_accuracy: 0.8758
# Epoch 3/3
# 79/79 [==============================] - 10s 126ms/step - loss: 0.1998 - accuracy: 0.9230 - val_loss: 0.3107 - val_accuracy: 0.8774
# Total params: 2,580,481
# Trainable params: 2,580,481
# Non-trainable params: 0
# Epoch 1/3
# 79/79 [==============================] - 15s 139ms/step - loss: 0.7204 - accuracy: 0.5856 - val_loss: 0.4003 - val_accuracy: 0.8204
# Epoch 2/3
# 79/79 [==============================] - 10s 127ms/step - loss: 0.2679 - accuracy: 0.8896 - val_loss: 0.3165 - val_accuracy: 0.8804
# Epoch 3/3
# 79/79 [==============================] - 10s 127ms/step - loss: 0.1992 - accuracy: 0.9230 - val_loss: 0.3142 - val_accuracy: 0.8756
# 98/98 [==============================] - 8s 47ms/step - loss: 0.3437 - accuracy: 0.8616












# Epoch 1/3
# 313/313 [==============================] - 50s 148ms/step - loss: 0.5764 - accuracy: 0.6474 - val_loss: 0.3518 - val_accuracy: 0.8470
# Epoch 2/3
# 313/313 [==============================] - 44s 139ms/step - loss: 0.3092 - accuracy: 0.8712 - val_loss: 0.5374 - val_accuracy: 0.7708
# Epoch 3/3
# 313/313 [==============================] - 43s 137ms/step - loss: 0.1955 - accuracy: 0.9232 - val_loss: 0.3746 - val_accuracy: 0.8500
# 391/391 [==============================] - 40s 76ms/step - loss: 0.4002 - accuracy: 0.8380
#
# Epoch 1/3
# 313/313 [==============================] - 13s 37ms/step - loss: 0.4514 - accuracy: 0.7703 - val_loss: 0.3670 - val_accuracy: 0.8390
# Epoch 2/3
# 313/313 [==============================] - 11s 36ms/step - loss: 0.2078 - accuracy: 0.9204 - val_loss: 0.3642 - val_accuracy: 0.8544
# Epoch 3/3
# 313/313 [==============================] - 11s 36ms/step - loss: 0.1574 - accuracy: 0.9426 - val_loss: 0.3473 - val_accuracy: 0.8812
# 391/391 [==============================] - 5s 13ms/step - loss: 0.4207 - accuracy: 0.8581