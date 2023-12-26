import tensorflow as tf
from keras.layers import TextVectorization
import string
import re
from keras import layers

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

batch_size = 128
raw_train_ds = tf.keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size, validation_split=0.2, subset="training", seed=1337)
raw_val_ds = tf.keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size, validation_split=0.2, subset="validation", seed=1337)
raw_test_ds = tf.keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)

print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")

for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(5):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"[{re.escape(string.punctuation)}]", "")


max_features = 200000
embedding_dim = 128
sequence_length = 500
vectorize_layer = TextVectorization(standardize=custom_standardization, max_tokens=max_features, output_mode="int", output_sequence_length=sequence_length)
text_ds = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text).cache().prefetch(buffer_size = 10)
val_ds = raw_val_ds.map(vectorize_text).cache().prefetch(buffer_size = 10)
test_ds = raw_test_ds.map(vectorize_text).cache().prefetch(buffer_size = 10)


inputs = tf.keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)
model = tf.keras.Model(inputs, predictions)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=3)
model.evaluate(test_ds)


inputs = tf.keras.Input(shape=(1,), dtype="string")
indices = vectorize_layer(inputs)
outputs = model(indices)
end_to_end_model = tf.keras.Model(inputs, outputs)
end_to_end_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
end_to_end_model.evaluate(raw_test_ds)



# Epoch 1/3
# 157/157 [==============================] - 9s 26ms/step - loss: 0.6597 - accuracy: 0.5632 - val_loss: 0.4223 - val_accuracy: 0.8080
# Epoch 2/3
# 157/157 [==============================] - 1s 9ms/step - loss: 0.3168 - accuracy: 0.8676 - val_loss: 0.3400 - val_accuracy: 0.8596
# Epoch 3/3
# 157/157 [==============================] - 1s 9ms/step - loss: 0.1406 - accuracy: 0.9501 - val_loss: 0.3697 - val_accuracy: 0.8750
# 196/196 [==============================] - 8s 38ms/step - loss: 0.3887 - accuracy: 0.8665
# 196/196 [==============================] - 4s 18ms/step - loss: 0.3887 - accuracy: 0.8665
