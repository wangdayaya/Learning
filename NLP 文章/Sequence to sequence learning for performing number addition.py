from tensorflow import keras
from keras import layers
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

TRAINING_SIZE = 50000
DIGITS = 3
MAXLEN = DIGITS + 1 + DIGITS


class CharacterTable:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[x] for x in x)


chars = "0123546789+ "
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print("Generatin data...")
while len(questions) < TRAINING_SIZE:
    f = lambda: int("".join(np.random.choice(list("1234567890")) for _ in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = "{}+{}".format(a, b)
    query = q + " " * (MAXLEN - len(q))
    ans = str(a + b)
    ans += " " * (DIGITS + 1 - len(ans))
    questions.append(query)
    expected.append(ans)
print("Total questions:", len(questions))

print("Vectorization ...")
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool_)  # [50000, 7, 12]
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool_)  # [50000, 4, 12]
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

print("build model...")
model = keras.Sequential()
model.add(layers.LSTM(128, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dense(len(chars), activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 30
batch_size = 32
for epoch in range(1, epochs):
    print()
    print("Iter", epoch)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_val, y_val),
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)])
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = np.argmax(model.predict(rowx), axis=-1)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print("Q", q, end=" ")
        print("T", correct, end=" ")
        if correct == guess:
            print("☑ " + guess)
        else:
            print("☒ " + guess)
