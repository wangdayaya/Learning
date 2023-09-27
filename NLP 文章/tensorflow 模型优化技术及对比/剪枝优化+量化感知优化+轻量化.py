import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=4, validation_split=0.1, )
# —————————剪枝技术进行微调—————————————
batch_size = 128
epochs = 2
validation_split = 0.1
num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = { 'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50, final_sparsity=0.80, begin_step=0,  end_step=end_step)}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), ]
model_for_pruning.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks)
model_for_pruning.evaluate(test_images, test_labels)
# ———————————轻量化+量化后的模型———————————
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()
with open('剪枝优化+量化感知优化+轻量化.tflite', 'wb') as f:
    f.write(quantized_and_pruned_tflite_model)
