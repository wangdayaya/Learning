import time
import warnings
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings('ignore')
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
batch_size = 128
epochs = 4

model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit( train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1,)
tf.keras.models.save_model(model, '基准模型.h5', include_optimizer=False)
