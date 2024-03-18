import os

import numpy
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from keras import layers
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#
#
# def load_images_from_directory(directory):
#     image_list = []
#     for filename in tqdm(os.listdir(directory)[:3]):
#         filepath = os.path.join(directory, filename)
#         with Image.open(filepath) as img:
#             image_array = np.array(img)
#             image_list.append(image_array)
#     return image_list
#
#
# DIV2K_train_HR = load_images_from_directory('.\\DIV2K\\DIV2K_train_HR')
# DIV2K_train_LR_bicubic = load_images_from_directory('.\\DIV2K\\DIV2K_train_LR_bicubic\\X4')
# DIV2K_valid_HR = load_images_from_directory('.\\DIV2K\\DIV2K_valid_HR')
# DIV2K_valid_LR_bicubic = load_images_from_directory('.\\DIV2K\\DIV2K_valid_LR_bicubic\\X4')
#


AUTOTUNE = tf.data.AUTOTUNE
div2k_data = tfds.image.Div2k(config="bicubic_x4")
div2k_data.download_and_prepare()
train = div2k_data.as_dataset(split="train", as_supervised=True)
train_cache = train.cache()
val = div2k_data.as_dataset(split="validation", as_supervised=True)
val_cache = val.cache()

def flip_left_right(lowres_img, highres_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5, lambda: (lowres_img, highres_img),
                   lambda: (tf.image.flip_left_right(lowres_img), tf.image.flip_left_right(highres_img)))


def random_rotate(lowres_img, highres_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)


def random_crop(lowres_img, highres_img, hr_crop_size=96, scale=4):
    lowres_crop_size = hr_crop_size // scale
    lowres_img_shape = tf.shape(lowres_img)[:2]
    lowres_width = tf.random.uniform(shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32)
    lowres_height = tf.random.uniform(shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32)
    lowres_img_cropped = lowres_img[lowres_height: lowres_height + lowres_crop_size,
                         lowres_width: lowres_width + lowres_crop_size]
    highres_width = lowres_width * scale
    highres_height = lowres_height * scale
    highres_img_cropped = highres_img[highres_height: highres_height + hr_crop_size,
                          highres_width: highres_width + hr_crop_size]
    return lowres_img_cropped, highres_img_cropped

# def dataset_object(X, Y, training=True):
#     for x, y in zip(X, Y):
#         x, y, = random_crop(x, y, scale=4)
#     if training:
#         for x, y in zip(X, Y):
#             x, y = random_rotate(x, y)
#             x, y = flip_left_right(x, y)
#     return X, Y
#

# dataset_object(DIV2K_train_HR, DIV2K_train_LR_bicubic)
def dataset_object(dataset_cache, training=True):
    ds = dataset_cache.map(lambda lowres, highres: random_crop(lowres, highres, scale=4), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(16)
    if training:
        ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = dataset_object(train_cache, training=True)
val_ds = dataset_object(val_cache, training=False)

lowres, highres = next(iter(train_ds))
plt.figure(figsize=(10, 10))
for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    plt.imshow(highres[i].numpy().astype("uint8"))
    plt.title(highres[i].shape)
    plt.axis("off")
    plt.show()

plt.figure(figsize=(10, 10))
for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    plt.imshow(lowres[i].numpy().astype("uint8"))
    plt.title(lowres[i].shape)
    plt.axis("off")
    plt.show()


def PSNR(super_resolution, high_resolution):
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value


class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, x):
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        super_resolution_img = self(x, training=False)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        super_resolution_img = tf.round(super_resolution_img)
        super_resolution_img = tf.squeeze(tf.cast(super_resolution_img, tf.uint8), axis=0)
        return super_resolution_img


def ResBlock(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


def Upsampling(inputs, factor=2, **kwargs):
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x


def make_model(num_filters, num_of_residual_blocks):
    input_layer = layers.Input(shape=(None, None, 3))
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)
    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])
    x = Upsampling(x)
    x = layers.Conv2D(3, 3, padding="same")(x)
    output_layer = layers.Rescaling(scale=255)(x)
    return EDSRModel(input_layer, output_layer)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="EDSR/checkpoint.keras", save_weights_only=False, monitor="loss",  mode="min", save_best_only=True, )
model = make_model(num_filters=64, num_of_residual_blocks=16)
optim_edsr = keras.optimizers.Adam(learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[5000], values=[1e-4, 5e-5]))
model.compile(optimizer=optim_edsr, loss="mae", metrics=[PSNR])
model.fit(train_ds, epochs=50, steps_per_epoch=200, validation_data=val_ds, callbacks=[early_stopping_callback, model_checkpoint_callback])


def plot_results(lowres, preds):
    plt.figure(figsize=(24, 14))
    plt.subplot(132), plt.imshow(lowres), plt.title("Low resolution")
    plt.subplot(133), plt.imshow(preds), plt.title("Prediction")
    plt.show()


for lowres, highres in val.take(10):
    lowres = tf.image.random_crop(lowres, (150, 150, 3))
    preds = model.predict_step(lowres)
    plot_results(lowres, preds)

# 6.5G-4.6G

# Epoch 1/50
# 2024-03-15 17:33:37.446546: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8100
# 200/200 [==============================] - 7s 20ms/step - loss: 26.4669 - PSNR: 19.8638 - val_loss: 14.3105 - val_PSNR: 23.2535
# Epoch 2/50
# 200/200 [==============================] - 3s 13ms/step - loss: 12.8289 - PSNR: 25.4992 - val_loss: 11.6453 - val_PSNR: 26.0368
# Epoch 3/50
# 200/200 [==============================] - 2s 12ms/step - loss: 10.9451 - PSNR: 27.2957 - val_loss: 10.3392 - val_PSNR: 24.5271
# Epoch 4/50
# 200/200 [==============================] - 2s 12ms/step - loss: 9.9218 - PSNR: 27.1388 - val_loss: 9.7049 - val_PSNR: 26.2940
# Epoch 5/50
# 200/200 [==============================] - 2s 12ms/step - loss: 9.6239 - PSNR: 28.0989 - val_loss: 8.7290 - val_PSNR: 28.2735
# Epoch 6/50
# 200/200 [==============================] - 2s 12ms/step - loss: 9.5346 - PSNR: 27.6521 - val_loss: 9.4293 - val_PSNR: 24.6870
# Epoch 7/50
# 200/200 [==============================] - 2s 12ms/step - loss: 8.7801 - PSNR: 28.2608 - val_loss: 9.0527 - val_PSNR: 30.3716
# Epoch 8/50
# 200/200 [==============================] - 2s 12ms/step - loss: 8.5680 - PSNR: 28.5681 - val_loss: 8.2357 - val_PSNR: 31.1381
# Epoch 9/50
# 200/200 [==============================] - 2s 12ms/step - loss: 8.7130 - PSNR: 29.5744 - val_loss: 8.8001 - val_PSNR: 30.0908
# Epoch 10/50
# 200/200 [==============================] - 2s 12ms/step - loss: 8.5190 - PSNR: 29.8409 - val_loss: 9.2607 - val_PSNR: 28.5442
# Epoch 11/50
# 200/200 [==============================] - 2s 12ms/step - loss: 8.1657 - PSNR: 29.4979 - val_loss: 7.6911 - val_PSNR: 29.1756
# Epoch 12/50
# 200/200 [==============================] - 2s 12ms/step - loss: 8.3187 - PSNR: 30.3074 - val_loss: 8.7230 - val_PSNR: 29.4424
# Epoch 13/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.9737 - PSNR: 31.1096 - val_loss: 8.4238 - val_PSNR: 27.1940
# Epoch 14/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.8842 - PSNR: 30.6993 - val_loss: 7.3708 - val_PSNR: 33.2809
# Epoch 15/50
# 200/200 [==============================] - 2s 12ms/step - loss: 8.0194 - PSNR: 30.1181 - val_loss: 7.6638 - val_PSNR: 28.6607
# Epoch 16/50
# 200/200 [==============================] - 2s 12ms/step - loss: 8.1145 - PSNR: 28.9978 - val_loss: 8.0316 - val_PSNR: 28.9922
# Epoch 17/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.7256 - PSNR: 31.1083 - val_loss: 7.5451 - val_PSNR: 31.2853
# Epoch 18/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.7396 - PSNR: 30.7906 - val_loss: 7.8016 - val_PSNR: 31.0204
# Epoch 19/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.7600 - PSNR: 30.6698 - val_loss: 7.5488 - val_PSNR: 26.3456
# Epoch 20/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.9794 - PSNR: 30.4598 - val_loss: 7.1711 - val_PSNR: 26.6702
# Epoch 21/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.5965 - PSNR: 31.5485 - val_loss: 7.4096 - val_PSNR: 26.7971
# Epoch 22/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.7922 - PSNR: 31.1160 - val_loss: 8.3919 - val_PSNR: 29.6128
# Epoch 23/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.6447 - PSNR: 30.8985 - val_loss: 7.7192 - val_PSNR: 32.8681
# Epoch 24/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.7343 - PSNR: 30.7942 - val_loss: 6.9050 - val_PSNR: 32.5451
# Epoch 25/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.7482 - PSNR: 31.0483 - val_loss: 7.9020 - val_PSNR: 30.0776
# Epoch 26/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.4504 - PSNR: 31.4250 - val_loss: 7.9389 - val_PSNR: 30.5068
# Epoch 27/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3959 - PSNR: 31.9584 - val_loss: 7.6069 - val_PSNR: 30.8053
# Epoch 28/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3398 - PSNR: 32.5318 - val_loss: 6.9680 - val_PSNR: 34.7906
# Epoch 29/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3842 - PSNR: 31.1897 - val_loss: 7.4768 - val_PSNR: 30.2146
# Epoch 30/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.5060 - PSNR: 31.4764 - val_loss: 8.0120 - val_PSNR: 29.4162
# Epoch 31/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3136 - PSNR: 31.7987 - val_loss: 7.8620 - val_PSNR: 26.6573
# Epoch 32/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3779 - PSNR: 32.0797 - val_loss: 7.2626 - val_PSNR: 32.4579
# Epoch 33/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.4706 - PSNR: 30.9865 - val_loss: 7.4808 - val_PSNR: 26.7492
# Epoch 34/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3001 - PSNR: 31.3675 - val_loss: 7.0601 - val_PSNR: 29.8221
# Epoch 35/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.2969 - PSNR: 31.5580 - val_loss: 7.6531 - val_PSNR: 31.0716
# Epoch 36/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.4493 - PSNR: 31.6907 - val_loss: 7.2452 - val_PSNR: 29.4563
# Epoch 37/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3356 - PSNR: 31.8137 - val_loss: 6.1610 - val_PSNR: 30.5943
# Epoch 38/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.1124 - PSNR: 32.7166 - val_loss: 7.3956 - val_PSNR: 28.2870
# Epoch 39/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3656 - PSNR: 31.9348 - val_loss: 6.7752 - val_PSNR: 30.8417
# Epoch 40/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3191 - PSNR: 32.4177 - val_loss: 7.2674 - val_PSNR: 31.3844
# Epoch 41/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.4515 - PSNR: 31.9981 - val_loss: 7.4356 - val_PSNR: 28.6615
# Epoch 42/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3990 - PSNR: 32.3725 - val_loss: 6.8398 - val_PSNR: 29.6918
# Epoch 43/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.1420 - PSNR: 32.7798 - val_loss: 7.4322 - val_PSNR: 29.6703
# Epoch 44/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.1697 - PSNR: 31.9976 - val_loss: 7.1463 - val_PSNR: 35.1633
# Epoch 45/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3378 - PSNR: 31.4095 - val_loss: 7.1845 - val_PSNR: 30.7239
# Epoch 46/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.2581 - PSNR: 32.2279 - val_loss: 7.3768 - val_PSNR: 31.8386
# Epoch 47/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.3122 - PSNR: 32.0149 - val_loss: 7.2080 - val_PSNR: 31.2420
# Epoch 48/50
# 200/200 [==============================] - 2s 12ms/step - loss: 7.2122 - PSNR: 33.6319 - val_loss: 7.1458 - val_PSNR: 31.3648