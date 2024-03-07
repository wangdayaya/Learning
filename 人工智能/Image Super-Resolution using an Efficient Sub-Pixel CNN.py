import math
import os
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.utils import load_img, img_to_array, image_dataset_from_directory
from keras_core import ops
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 11.8g - 4.5G

root_dir = "D:\\pythonProject\\HKYModel\\BSR\\BSDS500\\data"
crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor
batch_size = 8
train_ds = image_dataset_from_directory(root_dir, batch_size=batch_size, image_size=(crop_size, crop_size), validation_split=0.2,  subset="training", seed=1337,  label_mode=None)
valid_ds = image_dataset_from_directory(root_dir, batch_size=batch_size,  image_size=(crop_size, crop_size),  validation_split=0.2, subset="validation",  seed=1337, label_mode=None)


def scaling(input_image):
    input_image = input_image / 255.
    return input_image


train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)
dataset = os.path.join(root_dir, "images")
test_path = os.path.join(dataset, "test")
test_img_paths = sorted([os.path.join(test_path, fname) for fname in os.listdir(test_path) if fname.endswith(".jpg")])


def process_input(input, input_size):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


train_ds = train_ds.map(lambda x: (process_input(x, input_size), process_target(x))).prefetch(buffer_size=32)
valid_ds = valid_ds.map(lambda x: (process_input(x, input_size), process_target(x))).prefetch(buffer_size=32)


class DepthToSpace(layers.Layer):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config

    def call(self, input):
        batch, height, width, depth = ops.shape(input)
        depth = depth // (self.block_size**2)
        x = ops.reshape(input, [batch, height, width, self.block_size, self.block_size, depth])
        x = ops.transpose(x, [0, 1, 3, 2, 4, 5])
        x = ops.reshape(x, [batch, height * self.block_size, width * self.block_size, depth])
        return x


def get_model(upscale_factor=3, channels=1):
    conv_args = {"activation": "relu",  "kernel_initializer": "orthogonal", "padding": "same"}
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(512, 5, **conv_args)(inputs)
    x = layers.Conv2D(256, 3, **conv_args)(x)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor**2), 3, **conv_args)(x)
    outputs = DepthToSpace(upscale_factor)(x)
    return keras.Model(inputs, outputs)


def plot_results(img, prefix, title):
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.
    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")
    plt.title(title)
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(img_array[::-1], origin="lower")
    x1, x2, y1, y2 = 200, 300, 100, 200
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.yticks(visible=False)
    plt.xticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")
    plt.savefig(str(prefix) + "-" + title + ".png")
    plt.show()


def get_lowres_image(img, upscale_factor):
    return img.resize((img.size[0] // upscale_factor, img.size[1] // upscale_factor), PIL.Image.BICUBIC)


def upscale_image(model, img):
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.
    input = np.expand_dims(y, axis=0)
    out = model.predict(input)
    out_img_y = out[0]
    out_img_y *= 255.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert( "RGB" )
    return out_img


class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))


early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5)
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath="ESPCN/checkpoint.keras", save_weights_only=False, monitor="loss",  mode="min", save_best_only=True, )
model = get_model(upscale_factor=upscale_factor, channels=1)
model.summary()
callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.MeanSquaredError())
model.fit(train_ds, epochs=200, callbacks=callbacks, validation_data=valid_ds, verbose=2)


for index, test_img_path in enumerate(test_img_paths[30:35]):
    img = load_img(test_img_path)
    lowres_input = get_lowres_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)
    print("PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr )
    print("PSNR of predict and high resolution is %.4f" % test_psnr)
    plot_results(lowres_img, index, "lowres")
    plot_results(highres_img, index, "highres")
    plot_results(prediction, index, "prediction")

