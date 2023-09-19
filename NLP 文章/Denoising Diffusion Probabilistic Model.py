import os.path
import time
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from tqdm import tqdm
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# 22.2G-4.8G
batch_size = 64
num_epochs = 800
total_timesteps = 1000
norm_groups = 8
learning_rate = 2e-5
img_size = 64
img_channels = 3
clip_min = -1.0
clip_max = 1.0
first_conv_channels = 64
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, False, True, True]
num_res_blocks = 2


def augment(img):
    return tf.image.random_flip_left_right(img)


def resize_and_rescale(img, size):
    height = tf.shape(img)[0]
    width = tf.shape(img)[1]
    crop_size = tf.minimum(height, width)
    img = tf.image.crop_to_bounding_box(img, (height - crop_size) // 2, (width - crop_size) // 2, crop_size, crop_size)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.image.resize(img, size=size, antialias=True)
    img = img / 127.5 - 1.0
    img = tf.clip_by_value(img, clip_min, clip_max)
    return img


# ---------------------------
def train_preprocessing(x):
    img = resize_and_rescale(x, size=(img_size, img_size))
    img = augment(img)
    return img

root = 'D:\\stable diffusion\\stable-diffusion-webui\\outputs\\txt2img-images\\2023-09-14'
images_files = [os.path.join(root, name) for name in os.listdir(root)]
images = []
for filepath in images_files:
    image = tf.io.read_file(filepath)
    image = tf.image.decode_image(image)
    image = tf. image.resize(image, [img_size, img_size])
    images.append(image)
images = np.array(images)
ds = tf.data.Dataset.from_tensor_slices(images)
train_ds = ds.map(train_preprocessing).batch(batch_size, drop_remainder=True).shuffle(batch_size * 2).prefetch(tf.data.AUTOTUNE)
plt.figure(figsize=(10, 10))
for image in train_ds.take(1):
    for i in range(1):
        plt.imshow(image[i])
        plt.axis('off')
plt.show()

# ---------------------------
# def train_preprocessing(x):
#     img = x["image"]
#     img = resize_and_rescale(img, size=(img_size, img_size))
#     img = augment(img)
#     return img
#
# dataset_name = "oxford_flowers102"
# splits = ["train"]
# (ds,) = tfds.load(dataset_name, split=splits, with_info=False, shuffle_files=True)
# train_ds = (ds.map(train_preprocessing).batch(batch_size, drop_remainder=True).shuffle(batch_size * 2).prefetch(tf.data.AUTOTUNE))
# plt.figure(figsize=(10, 10))
# for image in train_ds.take(1):
#     for i in range(1):
#         plt.imshow(image[i].numpy())
#         plt.axis('off')
# plt.show()

# ---------------------------
#
# train_ds = ds.map(train_preprocessing).batch(batch_size, drop_remainder=True).shuffle(batch_size * 2).prefetch(tf.data.AUTOTUNE)
#
# plt.figure(figsize=(10, 10))
# for image in train_ds.take(1):
#     for i in range(1):
#         plt.imshow(image.numpy().astype('uint8'))
#         plt.axis('off')
# plt.show()


class GaussianDiffusion:
    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000, clip_min=-1.0, clip_max=1.0):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        self.betas = tf.constant(betas, dtype=tf.float32)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.alphas_cumprod = tf.constant(alphas_cumprod, dtype=tf.float32)
        self.alphas_cumprod_prev = tf.constant(alphas_cumprod_prev, dtype=tf.float32)
        self.sqrt_alphas_cumprod = tf.constant(np.sqrt(alphas_cumprod), dtype=tf.float32)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(np.sqrt(1.0 - alphas_cumprod), dtype=tf.float32)
        self.log_one_minus_alphas_cumprod = tf.constant(np.log(1.0 - alphas_cumprod), dtype=tf.float32)

        # 根据 xt 倒算 x0 所需要的两个系数
        self.sqrt_recip_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod), dtype=tf.float32)
        self.sqrt_recipm1_alphas_cumprod = tf.constant(np.sqrt(1.0 / alphas_cumprod - 1), dtype=tf.float32)

        # 最后求出来的方差
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_variance = tf.constant(posterior_variance, dtype=tf.float32)
        self.posterior_log_variance_clipped = tf.constant(np.log(np.maximum(posterior_variance, 1e-20)), dtype=tf.float32)

        # 最后求出来的均值的两个系数
        self.posterior_mean_coef1 = tf.constant(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=tf.float32)
        self.posterior_mean_coef2 = tf.constant((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=tf.float32)

    def _extract(self, a, t, x_shape):
        batch_size = x_shape[0]
        out = tf.gather(a, t)  # [B]
        return tf.reshape(out, [batch_size, 1, 1, 1])

    # 前向过程在不同的时间步进行加噪
    def q_sample(self, x_start, t, noise):  # x_start [B,64,64,3], t [B] , noise [B,64,64,3]
        x_start_shape = tf.shape(x_start)
        a = self._extract(self.sqrt_alphas_cumprod, t, tf.shape(x_start)) * x_start  # [B,64,64,3]
        b = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise  # [B,64,64,3]
        return (a + b)

    # 从模型预测的噪声中倒推 x0
    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = tf.shape(x_t)
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise)

    def q_posterior(self, x_start, x_t, t):
        x_t_shape = tf.shape(x_t)
        posterior_mean = (self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract( self.posterior_log_variance_clipped, t, x_t_shape )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = tf.clip_by_value(x_recon, self.clip_min, self.clip_max)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # 已经知道当前 t 时刻预测的噪声后，计算 t-1 时刻的图片
    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise, x=x, t=t, clip_denoised=clip_denoised)
        noise = tf.random.normal(shape=x.shape, dtype=x.dtype)   # [16,64,64,3]
        nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [tf.shape(x)[0], 1, 1, 1])
        return model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise


def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")


class AttentionBlock(layers.Layer):
    def __init__(self, units, groups=8, **kwargs):
        self.units = units
        self.groups = groups
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        # self.norm = layers.GroupNormalization(groups=groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        scale = tf.cast(self.units, tf.float32) ** (-0.5)

        inputs = self.norm(inputs)
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attn_score = tf.einsum("bhwc, bHWc->bhwHW", q, k) * scale
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height * width])
        attn_score = tf.nn.softmax(attn_score, -1)
        attn_score = tf.reshape(attn_score, [batch_size, height, width, height, width])
        proj = tf.einsum("bhwHW,bHWc->bhwc", attn_score, v)
        proj = self.proj(proj)
        return inputs + proj


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        emb = inputs[:, None] * self.emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb


def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)
        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        # x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        x = layers.Add()([x, temb])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        # x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = layers.Add()([x, residual])
        return x
    return apply


def DownSample(width):  # ( i - k + 2 * p ) / s + 1
    def apply(x):
        x = layers.Conv2D(width, kernel_size=3, strides=2, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x
    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(units, activation=activation_fn, kernel_initializer=kernel_init(1.0))(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb
    return apply


def build_model(img_size, img_channels, widths, has_attention, num_res_blocks=2, norm_groups=8, interpolation='nearest', activation_fn=keras.activations.swish):
    image_input = layers.Input(shape=(img_size, img_size, img_channels), name="image_input")
    time_input = keras.Input(shape=(), dtype=tf.int64, name="time_input")
    x = layers.Conv2D(first_conv_channels, kernel_size=(3, 3), padding='same', kernel_initializer=kernel_init(1.0))(image_input)
    temb = TimeEmbedding(dim=first_conv_channels * 4)(time_input)
    temb = TimeMLP(units=first_conv_channels * 4, activation_fn=activation_fn)(temb)
    skips = [x]
    # DownBlock
    for i in range(len(widths)):
        for _ in range(num_res_blocks):
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
            skips.append(x)
        if widths[i] != widths[-1]:
            x = DownSample(widths[i])(x)
            skips.append(x)
    # MiddleBlock
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    x = AttentionBlock(widths[-1], groups=norm_groups)(x)
    x = ResidualBlock(widths[-1], groups=norm_groups, activation_fn=activation_fn)([x, temb])
    # UpBlock
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks + 1):
            x = layers.Concatenate(axis=-1)([x, skips.pop()])
            x = ResidualBlock(widths[i], groups=norm_groups, activation_fn=activation_fn)([x, temb])
            if has_attention[i]:
                x = AttentionBlock(widths[i], groups=norm_groups)(x)
        if i != 0:
            x = UpSample(widths[i], interpolation=interpolation)(x)
    # End block
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # x = layers.GroupNormalization(groups=norm_groups)(x)
    x = activation_fn(x)
    x = layers.Conv2D(3, (3, 3), padding="same", kernel_initializer=kernel_init(0.0))(x)
    return keras.Model([image_input, time_input], x, name="unet")


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema

    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)
        with tf.GradientTape() as tape:
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
            images_t = self.gdf_util.q_sample(images, t, noise)  # [B,64,64,3]
            pred_noise = self.network([images_t, t], training=True)  # [B,64,64,3]
            loss = self.loss(noise, pred_noise)
        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        return {"loss": loss}

    def generate_images(self, num_images=4):
        samples = tf.random.normal(shape=(num_images, img_size, img_size, img_channels), dtype=tf.float32)  # [16,64,64,3]
        for t in tqdm(reversed(range(0, self.timesteps))):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)  # [16]
            pred_noise = self.ema_network.predict([samples, tt], verbose=0, batch_size=num_images)  # [16,64,64,3]
            samples = self.gdf_util.p_sample(pred_noise, samples, tt, clip_denoised=True)
        return samples

    def plot_images(self, epoch=None, logs=None, num_rows=1, num_cols=1, figsize=(12, 5)):
        if epoch and  epoch % 101 == 0:
            start = time.time()
            print('\nstart plot_images...')
            generated_samples = self.generate_images(num_images=num_rows * num_cols)
            generated_samples = (tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0).numpy().astype(np.uint8))
            _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
            for i, image in enumerate(generated_samples):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(image)
            plt.tight_layout()
            plt.show()
            print('\nplot_images 耗时:', time.time() - start)
        elif not epoch:
            start = time.time()
            print('\nstart plot_images...')
            generated_samples = self.generate_images(num_images=num_rows * num_cols)
            generated_samples = (tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0).numpy().astype(np.uint8))
            _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
            for i, image in enumerate(generated_samples):
                plt.subplot(num_rows, num_cols, i + 1)
                plt.imshow(image)
            plt.tight_layout()
            plt.show()
            print('\nplot_images 耗时:', time.time() - start)


network = build_model(img_size=img_size, img_channels=img_channels, widths=widths, has_attention=has_attention, num_res_blocks=num_res_blocks, norm_groups=norm_groups, activation_fn=keras.activations.swish)
ema_network = build_model(img_size=img_size, img_channels=img_channels, widths=widths, has_attention=has_attention, num_res_blocks=num_res_blocks, norm_groups=norm_groups, activation_fn=keras.activations.swish)
ema_network.set_weights(network.get_weights())
gdf_util = GaussianDiffusion(timesteps=total_timesteps)
model = DiffusionModel(network=network, ema_network=ema_network, gdf_util=gdf_util, timesteps=total_timesteps)
model.compile(loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
model.fit(train_ds, epochs=num_epochs, batch_size=batch_size)
model.save_weights('./DenoisingModel/DenoisingModel')
model.plot_images(num_rows=4, num_cols=4)



# Epoch 1/800
# 15/15 [==============================] - 19s 356ms/step - loss: 0.9907
# Epoch 2/800
# 15/15 [==============================] - 5s 357ms/step - loss: 0.9657
# ...
# Epoch 718/800
# 15/15 [==============================] - 6s 363ms/step - loss: 0.0163
# Epoch 719/800
# 15/15 [==============================] - 6s 364ms/step - loss: 0.0198
# ...
# Epoch 799/800
# 15/15 [==============================] - 6s 359ms/step - loss: 0.0233
# Epoch 800/800
# 15/15 [==============================] - 6s 360ms/step - loss: 0.0174