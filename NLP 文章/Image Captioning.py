import os, re
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import efficientnet
from keras.layers import TextVectorization

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

IMAGE_PATH = "Flicker8k_Dataset"
IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 25
EMBED_DIM = 512
FF_DIM = 512
BATCH_SIZE = 64
EPOCHS = 200
AUTOTUNE = tf.data.AUTOTUNE


def load_captions_data(filename):
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()
        for line in caption_data:
            line = line.rstrip('\n')
            img_name, caption = line.split('\t')
            img_name = img_name.split('#')[0]
            img_name = os.path.join(IMAGE_PATH, img_name.strip())
            tokens = caption.strip().split()
            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue
            if img_name.endswith('jpg') and img_name not in images_to_skip:
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)
                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]
        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]
        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    all_images = list(caption_data.keys())
    if shuffle:
        np.random.shuffle(all_images)
    train_size = int(len(caption_data) * train_size)
    train_data = {img_name: caption_data[img_name] for img_name in all_images[:train_size]}
    validation_data = {img_name: caption_data[img_name] for img_name in all_images[train_size:]}
    return train_data, validation_data


captions_mapping, text_data = load_captions_data("Flickr8k_text/Flickr8k.token.txt")
train_data, valid_data = train_val_split(captions_mapping)
print(f"训练集有 {len(train_data)} 个，测试集有 {len(valid_data)} 个")


def custom_standardization(input_sting):
    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")
    lowercase = tf.strings.lower(input_sting)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


vectorization = TextVectorization(max_tokens=VOCAB_SIZE, output_mode='int', output_sequence_length=SEQ_LENGTH,
                                  standardize=custom_standardization)
vectorization.adapt(text_data)
image_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.2), layers.RandomContrast(0.3)])


def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions):
    a = decode_and_resize(img_path)
    b = vectorization(captions)
    return a,  b


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))


# --------------------------- model ---------------------------
def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model.output.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model  # [B, 100, 1280]


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation='relu')

    def call(self, inputs, training, mask=None):  # [B, 100, 1280]
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)  # [B, 100, 512]
        attention_output_1 = self.attention_1(query=inputs, key=inputs, value=inputs, attention_mask=None,  training=training)  # [B, 100, 512]
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1  # [B, 100, 512]


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.seqence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):  # [B, 24]
        embedded_tokens = self.token_embeddings(inputs)   # [B, 24, 512]
        embedded_tokens = embedded_tokens * self.embed_scale
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)  # [24, 512]
        return embedded_tokens + embedded_positions   # [B, 24, 512]

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.ffn_layer_1 = layers.Dense(ff_dim, activation='relu')
        self.ffn_layer_2 = layers.Dense(embed_dim)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE)
        self.out = layers.Dense(VOCAB_SIZE, activation='softmax')
        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.support_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)  # [B, 24, 512]
        causal_mask = self.get_causal_attention_mask(inputs)   # [B, 24, 24]
        if mask is not None:  # [B, 24]
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)   # [B, 24, 1]
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)  # [B, 1, 24]
            combined_mask = tf.minimum(combined_mask, causal_mask)   # [B, 24, 24] 既是 padding 表示有内容，又要让当前步后面的内容隐藏
        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=combined_mask, training=training)   # [B, 24, 512]
        out_1 = self.layernorm_1(inputs + attention_output_1)   # [B, 24, 512]
        attention_output_2 = self.attention_2(query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask, training=training)   # [B, 24, 512]
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)    # [B, 24, 10000]
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)  # [B, 24, 512]
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]  # [24,1]
        j = tf.range(sequence_length)  # [24]
        mask = tf.cast(i >= j, dtype="int32")  # [24, 24]
        mask = tf.reshape(mask, (1, sequence_length, sequence_length))  # [1, 24, 24]
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)  # [B, 1, 1]
        return tf.tile(mask, mult)  # [B, 24, 24]


class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model, encoder, decoder ,num_captions_per_image=5, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean("loss")
        self.acc_tracker = keras.metrics.Mean("accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)  # [B, 24]
        mask = tf.cast(mask, dtype=loss.dtype)  # [B, 24]
        loss *= mask  # [B, 24]
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)  # [B, 100, 512]
        batch_seq_inp = batch_seq[:, :-1]  # [B, 24]
        batch_seq_true = batch_seq[:, 1:]  # [B, 24]
        mask = tf.math.not_equal(batch_seq_true, 0)  # [B, 24]
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=training, mask=mask)  # [B, 24, 10000]
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss , acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data  # [B, 299,299,3] , [B, 5, 25]
        batch_loss = 0
        batch_acc = 0
        if self.image_aug:
            batch_img = self.image_aug(batch_img)
        img_embed = self.cnn_model(batch_img)
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss,acc = self._compute_caption_loss_and_acc(img_embed, batch_seq[:, i, :], training=True)
                batch_loss += loss
                batch_acc += acc
            train_vars = (self.encoder.trainable_variables + self.decoder.trainable_variables)
            grads = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(grads, train_vars))
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        return {"损失":self.loss_tracker.result(), "准确率": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0
        img_embed = self.cnn_model(batch_img)
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(img_embed, batch_seq[:,i,:], training=False)
            batch_loss += loss
            batch_acc += acc
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        return {"损失": self.loss_tracker.result(), "准确率": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]


cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=8)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=8)
decoder(tf.random.normal((64, 24)), tf.random.normal((64, 100, 512)), training=True, mask=tf.random.normal((64, 24)))
caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation)

# ----------------------- compile and fit ---------------------------
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond( global_step < warmup_steps, lambda: warmup_learning_rate,  lambda: self.post_warmup_learning_rate, )


num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
caption_model.compile(optimizer=keras.optimizers.Adam(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)), loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none"))
caption_model.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset,
                  callbacks=[keras.callbacks.EarlyStopping(patience=3, monitor='val_损失')])
tf.saved_model.save(caption_model,'image_caption_model')

# ----------------------- test ---------------------------
vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())


def generate_caption():
    sample_img = np.random.choice(valid_images)
    print("Image :", sample_img)
    sample_img = decode_and_resize(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)
    encoded_img = caption_model.encoder(img, training=False)
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder( tokenized_caption, encoded_img, training=False, mask=mask )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token
    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption: ", decoded_caption)


generate_caption()
generate_caption()
generate_caption()

