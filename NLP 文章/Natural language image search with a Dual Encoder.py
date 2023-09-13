import collections
import json
import os
import pickle

import tensorflow_hub as hub
import keras.applications
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
from keras.applications.xception import preprocess_input
import tensorflow_text as text
import matplotlib.image as mpimg

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# --------------------------数据处理--------------------------------
tf.get_logger().setLevel("ERROR")
root_dir = "datasets"
annotations_dir = os.path.join(root_dir, "annotations")
images_dir = os.path.join(root_dir, "train2014")
tfrecords_dir = os.path.join(root_dir, "tfrecords")
annotation_file = os.path.join(annotations_dir, "captions_train2014.json")

if not os.path.exists(annotations_dir):
    annotation_zip = tf.keras.utils.get_file( "captions.zip", cache_dir=os.path.abspath("."),  origin="http://images.cocodataset.org/annotations/annotations_trainval2014.zip", extract=True,)
    os.remove(annotation_zip)

# Download image files
if not os.path.exists(images_dir):
    image_zip = tf.keras.utils.get_file("train2014.zip", cache_dir=os.path.abspath("."), origin="http://images.cocodataset.org/zips/train2014.zip",extract=True,)
    os.remove(image_zip)

print("Dataset is downloaded and extracted successfully.")

with open(annotation_file, "r") as f:
    annotations = json.load(f)["annotations"]

image_path_to_caption = collections.defaultdict(list)
for element in annotations:
    caption = f"{element['caption'].lower().rstrip('.')}"
    image_path = images_dir + "/COCO_train2014_" + "%012d.jpg" % (element["image_id"])
    image_path_to_caption[image_path].append(caption)

image_paths = list(image_path_to_caption.keys())
print(f"Number of images: {len(image_paths)}")


train_size = len(image_paths)
valid_size = int(len(image_paths) * 0.05)
captions_per_image = 5
images_per_file = 5000
train_image_paths = image_paths[:train_size]
num_train_files = int(np.ceil(train_size / images_per_file))
train_files_prefix = os.path.join(tfrecords_dir, "train")
valid_image_paths = image_paths[-valid_size:]
num_valid_files = int(np.ceil(valid_size / images_per_file))
valid_files_prefix = os.path.join(tfrecords_dir, "valid")
tf.io.gfile.makedirs(tfrecords_dir)
print(f'{train_size} train examples')
print(f'{valid_size} valid examples')


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_example(image_path, caption):
    feature = { "caption": bytes_feature(caption.encode()), "raw_image": bytes_feature(tf.io.read_file(image_path).numpy()),}
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(file_name, image_paths):
    caption_list = []
    image_path_list = []
    for image_path in image_paths:
        captions = image_path_to_caption[image_path][:captions_per_image]
        caption_list.extend(captions)
        image_path_list.extend([image_path] * len(captions))

    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx in range(len(image_path_list)):
            example = create_example( image_path_list[example_idx], caption_list[example_idx] )
            writer.write(example.SerializeToString())
    return example_idx + 1


def write_data(image_paths, num_files, files_prefix):
    example_counter = 0
    for file_idx in tqdm(range(num_files)):
        file_name = files_prefix + "-%02d.tfrecord" % (file_idx)
        start_idx = images_per_file * file_idx
        end_idx = start_idx + images_per_file
        example_counter += write_tfrecords(file_name, image_paths[start_idx:end_idx])
    return example_counter


train_example_count = write_data(train_image_paths, num_train_files, train_files_prefix)
print(f"{train_example_count} training examples were written to tfrecord files.")

valid_example_count = write_data(valid_image_paths, num_valid_files, valid_files_prefix)
print(f"{valid_example_count} evaluation examples were written to tfrecord files.")


feature_description = { "caption": tf.io.FixedLenFeature([], tf.string), "raw_image": tf.io.FixedLenFeature([], tf.string),}


def read_example(example):
    features = tf.io.parse_single_example(example, feature_description)
    raw_image = features.pop("raw_image")
    features["image"] = tf.image.resize(tf.image.decode_jpeg(raw_image, channels=3), size=(299, 299))
    return features


def get_dataset(file_pattern, batch_size):
    return ( tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern)).map( read_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False, ).shuffle(batch_size * 10).prefetch(buffer_size=tf.data.AUTOTUNE).batch(batch_size))

# --------------------------模型结构--------------------------------
def project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate):
    projected_embeddings = layers.Dense(projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings


def create_vision_encoder(num_projection_layers, projection_dims, dropout_rate, trainable=False):
    xception = keras.applications.Xception(include_top=False, weights='imagenet', pooling='avg')
    for layer in xception.layers:
        layer.trainable = trainable
    inputs = layers.Input(shape=(299,299,3), name='image_input')
    xception_input = preprocess_input(inputs)
    embeddings = xception(xception_input)
    outputs = project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate)  # [B, 128]
    return keras.Model(inputs, outputs, name='vision_encoder')


def create_text_encoder(num_projection_layers, projection_dims, dropout_rate, trainable=False):
    bert = hub.KerasLayer( "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2", trainable=trainable )
    preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    inputs = layers.Input(shape=(), dtype=tf.string, name='text_input')
    bert_inputs = preprocess(inputs)
    embeddings = bert(bert_inputs)['pooled_output']
    outputs = project_embeddings(embeddings, num_projection_layers, projection_dims, dropout_rate)  # [B, 128]
    return keras.Model(inputs, outputs, name='text_encoder')


vision_encoder = create_vision_encoder( num_projection_layers=1, projection_dims=256, dropout_rate=0.1)
text_encoder = create_text_encoder( num_projection_layers=1, projection_dims=256, dropout_rate=0.1)

class DualEncoder(keras.Model):
    def __init__(self, text_endocer, image_encoder, temperature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_endocer
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name='loss')

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        caption_embeddings = text_encoder(features['caption'], training=training)
        image_embeddings = vision_encoder(features['image'], training=training)
        return caption_embeddings, image_embeddings

    def compute_loss(self, caption_embeddings, image_embeddings):
        logits = ( tf.matmul(caption_embeddings, image_embeddings, transpose_b=True) / self.temperature)   # [B, B]
        images_similarity = tf.matmul(image_embeddings, image_embeddings, transpose_b=True)   # [B, B]
        caption_similarity = tf.matmul(caption_embeddings, caption_embeddings, transpose_b=True)  # [B, B]
        targets = keras.activations.softmax( (caption_similarity + images_similarity) / (2 * self.temperature))  # [B, B]
        caption_loss = keras.losses.categorical_crossentropy(y_true=targets, y_pred=logits, from_logits=True)  # [B,]
        images_loss = keras.losses.categorical_crossentropy(y_true=tf.transpose(targets), y_pred=tf.transpose(logits), from_logits=True)  # [B,]
        return (caption_loss + images_loss) / 2

    def train_step(self, features):
        with tf.GradientTape() as tape:
            caption_embeddings, image_embeddings = self(features, training=True)
            loss = self.compute_loss(caption_embeddings, image_embeddings)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        caption_embeddings, image_embeddings = self(features, training=False)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
print(text_encoder.summary())
print(vision_encoder.summary())
dual_encoder = DualEncoder(text_encoder, vision_encoder, temperature=0.05)
dual_encoder.compile(optimizer=tfa.optimizers.AdamW(learning_rate=0.0003, weight_decay=0.001))

# --------------------------训练--------------------------------
batch_size = 150
epochs = 100
print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")

root_dir = "datasets"
tfrecords_dir = os.path.join(root_dir, "tfrecords")
train_dataset = get_dataset(os.path.join(tfrecords_dir, "train-*.tfrecord"), batch_size)
valid_dataset = get_dataset(os.path.join(tfrecords_dir, "valid-*.tfrecord"), batch_size)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = dual_encoder.fit(train_dataset, epochs=epochs, validation_data=valid_dataset, callbacks=[reduce_lr, early_stopping])
print("Training completed. Saving vision and text encoders...")
vision_encoder.save("vision_encoder")
text_encoder.save("text_encoder")
print("Models are saved.")

import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["train", "valid"], loc="upper right")
plt.show()


# --------------------------测试效果--------------------------------


root_dir = "datasets"
print('加载文本和图像模型')
vision_encoder = keras.models.load_model('vision_encoder')
text_encoder = keras.models.load_model('text_encoder')


def read_img(image_path):
    image_array = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return tf.image.resize(image_array, (299, 299))

batch_size=256
images_dir = os.path.join(root_dir, "train2014")
annotations_dir = os.path.join(root_dir, "annotations")
annotation_file = os.path.join(annotations_dir, "captions_train2014.json")
with open(annotation_file, "r") as f:
    annotations = json.load(f)["annotations"]
image_path_to_caption = collections.defaultdict(list)
for element in annotations:
    caption = f"{element['caption'].lower().rstrip('.')}"
    image_path = images_dir + "/COCO_train2014_" + "%012d.jpg" % (element["image_id"])
    image_path_to_caption[image_path].append(caption)
image_paths = list(image_path_to_caption.keys())

# 执行一次后保存好向量，后面就可以注释掉
image_embeddings = vision_encoder.predict( tf.data.Dataset.from_tensor_slices(image_paths).map(read_img).batch(batch_size), verbose=1)
print(f"Image embeddings shape: {image_embeddings.shape}.")
with open('image_embeddings.pkl', 'wb') as file:
    pickle.dump(image_embeddings, file)
# 执行一次后保存好向量，后面就可以注释掉

with open('image_embeddings.pkl', 'rb') as file:
    image_embeddings = pickle.load(file)

def find_matches(image_embeddings, queries, k=20, normalize=True):
    query_embedding = text_encoder(tf.convert_to_tensor(queries))
    if normalize:
        image_embeddings = tf.math.l2_normalize(image_embeddings, axis=1)  # [82734, 64]
        query_embedding = tf.math.l2_normalize(query_embedding, axis=1)  # [1, 64]
    dot_similarity = tf.matmul(query_embedding, image_embeddings, transpose_b=True)
    results = tf.math.top_k(dot_similarity, k).indices.numpy()
    return [[image_paths[idx] for idx in indices] for indices in results]


query = "A man wearing a blue hat brushing his teeth with a giant bunny behind him."
matches = find_matches(image_embeddings, [query], normalize=True)[0]
plt.figure(figsize=(20, 20))
for i in range(20):
    ax = plt.subplot(4, 5, i + 1)
    print(matches[i])
    plt.imshow(mpimg.imread(matches[i]))
    plt.axis("off")
plt.show()

# 30000 个样本
# Epoch 1/5
# 235/235 [==============================] - 986s 4s/step - loss: 36.4844 - val_loss: 9.7046 - lr: 3.0000e-04
# Epoch 2/5
# 235/235 [==============================] - 973s 4s/step - loss: 6.7708 - val_loss: 5.8190 - lr: 3.0000e-04
# Epoch 3/5
# 235/235 [==============================] - 980s 4s/step - loss: 4.6690 - val_loss: 5.2683 - lr: 3.0000e-04
# Epoch 4/5
# 235/235 [==============================] - 972s 4s/step - loss: 4.1320 - val_loss: 4.8496 - lr: 3.0000e-04
# Epoch 5/5
# 235/235 [==============================] - 965s 4s/step - loss: 3.7614 - val_loss: 4.5517 - lr: 3.0000e-04



# 全部样本
# Epoch 1/100
# 2760/2760 [==============================] - 3521s 1s/step - loss: 10.6863 - val_loss: 3.3784 - lr: 3.0000e-04
# Epoch 2/100
# 2760/2760 [==============================] - 3503s 1s/step - loss: 7.5042 - val_loss: 15.8172 - lr: 3.0000e-04
# Epoch 3/100
# 2760/2760 [==============================] - 3502s 1s/step - loss: 12.3186 - val_loss: 4.3896 - lr: 3.0000e-04
# Epoch 4/100
# 2760/2760 [==============================] - 3505s 1s/step - loss: 11.6067 - val_loss: 30.6790 - lr: 3.0000e-04