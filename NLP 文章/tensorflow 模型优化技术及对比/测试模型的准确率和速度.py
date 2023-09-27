import tempfile
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import zipfile
import tensorflow_model_optimization as tfmot
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


def get_gzipped_model_size(file):
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)


def cal(pred, label):
    p = np.argmax(pred, axis=1)
    N = len(pred)
    b = np.equal(p, label)
    b = b.astype(float)
    s = np.sum(b)
    return s / N


def test(file, test_images, test_labels):
    model = tf.keras.models.load_model(file)
    start = time.time()
    result = model.predict(test_images, batch_size=batch_size, verbose=1)
    accuracy = cal(result, test_labels)
    size = get_gzipped_model_size(file)
    print(f'{file} 准确率为 {accuracy}%')
    print(f'{file} 耗时 {time.time() - start}s')
    print(f'{file} 文件大小为 {size} 字节')


def test1(file, test_images, test_labels):
    with tfmot.quantization.keras.quantize_scope():
        model = tf.keras.models.load_model(file)
        start = time.time()
        result = model.predict(test_images, batch_size=batch_size, verbose=1)
        accuracy = cal(result, test_labels)
        size = get_gzipped_model_size(file)
        print(f'{file} 准确率为 {accuracy}%')
        print(f'{file} 耗时 {time.time() - start}s')
        print(f'{file} 文件大小为 {size} 字节')


def evaluate_tfilte(file, test_images, test_labels):
    with open(file, 'rb') as f:
        model = f.read()
    interpreter = tf.lite.Interpreter(model_content=model)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.resize_tensor_input(input_index, [batch_size, 28, 28])
    interpreter.resize_tensor_input(output_index, [batch_size])
    interpreter.allocate_tensors()
    start = time.time()
    predictions = []
    test_images = test_images.astype(np.float32)
    for i in tqdm(range(0, len(test_images), batch_size)):
        batch = test_images[i:i + batch_size]
        interpreter.set_tensor(input_index, batch)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        predictions.extend(np.argmax(output(), axis=1))
    prediction_digits = np.array(predictions)
    accuracy = (prediction_digits == test_labels).mean()
    size = get_gzipped_model_size(file)
    print(f'{file.split(".")[0]} 准确率为 {accuracy}%')
    print(f'{file.split(".")[0]} 耗时 {time.time() - start}s')
    print(f'{file.split(".")[0]} 文件大小为 {size} 字节')



batch_size = 32
N = len(test_images) // batch_size * batch_size
test_images, test_labels = test_images[:batch_size], test_labels[:batch_size]
test('基准模型.h5', test_images, test_labels)
test('剪枝微调模型.h5', test_images, test_labels)
test1('量化感知微调模型.h5', test_images, test_labels)
evaluate_tfilte('剪枝优化+轻量化模型.tflite', test_images, test_labels)
evaluate_tfilte('量化感知优化+轻量化.tflite', test_images, test_labels)
evaluate_tfilte('剪枝优化+量化感知优化+轻量化.tflite', test_images, test_labels)


# 1/1 [==============================] - 0s 48ms/step
# 基准模型.h5 准确率为 0.96875%
# 基准模型.h5 耗时 0.07200002670288086s
# 基准模型.h5 文件大小为 78117 字节
# 剪枝微调模型.h5 准确率为 0.96875%
# 剪枝微调模型.h5 耗时 0.03799915313720703s
# 剪枝微调模型.h5 文件大小为 25584 字节
# 量化感知微调模型.h5 准确率为 1.0%
# 量化感知微调模型.h5 耗时 0.06202530860900879s
# 量化感知微调模型.h5 文件大小为 79774 字节
# 剪枝优化+轻量化模型 准确率为 0.9375%
# 剪枝优化+轻量化模型 耗时 0.004999876022338867s
# 剪枝优化+轻量化模型 文件大小为 24851 字节
# 量化优化+轻量化模型 准确率为 0.9375%
# 量化优化+轻量化模型 耗时 0.01199960708618164s
# 量化优化+轻量化模型 文件大小为 17690 字节
# 剪枝优化+量化优化+轻量化 准确率为 0.96875%
# 剪枝优化+量化优化+轻量化 耗时 0.0030007362365722656s
# 剪枝优化+量化优化+轻量化 文件大小为 8044 字节



# 312/312
# 基准模型.h5 准确率为 0.9699519230769231%
# 基准模型.h5 耗时 0.3200802803039551s
# 基准模型.h5 文件大小为 78117 字节
# 剪枝微调模型.h5 准确率为 0.9719551282051282%
# 剪枝微调模型.h5 耗时 0.27707648277282715s
# 剪枝微调模型.h5 文件大小为 25584 字节
# 量化感知微调模型.h5 准确率为 0.9731570512820513%
# 量化感知微调模型.h5 耗时 0.31646013259887695s
# 量化感知微调模型.h5 文件大小为 79774 字节
# 剪枝优化+轻量化模型 准确率为 0.9707532051282052%
# 剪枝优化+轻量化模型 耗时 0.28508996963500977s
# 剪枝优化+轻量化模型 文件大小为 24851 字节
# 量化感知优化+轻量化模型 准确率为 0.9777644230769231%
# 量化感知优化+轻量化模型 耗时 3.5285239219665527s
# 量化感知优化+轻量化模型 文件大小为 17690 字节
# 剪枝优化+量化感知优化+轻量化 准确率为 0.9723557692307693%
# 剪枝优化+量化感知优化+轻量化 耗时 0.46264100074768066s
# 剪枝优化+量化感知优化+轻量化 文件大小为 8044 字节