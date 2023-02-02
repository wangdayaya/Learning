from numpy import array
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

data = list('中国人')
N = len(data)
values = array(data)
print(values)
# 索引化
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# one-hot 向量化
encoded = tf.one_hot(integer_encoded, N)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for c, i, v in zip(data, integer_encoded, encoded.eval()):
        print("token：%s，索引：%d，one-hot 向量：[%s]" % (c, i, " ".join([str(t) for t in v])))
