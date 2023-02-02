from numpy import array
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = list('中国人')
values = array(data)
# 索引化
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# one-hot 向量化
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
for c, i, v in zip(data, integer_encoded, onehot_encoded):
    print("token：%s，索引：%d，one-hot 向量：[%s]" % (c, i[0], " ".join([str(t) for t in v])))
