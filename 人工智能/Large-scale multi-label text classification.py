from keras import layers
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ast import literal_eval
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

arxiv_data = pd.read_csv( "https://github.com/soumik12345/multi-label-text-classification/releases/download/v0.2/arxiv_data.csv")
arxiv_data = arxiv_data[~arxiv_data["titles"].duplicated()]
arxiv_data_filtered = arxiv_data.groupby("terms").filter(lambda x: len(x) > 1)
print(arxiv_data_filtered["terms"].values[:3])
arxiv_data_filtered["terms"] = arxiv_data_filtered["terms"].apply(lambda x: literal_eval(x))
print(arxiv_data_filtered["terms"].values[:3])

test_split = 0.1
train_df, test_df = train_test_split(arxiv_data_filtered, test_size=test_split, stratify=arxiv_data_filtered["terms"].values)
val_df = test_df.sample(frac=0.5)
test_df.drop(val_df.index, inplace=True)
print(f"Number of rows in training set: {len(train_df)}")
print(f"Number of rows in validation set: {len(val_df)}")
print(f"Number of rows in test set: {len(test_df)}")

terms = tf.ragged.constant(train_df["terms"].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot")
lookup.adapt(terms)
vocab = lookup.get_vocabulary()
print("Vocab:", vocab)


sample_label = train_df["terms"].iloc[0]
print(f"Original label： {sample_label}")
label_binarized = lookup([sample_label])
print(f"Label-binarized representation: {label_binarized}")

max_length = 150
batch_size = 128
padding_token = "<pad>"
auto = tf.data.AUTOTUNE


def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["terms"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices((dataframe["summaries"].values, label_binarized))
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


train_dataset = make_dataset(train_df, is_train=True)
validatoin_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)


def invert_multi_hot(encoded_labels):
    hot_indices = np.argwhere(encoded_labels == 1.0)
    hot_indices = hot_indices[..., 0]
    return np.take(vocab, hot_indices)


text_batch, label_batch = next(iter(train_dataset))
for i, text in enumerate(text_batch[:3]):
    label = label_batch[i].numpy()
    label = label[None, ...]
    print("Abstract:", text)
    print("Label:", invert_multi_hot(label[0]))

vocabulary = set()
train_df["summaries"].str.lower().str.split().apply(vocabulary.update)
vocabulary_size = len(vocabulary)
print(vocabulary_size)

text_vectorizer = layers.TextVectorization(max_tokens=vocabulary_size, ngrams=2, output_mode="tf_idf")

with tf.device("/CPU:0"):
    text_vectorizer.adapt(train_dataset.map(lambda x, y: x))
train_dataset = train_dataset.map(lambda x, y: (text_vectorizer(x), y)).prefetch(auto)
validation_dataset = validatoin_dataset.map(lambda x, y: (text_vectorizer(x), y)).prefetch(auto)
test_dataset = test_dataset.map(lambda x, y: (text_vectorizer(x), y)).prefetch(auto)


def make_model():
    shallow_mlp_model = keras.Sequential([
                                        layers.Dense(512, activation="relu"),
                                        layers.Dense(256, activation="relu"),
                                        layers.Dense(lookup.vocabulary_size(), activation="sigmoid")])
    return shallow_mlp_model


epochs = 10
shallow_mlp_model = make_model()
shallow_mlp_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
history = shallow_mlp_model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)


def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_result("loss")
plot_result("binary_accuracy")


_, binary_acc = shallow_mlp_model.evaluate(test_dataset)
print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")


model_for_inference = keras.Sequential([text_vectorizer, shallow_mlp_model])
inference_dataset = make_dataset(test_df.sample(100), is_train=False)
text_batch, label_batch = next(iter(inference_dataset))
predicted_probabilities = model_for_inference.predict(text_batch)

for i,text in enumerate(text_batch[:5]):
    label = label_batch[i].numpy()[None, ...]
    print(f"Abstract: {text}")
    print(f"Label: {invert_multi_hot(label[0])}")
    top_3_labels = [ x for _,x in sorted(zip(predicted_probabilities[i], lookup.get_vocabulary()), key=lambda pair:pair[0], reverse=True)][:3]
    print(f"Predicted Label(s): ({', '.join([label for label in top_3_labels])}) \n")


# Connected to pydev debugger (build 232.9559.58)
# ["['cs.CV', 'cs.LG']" "['cs.CV', 'cs.AI', 'cs.LG']" "['cs.CV', 'cs.AI']"]
# [list(['cs.CV', 'cs.LG']) list(['cs.CV', 'cs.AI', 'cs.LG'])
#  list(['cs.CV', 'cs.AI'])]
# Number of rows in training set: 32985
# Number of rows in validation set: 1833
# Number of rows in test set: 1833
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Vocab: ['[UNK]', 'cs.CV', 'cs.LG', 'stat.ML', 'cs.AI', 'eess.IV', 'cs.RO', 'cs.CL', 'cs.NE', 'cs.CR', 'math.OC', 'eess.SP', 'cs.GR', 'cs.SI', 'cs.MM', 'cs.SY', 'cs.IR', 'cs.MA', 'eess.SY', 'cs.HC', 'math.IT', 'cs.IT', 'cs.DC', 'cs.CY', 'stat.AP', 'stat.TH', 'math.ST', 'stat.ME', 'eess.AS', 'cs.SD', 'q-bio.QM', 'q-bio.NC', 'cs.DS', 'cs.GT', 'cs.CG', 'cs.SE', 'cs.NI', 'I.2.6', 'stat.CO', 'math.NA', 'cs.NA', 'physics.chem-ph', 'cs.DB', 'q-bio.BM', 'cs.LO', 'cs.PL', 'cond-mat.dis-nn', '68T45', 'math.PR', 'physics.comp-ph', 'cs.CE', 'cs.AR', 'I.2.10', 'q-fin.ST', 'cond-mat.stat-mech', '68T05', 'quant-ph', 'math.DS', 'physics.data-an', 'cs.CC', 'I.4.6', 'physics.soc-ph', 'physics.ao-ph', 'cs.DM', 'econ.EM', 'q-bio.GN', 'physics.med-ph', 'astro-ph.IM', 'I.4.8', 'math.AT', 'cs.PF', 'I.4', 'q-fin.TR', 'cs.FL', 'I.5.4', 'I.2', '68U10', 'hep-ex', '68T10', 'physics.geo-ph', 'cond-mat.mtrl-sci', 'physics.optics', 'physics.flu-dyn', 'math.CO', 'math.AP', 'I.4; I.5', 'I.4.9', 'I.2.6; I.2.8', '68T01', '65D19', 'q-fin.CP', 'nlin.CD', 'cs.MS', 'I.2.6; I.5.1', 'I.2.10; I.4; I.5', 'I.2.0; I.2.6', '68T07', 'q-fin.GN', 'cs.SC', 'cs.ET', 'K.3.2', 'I.2.8', '68U01', '68T30', 'q-fin.EC', 'q-bio.MN', 'econ.GN', 'I.4.9; I.5.4', 'I.4.5', 'I.2; I.5', 'I.2; I.4; I.5', 'I.2.6; I.2.7', 'I.2.10; I.4.8', '68T99', '68Q32', '68', '62H30', 'q-fin.RM', 'q-fin.PM', 'q-bio.TO', 'q-bio.OT', 'physics.bio-ph', 'nlin.AO', 'math.LO', 'math.FA', 'hep-ph', 'cond-mat.soft', 'I.4.6; I.4.8', 'I.4.4', 'I.4.3', 'I.4.0', 'I.2; J.2', 'I.2; I.2.6; I.2.7', 'I.2.7', 'I.2.6; I.5.4', 'I.2.6; I.2.9', 'I.2.6; I.2.7; H.3.1; H.3.3', 'I.2.6; I.2.10', 'I.2.6, I.5.4', 'I.2.1; J.3', 'I.2.10; I.5.1; I.4.8', 'I.2.10; I.4.8; I.5.4', 'I.2.10; I.2.6', 'I.2.1', 'H.3.1; I.2.6; I.2.7', 'H.3.1; H.3.3; I.2.6; I.2.7', 'G.3', 'F.2.2; I.2.7', 'E.5; E.4; E.2; H.1.1; F.1.1; F.1.3', '68Txx', '62H99', '62H35', '14J60 (Primary) 14F05, 14J26 (Secondary)']
# Original label： ['cs.CV']
# Label-binarized representation: [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#   0. 0. 0. 0. 0. 0. 0. 0. 0.]]
# Abstract: tf.Tensor(b'Graph convolutional networks produce good predictions of unlabeled samples\ndue to its transductive label propagation. Since samples have different\npredicted confidences, we take high-confidence predictions as pseudo labels to\nexpand the label set so that more samples are selected for updating models. We\npropose a new training method named as mutual teaching, i.e., we train dual\nmodels and let them teach each other during each batch. First, each network\nfeeds forward all samples and selects samples with high-confidence predictions.\nSecond, each model is updated by samples selected by its peer network. We view\nthe high-confidence predictions as useful knowledge, and the useful knowledge\nof one network teaches the peer network with model updating in each batch. In\nmutual teaching, the pseudo-label set of a network is from its peer network.\nSince we use the new strategy of network training, performance improves\nsignificantly. Extensive experimental results demonstrate that our method\nachieves superior performance over state-of-the-art methods under very low\nlabel rates.', shape=(), dtype=string)
# Label: ['cs.CV' 'cs.LG' 'stat.ML']
# Abstract: tf.Tensor(b'Visual saliency is a fundamental problem in both cognitive and computational\nsciences, including computer vision. In this CVPR 2015 paper, we discover that\na high-quality visual saliency model can be trained with multiscale features\nextracted using a popular deep learning architecture, convolutional neural\nnetworks (CNNs), which have had many successes in visual recognition tasks. For\nlearning such saliency models, we introduce a neural network architecture,\nwhich has fully connected layers on top of CNNs responsible for extracting\nfeatures at three different scales. We then propose a refinement method to\nenhance the spatial coherence of our saliency results. Finally, aggregating\nmultiple saliency maps computed for different levels of image segmentation can\nfurther boost the performance, yielding saliency maps better than those\ngenerated from a single segmentation. To promote further research and\nevaluation of visual saliency models, we also construct a new large database of\n4447 challenging images and their pixelwise saliency annotation. Experimental\nresults demonstrate that our proposed method is capable of achieving\nstate-of-the-art performance on all public benchmarks, improving the F-Measure\nby 5.0% and 13.2% respectively on the MSRA-B dataset and our new dataset\n(HKU-IS), and lowering the mean absolute error by 5.7% and 35.1% respectively\non these two datasets.', shape=(), dtype=string)
# Label: ['cs.CV']
# Abstract: tf.Tensor(b'Lifelong learning is a very important step toward realizing robust autonomous\nartificial agents. Neural networks are the main engine of deep learning, which\nis the current state-of-the-art technique in formulating adaptive artificial\nintelligent systems. However, neural networks suffer from catastrophic\nforgetting when stressed with the challenge of continual learning. We\ninvestigate how to exploit modular topology in neural networks in order to\ndynamically balance the information load between different modules by routing\ninputs based on the information content in each module so that information\ninterference is minimized. Our dynamic information balancing (DIB) technique\nadapts a reinforcement learning technique to guide the routing of different\ninputs based on a reward signal derived from a measure of the information load\nin each module. Our empirical results show that DIB combined with elastic\nweight consolidation (EWC) regularization outperforms models with similar\ncapacity and EWC regularization across different task formulations and\ndatasets.', shape=(), dtype=string)
# Label: ['cs.LG' 'stat.ML' 'cs.NE']
# 153419
# Epoch 1/20
# 2023-10-26 14:47:17.863420: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
# 258/258 [==============================] - 8s 25ms/step - loss: 0.0334 - binary_accuracy: 0.9890 - val_loss: 0.0190 - val_binary_accuracy: 0.9941
# Epoch 2/20
# 258/258 [==============================] - 6s 25ms/step - loss: 0.0031 - binary_accuracy: 0.9991 - val_loss: 0.0262 - val_binary_accuracy: 0.9938
# Epoch 3/20
# 258/258 [==============================] - 9s 36ms/step - loss: 8.7633e-04 - binary_accuracy: 0.9998 - val_loss: 0.0322 - val_binary_accuracy: 0.9937
# Epoch 4/20
# 258/258 [==============================] - 6s 24ms/step - loss: 3.9149e-04 - binary_accuracy: 0.9999 - val_loss: 0.0381 - val_binary_accuracy: 0.9938
# Epoch 5/20
# 258/258 [==============================] - 7s 26ms/step - loss: 2.2129e-04 - binary_accuracy: 1.0000 - val_loss: 0.0406 - val_binary_accuracy: 0.9938
# Epoch 6/20
# 258/258 [==============================] - 9s 36ms/step - loss: 1.2343e-04 - binary_accuracy: 1.0000 - val_loss: 0.0428 - val_binary_accuracy: 0.9937
# Epoch 7/20
# 258/258 [==============================] - 9s 37ms/step - loss: 8.3425e-05 - binary_accuracy: 1.0000 - val_loss: 0.0443 - val_binary_accuracy: 0.9937
# Epoch 8/20
# 258/258 [==============================] - 9s 36ms/step - loss: 6.5395e-05 - binary_accuracy: 1.0000 - val_loss: 0.0459 - val_binary_accuracy: 0.9937
# Epoch 9/20
# 258/258 [==============================] - 9s 36ms/step - loss: 5.5046e-05 - binary_accuracy: 1.0000 - val_loss: 0.0488 - val_binary_accuracy: 0.9938
# Epoch 10/20
# 258/258 [==============================] - 9s 36ms/step - loss: 4.7955e-05 - binary_accuracy: 1.0000 - val_loss: 0.0488 - val_binary_accuracy: 0.9939
# Epoch 11/20
# 258/258 [==============================] - 7s 25ms/step - loss: 3.4329e-05 - binary_accuracy: 1.0000 - val_loss: 0.0495 - val_binary_accuracy: 0.9938
# Epoch 12/20
# 258/258 [==============================] - 6s 24ms/step - loss: 3.3442e-05 - binary_accuracy: 1.0000 - val_loss: 0.0520 - val_binary_accuracy: 0.9939
# Epoch 13/20
# 258/258 [==============================] - 6s 24ms/step - loss: 2.9730e-05 - binary_accuracy: 1.0000 - val_loss: 0.0520 - val_binary_accuracy: 0.9939
# Epoch 14/20
# 258/258 [==============================] - 6s 24ms/step - loss: 2.8862e-05 - binary_accuracy: 1.0000 - val_loss: 0.0521 - val_binary_accuracy: 0.9939
# Epoch 15/20
# 258/258 [==============================] - 6s 24ms/step - loss: 3.6246e-05 - binary_accuracy: 1.0000 - val_loss: 0.0524 - val_binary_accuracy: 0.9940
# Epoch 16/20
# 258/258 [==============================] - 6s 24ms/step - loss: 2.5106e-04 - binary_accuracy: 0.9999 - val_loss: 0.0359 - val_binary_accuracy: 0.9933
# Epoch 17/20
# 258/258 [==============================] - 6s 24ms/step - loss: 0.0017 - binary_accuracy: 0.9995 - val_loss: 0.0409 - val_binary_accuracy: 0.9932
# Epoch 18/20
# 258/258 [==============================] - 6s 24ms/step - loss: 0.0023 - binary_accuracy: 0.9993 - val_loss: 0.0447 - val_binary_accuracy: 0.9931
# Epoch 19/20
# 258/258 [==============================] - 6s 24ms/step - loss: 0.0017 - binary_accuracy: 0.9995 - val_loss: 0.0469 - val_binary_accuracy: 0.9930
# Epoch 20/20
# 258/258 [==============================] - 6s 24ms/step - loss: 7.4884e-04 - binary_accuracy: 0.9998 - val_loss: 0.0550 - val_binary_accuracy: 0.9931
# 15/15 [==============================] - 1s 28ms/step - loss: 0.0552 - binary_accuracy: 0.9932
# Categorical accuracy on the test set: 99.32%.
# 4/4 [==============================] - 0s 26ms/step
# Abstract: b'Graph representation learning is a fundamental problem for modeling\nrelational data and benefits a number of downstream applications. Traditional\nBayesian-based graph models and recent deep learning based GNN either suffer\nfrom impracticability or lack interpretability, thus combined models for\nundirected graphs have been proposed to overcome the weaknesses. As a large\nportion of real-world graphs are directed graphs (of which undirected graphs\nare special cases), in this paper, we propose a Deep Latent Space Model (DLSM)\nfor directed graphs to incorporate the traditional latent variable based\ngenerative model into deep learning frameworks. Our proposed model consists of\na graph convolutional network (GCN) encoder and a stochastic decoder, which are\nlayer-wise connected by a hierarchical variational auto-encoder architecture.\nBy specifically modeling the degree heterogeneity using node random factors,\nour model possesses better interpretability in both community structure and\ndegree heterogeneity. For fast inference, the stochastic gradient variational\nBayes (SGVB) is adopted using a non-iterative recognition model, which is much\nmore scalable than traditional MCMC-based methods. The experiments on\nreal-world datasets show that the proposed model achieves the state-of-the-art\nperformances on both link prediction and community detection tasks while\nlearning interpretable node embeddings. The source code is available at\nhttps://github.com/upperr/DLSM.'
# Label: ['cs.LG' 'stat.ML']
# Predicted Label(s): (cs.LG, stat.ML, cs.AI)
#
# Abstract: b'In recent years, there has been a rapid progress in solving the binary\nproblems in computer vision, such as edge detection which finds the boundaries\nof an image and salient object detection which finds the important object in an\nimage. This progress happened thanks to the rise of deep-learning and\nconvolutional neural networks (CNN) which allow to extract complex and abstract\nfeatures. However, edge detection and saliency are still two different fields\nand do not interact together, although it is intuitive for a human to detect\nsalient objects based on its boundaries. Those features are not well merged in\na CNN because edges and surfaces do not intersect since one feature represents\na region while the other represents boundaries between different regions. In\nthe current work, the main objective is to develop a method to merge the edges\nwith the saliency maps to improve the performance of the saliency. Hence, we\ndeveloped the gradient-domain merging (GDM) which can be used to quickly\ncombine the image-domain information of salient object detection with the\ngradient-domain information of the edge detection. This leads to our proposed\nsaliency enhancement using edges (SEE) with an average improvement of the\nF-measure of at least 3.4 times higher on the DUT-OMRON dataset and 6.6 times\nhigher on the ECSSD dataset, when compared to competing algorithm such as\ndenseCRF and BGOF. The SEE algorithm is split into 2 parts, SEE-Pre for\npreprocessing and SEE-Post pour postprocessing.'
# Label: ['cs.CV']
# Predicted Label(s): (cs.CV, I.4.9, cs.LG)
#
# Abstract: b'Despite the remarkable advances in visual saliency analysis for natural scene\nimages (NSIs), salient object detection (SOD) for optical remote sensing images\n(RSIs) still remains an open and challenging problem. In this paper, we propose\nan end-to-end Dense Attention Fluid Network (DAFNet) for SOD in optical RSIs. A\nGlobal Context-aware Attention (GCA) module is proposed to adaptively capture\nlong-range semantic context relationships, and is further embedded in a Dense\nAttention Fluid (DAF) structure that enables shallow attention cues flow into\ndeep layers to guide the generation of high-level feature attention maps.\nSpecifically, the GCA module is composed of two key components, where the\nglobal feature aggregation module achieves mutual reinforcement of salient\nfeature embeddings from any two spatial locations, and the cascaded pyramid\nattention module tackles the scale variation issue by building up a cascaded\npyramid framework to progressively refine the attention map in a coarse-to-fine\nmanner. In addition, we construct a new and challenging optical RSI dataset for\nSOD that contains 2,000 images with pixel-wise saliency annotations, which is\ncurrently the largest publicly available benchmark. Extensive experiments\ndemonstrate that our proposed DAFNet significantly outperforms the existing\nstate-of-the-art SOD competitors. https://github.com/rmcong/DAFNet_TIP20'
# Label: ['cs.CV']
# Predicted Label(s): (cs.CV, cs.LG, I.4.9)
#
# Abstract: b"We present a novel approach for the reconstruction of dynamic geometric\nshapes using a single hand-held consumer-grade RGB-D sensor at real-time rates.\nOur method does not require a pre-defined shape template to start with and\nbuilds up the scene model from scratch during the scanning process. Geometry\nand motion are parameterized in a unified manner by a volumetric representation\nthat encodes a distance field of the surface geometry as well as the non-rigid\nspace deformation. Motion tracking is based on a set of extracted sparse color\nfeatures in combination with a dense depth-based constraint formulation. This\nenables accurate tracking and drastically reduces drift inherent to standard\nmodel-to-depth alignment. We cast finding the optimal deformation of space as a\nnon-linear regularized variational optimization problem by enforcing local\nsmoothness and proximity to the input constraints. The problem is tackled in\nreal-time at the camera's capture rate using a data-parallel flip-flop\noptimization strategy. Our results demonstrate robust tracking even for fast\nmotion and scenes that lack geometric features."
# Label: ['cs.CV']
# Predicted Label(s): (cs.CV, cs.GR, cs.CG)
#
# Abstract: b'Latent fingerprint recognition is not a new topic but it has attracted a lot\nof attention from researchers in both academia and industry over the past 50\nyears. With the rapid development of pattern recognition techniques, automated\nfingerprint identification systems (AFIS) have become more and more ubiquitous.\nHowever, most AFIS are utilized for live-scan or rolled/slap prints while only\na few systems can work on latent fingerprints with reasonable accuracy. The\nquestion of whether taking higher resolution scans of latent fingerprints and\ntheir rolled/slap mate prints could help improve the identification accuracy\nstill remains an open question in the forensic community. Because pores are one\nof the most reliable features besides minutiae to identify latent fingerprints,\nwe propose an end-to-end automatic pore extraction and matching system to\nanalyze the utility of pores in latent fingerprint identification. Hence, this\npaper answers two questions in the latent fingerprint domain: (i) does the\nincorporation of pores as level-3 features improve the system performance\nsignificantly? and (ii) does the 1,000 ppi image resolution improve the\nrecognition results? We believe that our proposed end-to-end pore extraction\nand matching system will be a concrete baseline for future latent AFIS\ndevelopment.'
# Label: ['cs.CV']
# Predicted Label(s): (cs.CV, cs.LG, q-bio.QM)