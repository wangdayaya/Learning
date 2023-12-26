### 知识准备


`AutoEncoder` 是一种用于数据降维和特征提取的无监督学习模型，它由一个 `encoder` 和一个 `decoder` 组成。 `encoder` 将输入数据转换为一个潜在空间的向量，而 `decoder` 将这个向量转换回输入的数据。这个模型可以学习到数据的紧凑表示，将高维的输入数据转换为低维的潜在空间向量，并且可以用于数据的压缩、去噪、特征提取等多种任务。

在训练过程中，我们的目标是尽可能减小输入数据和解码数据之间的差异，通常使用均方误差 MSE 作为损失函数。通过反向传播算法更新参数，最终得到  `encoder` 和 `decoder` 的权重。在实际使用中我们一般使用到的是训练好的 `encoder`  。

![v2-86cbd5045efdbebec7961c1cce619fcd_r.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8716c02e8f264717897f2e91cb3adab1~tplv-k3u1fbpfcp-watermark.image?)

### 数据准备
这里从 `CelebA` 数据集中加载一些图像数据并将其预处理成可以用于机器学习模型的形式。具体如下：

1.  从 `datasets` 库中加载 `CelebA` 数据集的训练集数据。
1.  选择数据集中的 `3000` 张图像进行处理，主要进行了图像的 `resize` 和归一化，将像素值范围缩放到[-1,1]之间。
1.  将处理后的数据存储在 `numpy` 数组中并返回。

`getLoader()` 函数的作用是返回一个的` DataLoader` 对象，该对象可以用于将数据集中的图像数据进行批量处理，方便训练神经网络模型。

```
import time
from datasets import load_dataset
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
H, W = 64, 64
batch_size = 64
def get_data():
    print('load and handle data ...')
    dataset = load_dataset('lansinuote/gen.1.celeba', split='train')
    N = min(dataset.shape[0], 3000)
    dataset = dataset.shuffle(seed=0).select(range(N))
    def f(d):
        images = d['image']
        d = []
        for img in images:
            img = img.resize((H, W))
            img = np.array(img)
            img = (img - 127.5) / 127.5
            img = img.transpose(2, 0, 1)
            d.append(img)
        return {'image': d}
    dataset = dataset.map(function=f,  batched=True, batch_size=1000,  remove_columns=list(dataset.features)[1:])
    data = np.empty((N, 3, H, W), dtype=np.float32)
    for i in tqdm(range(len(dataset))):
        data[i] = dataset[i]['image']
    return data

def getLoader():
    return torch.utils.data.DataLoader(dataset=get_data(),  batch_size=batch_size, shuffle=True,  drop_last=True)
```

`show()` 函数是一个工具函数，作用是将传入的图像数据可视化，以便于观察和调试。具体来说，该函数将 25 张图像按照 `5x5` 的网格排列，并最后将绘制的图像保存在文件中。
```
def show(images):
    if type(images) == torch.Tensor:
        images = images.to('cpu').detach().numpy()
    images = images[:25]
    plt.figure(figsize=(5, 5))
    for i in range(len(images)):
        image = images[i]
        image = image.transpose(1, 2, 0)
        image = (image + 1) / 2
        plt.subplot(4, 5, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.savefig(f"image_{int(time.time())}.png")
    plt.pause(1)
    plt.close()
```
### 模型定义


这里定义了一个 `Block` 类，该类是一个 `CNN` 包装集成的块，包含了卷积神经网络中的若干层卷积、批归一化、`LeakyReLU` 激活函数以及残差连接等操作，用于搭建后续的 `Encoder` 或 `Decoder` 模块。具体如下：

1.  `Block` 类的构造函数` __init__()` 包含两个参数 `dim_in` 和 `dim_out`，分别表示输入和输出通道数，`is_encoder` 表示该 `Block` 是否用于 `Encoder` 模块，默认为 `True`。
1.  `block()` 函数是用于构造卷积层、批归一化层和激活函数等操作的辅助函数，该函数的参数包括输入通道数 `dim_in` ，输出通道数 `dim_out` ，卷积核大小 `kernel_size`、步长 `stride` 和填充 `padding` 等。
1.  `Block` 类中的 s 对象是一个由多个 `block()` 函数构成的模型序列，用于构造多层卷积神经网络。该部分包括了多个卷积层、批归一化层和激活函数等操作，其中前四层卷积的输入通道数和输出通道数都为 `dim_in` ，后四层卷积的输入和输出通道数都为 `dim_out` ，只有再第五层卷积的步长为 `2` ，用于进行下采样操作。
1.  `Block` 类中的 `res` 对象是一个卷积层，用于实现残差连接操作。
1.  `Block` 类中的 `forward()` 函数实现了前向传播过程。在该函数中，首先使用 `s` 操作对输入数据进行多层卷积操作得到 `s` ，然后将输入数据通过残差连接 `res` 操作得到 `res`，最后将 `s` 和 `res` 相加作为该 `Block` 的输出。

```
import torch
class Block(torch.nn.Module):
    def __init__(self, dim_in, dim_out, is_encoder=True):
        super().__init__()
        cnn_type = torch.nn.Conv2d
        if not is_encoder:
            cnn_type = torch.nn.ConvTranspose2d
        def block(dim_in, dim_out, kernel_size=3, stride=1, padding=1):
            return (
                cnn_type(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding),
                torch.nn.BatchNorm2d(dim_out),
                torch.nn.LeakyReLU(),
            )
        self.s = torch.nn.Sequential(
            *block(dim_in, dim_in),
            *block(dim_in, dim_in),
            *block(dim_in, dim_in),
            *block(dim_in, dim_in),
            *block(dim_in, dim_out, kernel_size=3, stride=2, padding=0),
            *block(dim_out, dim_out),
            *block(dim_out, dim_out),
            *block(dim_out, dim_out),
            *block(dim_out, dim_out),
        )
        self.res = cnn_type(dim_in, dim_out, kernel_size=3, stride=2, padding=0)
    def forward(self, x):
        s = self.s(x)
        res = self.res(x)
        return s + res
```
这里定义了一个自编码器模型，其中 `encoder` 将输入的 `(B, 3, 64, 64)` 图像编码成长度为 128 的向量，也就是得到的结果大小为  `(B, 128)`，`decoder` 将长度为 128 的向量解码成原始的 `(B, 3, 64, 64)` 图像。

具体来说，`encoder` 包括四个 `Block` 层，每个 `Block` 层，最后将编码后的特征压平并通过一个全连接层将其转换为长度为 128 的向量。

`decoder` 则将长度为 128 的向量通过一系列的全连接层、多个 Block 层、上采样、卷积等操作，输出一个 `(B, 3, 64, 64)` 大小的图像。

最后的 `get_encoder_and_decoder` 函数将 `encoder` 和 `decoder` 一起返回。


```
encoder = torch.nn.Sequential(
    Block(3, 32, True),
    Block(32, 64, True),
    Block(64, 128, True),
    Block(128, 256, True),
    torch.nn.Flatten(),
    torch.nn.Linear(2304, 128),
)
decoder = torch.nn.Sequential(
    torch.nn.Linear(128, 256 * 4 * 4),
    torch.nn.InstanceNorm1d(256 * 4 * 4),
    torch.nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
    Block(256, 128, False),
    Block(128, 64, False),
    Block(64, 32, False),
    Block(32, 3, False),
    torch.nn.UpsamplingNearest2d(size=64),
    torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
    torch.nn.Tanh(),
)
def get_encoder_and_decoder():
    return encoder, decoder
```

### 训练过程
这里首先导入了一些需要使用的库和函数，接着调用`get_encoder_and_decoder()`函数得到编码器和解码器模型，然后通过`getLoader()`函数获取训练数据的加载器。最后定义了一个`Model`类，继承自`PreTrainedModel`，并在类中将编码器和解码器模型赋值给`self.encoder`和`self.decoder`属性。

```
from tqdm import tqdm
from data import getLoader, show
from model import get_encoder_and_decoder
import torch
from transformers import PreTrainedModel, PretrainedConfig
encoder, decoder = get_encoder_and_decoder()
loader = getLoader()
epoch = 2000
class Model(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self, config):
        super().__init__(config)
        self.encoder = encoder
        self.decoder = decoder
```
这里定义了一个函数 `train()` ，用于训练一个 AE 模型。在函数内部，首先定义了一个 `Adam` 优化器，使用 `MSE` 作为损失函数，并将模型移动到 `GPU` 上进行训练。接着对模型进行训练，循环遍历数据集中的每一个 `batch` ，将 `batch` 传递给编码器，得到编码器的输出，并将其作为输入传递给解码器。解码器会尽力将编码器输出解码为原始的图像，并计算解码图像与原始图像之间的损失。接下来使用反向传播算法计算梯度并更新参数。在训练过程中，使用了学习率调度器 `scheduler` 对学习率进行动态调整，以提高模型的训练效果。每 100 个 `epoch` 保存一次模型，并将训练好的最新模型保存到 `huggingface hub` 账号里。这里我把 `hub` 的 `token` 放到了文件 `tokens.txt` 中，大家可以改成自己的 `token` 进行模型的上传。

```
def train():
    optimizer = torch.optim.Adam(decoder.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=epoch * len(loader))
    criterion = torch.nn.MSELoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()
    loss = 0
    for e in range(epoch):
        print("epoch:", e)
        for data in tqdm(loader):
            data = data.to(device)   # [64, 3, 64, 64]
            pred = decoder(encoder(data))  # [64, 3, 64, 64]
            loss = criterion(pred, data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        if e % 100 == 0:
            print("epoch:", e, "loss:", loss.item(), "lr:",optimizer.param_groups[0]['lr'])
            with torch.no_grad():
                gen = decoder(torch.randn(4, 128, device=device))
            torch.save(gen, "data_%d.txt" % e)  
            show(gen)
            torch.save(Model(PretrainedConfig()), "ae_%d.pt" % e)   
            Model(PretrainedConfig()).push_to_hub( repo_id='wangdayaya/my_auto_encoder', use_auth_token=open('tokens.txt').read().strip())
train()
```

### 推理
因为我已经将模型保存到了 hub 上面，所以直接调用进行推理即可

```
import torch
from transformers import PreTrainedModel, PretrainedConfig
from ae.data import show
from ae.model import get_encoder_and_decoder
encoder, decoder = get_encoder_and_decoder()
class Model(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self, config):
        super().__init__(config)
        self.encoder = encoder.to('cpu')
        self.decoder = decoder.to('cpu')
decoder = Model.from_pretrained('wangdayaya/my_auto_encoder').decoder
with torch.no_grad():
    gen = decoder(torch.randn(4, 128))
show(gen)
```
我们看到画出了 4 个头像，但是比较模糊。但是考虑到我们的模型很简单，能进行特征的压缩和图像的还原已经很不错了。

![image_1684137865.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/08e9ab82cea549e995edf54f983dca4f~tplv-k3u1fbpfcp-watermark.image?)


### 优缺点


`AutoEncoder` 模型的优点包括：

1.  无需标注数据，可以使用非常大的无标签数据集进行训练。
1.  可以用于特征提取、数据降维等任务。在无监督场景下，可以使用 `AutoEncoder` 从原始数据中学习一组特征，并将这些特征用于其他下游任务，如分类或聚类。
1.  可以用于数据去噪，因为 `AutoEncoder` 可以学习如何从受损的数据中还原原始数据。
1.  `AutoEncoder` 可以自适应地学习数据分布，从而能够生成具有相似特征的新数据。

`AutoEncoder` 模型的缺点包括：

1.  `AutoEncoder` 容易受到过度拟合的影响，因此需要谨慎选择模型参数和正则化方法。
1.  `AutoEncoder` 通常需要大量的计算资源和时间才能训练出准确的模型。
1.  `AutoEncoder` 学习到的特征可能会难以解释，因此在某些任务上，如分类等，可能不如手工设计的特征效果好。
1.  `AutoEncoder` 在数据分布较为复杂的情况下可能无法捕捉到所有的数据特征，从而导致生成的数据出现缺陷。

### 感谢

- https://huggingface.co/datasets/lansinuote/gen.1.celeba
- https://github.com/lansinuote/Simple_Generative_in_PyTorch