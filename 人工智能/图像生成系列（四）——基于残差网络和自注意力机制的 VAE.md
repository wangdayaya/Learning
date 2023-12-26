# 准备
- python=3.8
- torch=1.13.1
- 显存最低 14G

# 数据

这里是加载数据集并创建一个适用于训练的数据加载器，方便后续的模型训练过程中使用。

*   加载了一个名为`lansinuote/diffsion_from_scratch`的数据集，里面就是皮卡丘系列的图像数据和对应的文本描述，还定义了一个数据预处理的组合`compose`，通过一系列的图像预处理操作对图像进行处理，主要包括图像尺寸的调整、图像尺寸的裁剪、图像值归一化等常规操作。

*   定义了一个函数`f`，用于对数据中图像和文本进行进一步的处理。将图像通过预处理组合处理为（`pixel_values`），将文本通过`tokenizer`进行批量编码处理为（`input_ids`），使用`dataset.map`函数将数据集使用函数`f`进行批量处理。

*   然后，定义了一个`collate_fn`函数，该函数将 `pixel_values`和 `input_ids` 分别提取出来，将它们转换为张量，并将它们放置在设备上（`device`）。 使用`torch.utils.data.DataLoader`创建数据加载器对象，传入数据集，返回创建的数据加载器对象。

```

def getDataLoader(device, tokenizer):
    dataset = load_dataset(path='lansinuote/diffsion_from_scratch', split='train')
    compose = torchvision.transforms.Compose([
        torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(512),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])

    def f(data):
        pixel_values = [compose(i) for i in data['image']]   # [b,3,512,512]
        input_ids = tokenizer.batch_encode_plus(data['text'], padding='max_length', truncation=True, max_length=77).input_ids  # [b, 77]
        return {'pixel_values': pixel_values, 'input_ids': input_ids}

    dataset = dataset.map(f, batched=True, batch_size=100, remove_columns=['image', 'text'])
    dataset.set_format(type='torch')

    def collate_fn(data):
        pixel_values = [i['pixel_values'] for i in data]
        input_ids = [i['input_ids'] for i in data]
        pixel_values = torch.stack(pixel_values).to(device)
        input_ids = torch.stack(input_ids).to(device)
        return {'pixel_values': pixel_values, 'input_ids': input_ids}

    loader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=1)
    return loader
```

# 工具

这里定义了一个名为`show`的函数，用于显示图像。输入就算是多张图像，也只显示第一张图像，这个函数还有一个作用就是在训练模型的时候每隔一段时间展示一下模型的图像生成效果，暂停显示 2秒，并且将图像保存下来。

```

def show(images):
    if type(images) == torch.Tensor:
        images = images.to('cpu').detach().numpy()
    images = images[:1]
    plt.figure(figsize=(3, 3))
    for i in range(len(images)):
        image = images[i]
        image = image.transpose(1, 2, 0)
        plt.subplot(1, 1, i + 1)
        plt.imshow(image)
        plt.axis('off')
    plt.pause(2)
    plt.savefig(f"data/diffusion_result_{time.time()}.png")
    plt.close()
```

# 模型

## Resnet

这里定义了一个`Resnet`的类，它实现的是残差网络模块。实现比较简单，就是经过 `GroupNorm` 归一化操作、`SiLU` 激活函数、`Conv2d` 卷积操作等实现输入的的残差网络连接，这里需要注意的是在进行残差连接的时候，可能会将特征图的通道数从 `dim_in` 变为 `dim_out` ，但是特征图大小不变。

在`变分自编码器（Variational Autoencoder，VAE）`中使用残差网络有以下几个可能的用途：

*   模型复杂性增加：残差网络可以引入更多的非线性变换，增加模型的表示能力和学习能力。在 `VAE` 中使用残差网络可以增加模型的表达能力，使其能够更好地捕捉输入数据的复杂特征。
*   模型深度扩展：`VAE`的编码器和解码器通常是由多个层组成的神经网络。使用残差网络可以帮助解决梯度消失和梯度爆炸等问题，使得模型可以更容易地训练和优化。残差连接可以将梯度直接传递到浅层网络，使得信息的传递更加高效。
*   特征重用：在 `VAE` 中，编码器和解码器可以被视为特征提取器和特征生成器。使用残差网络可以促进特征的重用，即编码器提取的特征可以直接传递给解码器进行重建，从而提高重建质量和模型的效率。
*   高维数据处理：对于高维数据，如图像数据，在编码和解码过程中使用残差网络可以帮助减少参数数量，提高计算效率，并且可以更好地保留和重建图像的细节和纹理特征。

总之，使用残差网络在 `VAE` 中可以增加模型的表达能力，改善训练效果，并提高重建质量，尤其对于复杂的高维数据处理任务非常有用。

    class Resnet(torch.nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.s = torch.nn.Sequential(
                torch.nn.GroupNorm(num_groups=32, num_channels=dim_in, eps=1e-6, affine=True),
                torch.nn.SiLU(),
                torch.nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
                torch.nn.GroupNorm(num_groups=32, num_channels=dim_out,  eps=1e-6, affine=True),
                torch.nn.SiLU(),
                torch.nn.Conv2d(dim_out,  dim_out,  kernel_size=3, stride=1, padding=1),
            )
            self.res = None
            if dim_in != dim_out:
                self.res = torch.nn.Conv2d(dim_in,  dim_out,  kernel_size=1, stride=1, padding=0)

        def forward(self, x):   
            res = x
            if self.res:
                res = self.res(x)  
            r = self.s(x)  
            return res + r 

## Atten

这里定义了一个自注意力机制（`Self-Attention`），可以处理图像数据中的关联关系。

在这个模块中，是一个常规的自注意力机制的实现，首先对输入进行归一化处理，然后通过线性变换将输入映射为（`q`）、（`k`）和（`v`）的表示。然后对键（`k`）进行转置操作，以便进行点积计算。点积计算通过 `torch.bmm` 函数实现，得到注意力权重（`atten`）。注意力权重经过 `softmax` 操作得到最终的注意力分布，然后将注意力分布与值（`v`）相乘得到加权的注意力表示。最后通过线性变换将加权的注意力表示映射回原始的特征维度，并重新调整形状，最后加上残差连接（`residual connection`）。

在`VAE`（变分自编码器）中使用 `Attention` 机制可以带来以下几个意义或作用：

*   模型学习全局依赖关系：`VAE` 是一种生成模型，用于学习数据分布。在某些情况下，数据中存在全局的依赖关系，而传统的 `VAE` 模型可能难以捕捉到这种全局关系。引入 `Attention` 机制可以帮助模型学习数据中不同位置之间的依赖关系，提高模型对全局信息的建模能力。
*   提高重建质量：`VAE` 的目标之一是实现高质量的重建，即将输入数据重构为原始数据的近似。`Attention` 机制可以帮助模型更好地关注重要的特征区域，对输入数据进行更准确的重建。通过将注意力加权的特征用于解码过程，可以增强模型对重要特征的重建能力，提高重建质量。
*   提升生成样本的多样性：`VAE` 不仅可以进行数据重建，还可以生成新的样本。引入`Attention` 机制可以在生成过程中有选择性地关注不同部分的隐变量表示，从而在生成样本时增加多样性。通过对生成器的注意力分布进行控制，可以指导模型在生成过程中注重不同的特征或位置，从而生成具有更多样性的样本。

总之，将 `Attention` 机制引入到 `VAE` 中可以增强模型对全局依赖关系的学习能力，提高重建质量，并增加生成样本的多样性。这些效果有助于改进 `VAE` 模型的表现和生成能力，使其更好地适应不同的数据分布和应用场景。

```

class Atten(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=512, num_groups=32,  eps=1e-6, affine=True)

        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)
        self.out = torch.nn.Linear(512, 512)

    def forward(self, x):
        res = x   
        x = self.norm(x)
        x = x.flatten(start_dim=2).transpose(1, 2)   

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        k = k.transpose(1, 2)   # [1, 4096, 512] -> [1, 512, 4096]
        # 照理来说应该和 atten = q.bmm(k) * 0.044194173824159216 是等价的,但是却有很小的误差
        atten = torch.baddbmm(torch.empty(1, 4096, 4096, device=q.device),
                              q, k, beta=0,
                              alpha=0.044194173824159216)  # 0.044194173824159216 = 1 / 512**0.5
        atten = torch.softmax(atten, dim=2)
        atten = atten.bmm(v)   
        atten = self.out(atten)  
        atten = atten.transpose(1, 2).reshape(-1, 512, 64, 64)   
        atten = atten + res
        return atten
```

## Pad

这里定义了一个名为 Pad 的类，用于在输入张量上进行零填充操作，在特征图的右侧和下方加一排 0 。

    class Pad(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.pad(x, (0, 1, 0, 1), mode='constant', value=0)

# VAE

这段代码定义了一个名为 `VAE（Variational Autoencoder）`的模型。模型结构分为两部分：`编码器（encoder）`和`解码器（decoder）`。

编码器部分：

*   编码器接受输入张量 `x`（尺寸为\[1, 3, 512, 512]）作为输入。
*   编码器通过一系列的卷积层和残差网络（`Resnet`）对输入进行特征提取和降维操作。
*   在中间层，引入了注意力机制（`Attention`）来对特征进行加权融合。
*   最后，编码器输出一个正态分布的潜在表示h（尺寸为\[1, 8, 64, 64]），其中前 4 个通道表示均值，后 4 个通道表示标准差。

解码器部分：

*   解码器接受潜在表示 `h` 作为输入。

*   解码器首先对潜在表示 `h` 进行采样，得到一个新的潜在表示 `h'`（尺寸为\[1, 4, 64, 64]）。

*   解码器通过一系列的卷积层和残差网络对潜在表示`h'`进行上采样和特征重建。

*   最后，解码器输出重建的图像（尺寸为\[1, 3, 512, 512]）。

需要注意的是，模型中的残差网络（`Resnet`）和注意力机制（`Attention`）模块可以帮助提取输入数据的重要特征，并提高模型的表达能力和生成效果。同时，模型中使用了标准正态分布来采样潜在表示，以便在训练过程中进行随机性的重构和生成。

```

class VAE(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            #in
            torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            #down
            Resnet(128, 128),
            Resnet(128, 128),
            torch.nn.Sequential( Pad(), torch.nn.Conv2d(128, 128, 3, stride=2, padding=0), ),
            Resnet(128, 256),
            Resnet(256, 256),
            torch.nn.Sequential(  Pad(),  torch.nn.Conv2d(256, 256, 3, stride=2, padding=0), ),
            Resnet(256, 512),
            Resnet(512, 512),
            torch.nn.Sequential(  Pad(), torch.nn.Conv2d(512, 512, 3, stride=2, padding=0), ),
            Resnet(512, 512),
            Resnet(512, 512),
            #mid
            Resnet(512, 512),
            Atten(),
            Resnet(512, 512),
            #out
            torch.nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Conv2d(512, 8, 3, padding=1),
            #正态分布层
            torch.nn.Conv2d(8, 8, 1),
        )

        self.decoder = torch.nn.Sequential(
            #正态分布层
            torch.nn.Conv2d(4, 4, 1),
            #in
            torch.nn.Conv2d(4, 512, kernel_size=3, stride=1, padding=1),
            #middle
            torch.nn.Sequential(Resnet(512, 512), Atten(), Resnet(512, 512)),
            #up
            Resnet(512, 512),
            Resnet(512, 512),
            Resnet(512, 512),
            torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Resnet(512, 512),
            Resnet(512, 512),
            Resnet(512, 512),
            torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            Resnet(512, 256),
            Resnet(256, 256),
            Resnet(256, 256),
            torch.nn.Upsample(scale_factor=2.0, mode='nearest'),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            Resnet(256, 128),
            Resnet(128, 128),
            Resnet(128, 128),
            #out
            torch.nn.GroupNorm(num_channels=128, num_groups=32, eps=1e-6),
            torch.nn.SiLU(),
            torch.nn.Conv2d(128, 3, 3, padding=1),
        )

    def sample(self, h):
        mean = h[:, :4]
        logvar = h[:, 4:]
        std = logvar.exp()**0.5
        h = torch.randn(mean.shape, device=mean.device)
        h = mean + std * h
        return h

    def forward(self, x):
        h = self.encoder(x)
        h = self.decoder(h)
        return h
```

# 训练

这实现了一个用于训练 `VAE` 模型的训练函数`train()`，是一个常规的模型训练过程。

其中的 tokenizer 是使用 `DiffusionPipeline` 库加载预训练的扩散模型和标记器（`tokenizer`）。该扩散模型用于生成文本描述，标记器用于对文本进行编码。然后，函数定义了训练过程中所需的一些变量，如损失函数（均方误差），优化器（AdamW）和训练数据加载器。在每个训练周期（epoch）中，对数据加载器中的每个批次进行迭代。在每个批次中，模型首先通过前向传播生成预测输出。然后，计算预测输出和原始输入之间的均方误差损失，并进行反向传播和参数更新。

在每个指定的训练步骤，会打印当前训练周期、训练步骤和累计的损失。同时，调用`show(pred)`函数展示生成的预测结果。最后，将训练后的模型保存到指定的路径。

```

def train():
    global pred
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = 'saved/vae.model'
    if os.path.exists(path):
        vae = torch.load(path)
    else:
        vae = VAE()
    vae.to(device)
    vae.train()
    diffusion = DiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    tokenizer = diffusion.tokenizer

    loss_sum = 0
    loader = getDataLoader(device, tokenizer)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    for epoch in tqdm(range(50)):
        for i, data in enumerate(loader):
            pred = vae(data['pixel_values'])
            loss = criterion(pred, data['pixel_values'])
            loss.backward()
            loss_sum += loss.item()
            if i % 200 == 1:
                optimizer.step()
                optimizer.zero_grad()
                print(epoch, i, loss_sum)
                loss_sum = 0
                show(pred)
        torch.save(vae.to(device), path)
```

# 效果
当损失值降到 9 以下，基本效果就不错了，这里是从一开始到最后训练生成的效果图，挑选了 16 张来展示模型的进化效果，可以看出来，将动漫角色还原的很好，颜色也比较到位。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/098e1773000349e08d3cd0291f9d6c7a~tplv-k3u1fbpfcp-watermark.image?)

# 致谢
 
*   <https://huggingface.co/datasets/lansinuote/diffsion_from_scratch>
*   <https://huggingface.co/CompVis/stable-diffusion-v1-4>
