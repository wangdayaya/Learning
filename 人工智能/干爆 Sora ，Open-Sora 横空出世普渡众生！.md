# Sora 最新动态

Sora 最近发布了一条 [Blog](https://openai.com/blog/sora-first-impressions) ，里面介绍了最新的进展，团队声称已经从创意社区获得了很多宝贵的反馈建议，并且与视觉艺术家、设计师、创意总监和电影制作人等合作，将内测的一些创意视频公布了出来，大家可以欣赏一下。



<p align=center><img src="https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/852c2c5e55a941439443e8332907bb68~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=776&h=435&s=591250&e=png&b=726b69" alt="image.png"  /></p>


<p align=center><img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b3d1a110aae44daf9f3549831ff8d114~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=775&h=435&s=712921&e=png&b=38382c" alt="image.png"  /></p>


<p align=center><img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/6d9ffe4cff5c40b7af6787c0580907f9~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=773&h=433&s=755948&e=png&b=122326" alt="image.png"  /></p>

`但是仍然是只能看，不能用！`

#  Open-Sora
Sora 是等不到了，但是难不住我们开源社区的同志，这不是，开源项目 [Open-Sora](https://github.com/hpcaitech/Open-Sora) 横空出世，Open-Sora 实现了先进文生视频技术，并且将`源代码、预训练模型、数据处理过程、安装和推理说明、训练过程`都进行了详细的介绍，可谓是功德无量。 `2024.03.18` 发布了` Open-Sora 1.0` ，支持 `2s` 的 `512x512` 分辨率的文生视频，目前该项目仍然是`早期阶段`，还在积极的开发过程中。Open-Sora 为扩散模型提供了高速训练框架，在 64 帧 512x512 视频上训练时，可以实现 55% 的训练速度加速，`并且该框架支持训练 1 分钟 1080p 视频`。

按照现在技术这么快的发展速度，我觉得如果 Sora 再继续狗着不开放，那应该最后免不了被 Open-Sora 挤掉，成为束之高阁得历史展品。下面是 Open-Sora 生成得夜景马路得短视频，我们先睹为快。


 <p align=center><img src="https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/15f625dd5c644af7b278476713b9a5fd~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=512&h=512&s=1197628&e=gif&f=20&b=1a1916" alt="open-sora.gif"  /></p> 

## 项目整体目录结构


```
Open-Sora
├── README.md
├── docs
│   ├── acceleration.md            -> Acceleration & Speed benchmark
│   ├── command.md                 -> Commands for training & inference
│   ├── datasets.md                -> Datasets used in this project
│   ├── structure.md               -> This file
│   └── report_v1.md               -> Report for Open-Sora v1
├── scripts
│   ├── train.py                   -> diffusion training script
│   └── inference.py               -> Report for Open-Sora v1
├── configs                        -> Configs for training & inference
├── opensora
│   ├── __init__.py
│   ├── registry.py                -> Registry helper
│   ├── acceleration               -> Acceleration related code
│   ├── dataset                    -> Dataset related code
│   ├── models
│   │   ├── layers                 -> Common layers
│   │   ├── vae                    -> VAE as image encoder
│   │   ├── text_encoder           -> Text encoder
│   │   │   ├── classes.py         -> Class id encoder (inference only)
│   │   │   ├── clip.py            -> CLIP encoder
│   │   │   └── t5.py              -> T5 encoder
│   │   ├── dit
│   │   ├── latte
│   │   ├── pixart
│   │   └── stdit                  -> Our STDiT related code
│   ├── schedulers                 -> Diffusion schedulers
│   │   ├── iddpm                  -> IDDPM for training and inference
│   │   └── dpms                   -> DPM-Solver for fast inference
│   └── utils
└── tools                          -> Tools for data processing and more
```

训练和推理的配置目录如下：

```
Open-Sora
└── configs                        -> Configs for training & inference
    ├── opensora                   -> STDiT related configs
    │   ├── inference
    │   │   ├── 16x256x256.py      -> Sample videos 16 frames 256x256
    │   │   ├── 16x512x512.py      -> Sample videos 16 frames 512x512
    │   │   └── 64x512x512.py      -> Sample videos 64 frames 512x512
    │   └── train
    │       ├── 16x256x256.py      -> Train on videos 16 frames 256x256
    │       ├── 16x256x256.py      -> Train on videos 16 frames 256x256
    │       └── 64x512x512.py      -> Train on videos 64 frames 512x512
    ├── dit                        -> DiT related configs
    │   ├── inference
    │   │   ├── 1x256x256-class.py -> Sample images with ckpts from DiT
    │   │   ├── 1x256x256.py       -> Sample images with clip condition
    │   │   └── 16x256x256.py      -> Sample videos
    │   └── train
    │       ├── 1x256x256.py       -> Train on images with clip condition
    │       └── 16x256x256.py      -> Train on videos
    ├── latte                      -> Latte related configs
    └── pixart                     -> PixArt related configs
```


## 模型

本项目还不忘 diss 了 openai 的“闭源”精神，为了让 AI 更加开放，Open-Sora 致力于打造 Sora 的开源版本，并在[报告](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_v1.md)中详细描述了首次尝试训练基于 Transformer 的视频扩散模型，主要内容如下：

- 使用 Stability-AI 的 `2D VAE` 。
- 视频训练涉及大量的 `token` ，考虑到 `24 帧`的 `1 分钟`视频共有 `1440 帧`。 通过 `VAE` 的 4 倍下采样和 `patch 大小`的 2 倍下采样 ，一共约有有 `1440x1024≈1.5M token` 。 计算 `150 万`个 token 的全注意力会导致巨大的计算成本，因此改用 `spatial-temporal attention` 来降低 [Latte](https://github.com/Vchitect/Latte) （Latent Diffusion Transformer for Video Generation）的成本。 
- 下图是经过测试的四种模型，出于效率考虑选择 STDiT（Sequential），具体的[报告](https://github.com/hpcaitech/Open-Sora/blob/main/docs/acceleration.md#efficient-stdit)可以看这里。

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/11f0a0ba6f2144d080a5e1eb057692c9~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=837&h=375&s=53286&e=png&b=ffebe7)
- 本项目基于有效训练的高质量文生图模型框架 `PixArt-α` 为底座，它具有 `T5 条件`的 `DiT 结构`。用 `PixArt-α` 初始化模型，并将插入的`时间注意力`的投影层初始化为零。这种初始化保留了模型在开始时生成图像的能力，与同样使用`空间-时间注意力机制的 Latte 模型`相比，`STDiT` 可以更好的利用已经预训练好的图像 DiT 的权重，从而在视频数据上继续训练。插入的 attention 使参数数量从 `580M` 增加到 `724M` 。具体来说整个架构包括一个`预训练好的 VAE`，一个`文本编码器`和一个利用`空间-时间注意力机制的 STDiT` (Spatial Temporal Diffusion Transformer) 模型。其中 STDiT 每层的结构如下图所示。它采用串行的方式在二维的空间注意力模块上叠加一维的时间注意力模块，用于建模时序关系。在时间注意力模块之后，交叉注意力模块用于对齐文本的语意。与全注意力机制相比，这样的结构大大降低了训练和推理开销。

<p align=center><img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dee40e3c37614ae38ddaf4de66a9d457~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=372&h=612&s=180760&e=png&b=f7f1f0" alt="image.png"  /></p>

- 借鉴 PixArt-α 和 Stable Video Diffusion 的成功经验，主要分为三个阶段，`第一阶段`是进行图像预训练，初始化一个图像生成模型。`第二阶段`是在已经构建的大规模视频数据集上进行视频预训练，学习运动表征。`第三阶段`是在一个小规模的高质量视频数据集上进行微调。本项目采用渐进式训练策略：在 `366K` 预训练数据集上训练 `16x256x256` 分辨率，然后在 `20K` 数据集上训练 `16x256x256 、16x512x512 和 64x512x512` 三种分辨率。 使用 `scaled position embedding` ，能够该策略大大降低了计算成本。
- 尝试在 DiT 中使用 3D 补丁嵌入器，目前在 `16 帧`训练中每 `3 帧`采样一次，在 `64 帧`训练中每 `2 帧`采样一次。
- 本项目开放出来 `3` 个预训练模型，分辨有 `512×512` 和 `256×256` ，其中标识 `HQ` 表示使用的是高质量数据。但是需要注意的是，这些模型是在有限的预算下进行训练的。 质量和文本对齐相对较差。 该模型表现不佳，尤其是在生成人的样子，并且无法遵循详细的指令。开发人员仍然在努力提高质量和文本对齐方式。

Resolution | Data   | iterations | Batch Size | GPU days (H800) | URL                                                                                       |
| ---------- | ------ | ----------- | ---------- | --------------- | ----------------------------------------------------------------------------------------- |
| 16×512×512 | 20K HQ | 20k         | 2×64       | 35              | [🔗](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth) |
| 16×256×256 | 20K HQ | 24k         | 8×64       | 45              | [🔗](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x256x256.pth) |
| 16×256×256 | 366K   | 80k         | 8×64       | 117             | [🔗](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-16x256x256.pth)




## 数据

高质量的数据是高质量模型的关键。本项目使用的数据集如下，不仅提供了代码，还提供了处理视频数据的思路。

#### 数据集准备

训练数据应以 CSV 文件形式提供，格式如下：

    /absolute/path/to/image1.jpg，标题 1，帧数
    /absolute/path/to/image2.jpg，标题2，帧数


本项目中用到的数据集有 [HD-VG-130M](https://github.com/daooshee/HD-VG-130M?tab=readme-ov-file)、[Inter4k](https://github.com/alexandrosstergiou/Inter4K)、[Pexels.com](https://www.pexels.com/)，后续还会视情况考虑使用以下数据集。

| Name              | Size         | Description                   |
|----------------- | ------------ | ----------------------------- |
| Panda-70M         | 70M videos   | High quality video-text pairs |
| WebVid-10M        | 10M videos   | Low quality                   |
| InternVid-10M-FLT | 10M videos   |                               |
| EGO4D             | 3670 hours   |                               |
| OpenDV-YouTube    | 1700 hours   |                               |
| VidProM           | 6.69M videos |                               |


#### 将视频分割成片段

来自互联网的原始视频对于训练来说还是太长了， 因此本项目给出脚本检测原始视频中的场景，并根据场景将它们分割成短片，可以安装下面的包来并运行 scene_detect.py 来进行高效处理，需要注意的是运行代码的时候要指定自己的数据集路径。

    pip install sceneDetect moviepy opencv-python 



#### 生成视频字幕 

人工为视频添加描述既昂贵又耗时，所以本项目采用强大的 `image captioning` 模型来生成视频字幕。虽然 `GPT-4V` 表现更好，但其 `20s/sample` 的速度太慢了，而是改为了用 `LLaVA` 实现 `3s/sample` 的速度，并且生成质量相当。`LLaVA` 是 [MMMU](https://mmmu-benchmark.github.io/) 中第二好的开源模型，可以接受任何分辨率。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b32376eb4df24aa3930790ad7128458f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1186&h=470&s=803876&e=png&b=fbf9f9)


如果使用[ GPT-4V](https://openai.com/research/gpt-4v-system-card) 模型，可以运行以下命令为使用 GPT-4V 的视频生成字幕，每个 3 帧的视频成本约为 0.01 美元，输出结果是带有路径和标题描述的 CSV 文件。

```
python -m tools.caption.caption_gpt4 FOLDER_WITH_VIDEOS output.csv --key $OPENAI_API_KEY
```


如果使用 LLaVA 模型，要先按照官方说明安装 LLaVA ，本项目使用的是 [liuhaotian/llava-v1.6-34b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) 模型来制作视频描述，然后运行以下命令可以获得带有路径和标题的 CSV 文件。

```
CUDA_VISIBLE_DEVICES=0,1 python -m tools.caption.caption_llava samples output.csv
```

## 其他

至于安装部署说明、训练指令说明、推理指令说明等内容请详见 [github](https://github.com/hpcaitech/Open-Sora/tree/main) 项目介绍。

## 效果展示
 
<p align=center><img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/29f6ed22dc094f79bfce2fa0111ef4e4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=512&h=512&s=2194651&e=gif&f=20&b=10badc" alt="乌龟.gif"  /></p>

<p align=center><img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7dc32f425e1b46dcb241fb41006cc514~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=512&h=512&s=1923940&e=gif&f=20&b=020202" alt="星空.gif"  /></p>

<p align=center><img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/08bb6fb1056744c98c0b6ecef732b1ff~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=512&h=512&s=2878277&e=gif&f=20&b=bfc7d7" alt="瀑布.gif"  /></p>

<p align=center><img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f788cb1745194e99898896ae06f7bce6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=512&h=512&s=2208780&e=gif&f=20&b=a9bcca" alt="海岸.gif"  /></p>


# 参考
- https://www.jiqizhixin.com/articles/2024-03-18-12

 