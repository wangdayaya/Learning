如果我的文章对你有帮助，就请投我一票吧！


<p align=center><img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/48b098a174a24d468e77cdd5cf546ac5~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=751&h=1199&s=101001&e=jpg&b=bce4f8" alt="海报.jpg" width="50%" /></p>

# 前言

传统的三维场景渲染通常依赖于手工设计的三维模型和复杂的渲染方程。相比之下，`NeRF` 使用神经网络，通过学习从场景中的点到`颜色和密度的映射`，直接从数据中学习如何渲染，并且效果惊人，NeRF 技术主要应用于`计算机图形学、虚拟现实、增强现实`等领域。


# 基本概念介绍
**定义**：NeRF（神经辐射场）是一种计算机视觉技术模型，用于生成逼真三维场景。NeRF 的研究目的是合成同一场景不同视角下的图像。方法很简单，根据给定一个场景的若干张图片，重构出这个场景的 3D 表示，然后推理的时候输入不同视角就可以合成（渲染）这个视角下的图像了。

**输入：** NeRF 模型的输入是一组射线（rays）或光线，通常是从摄像机位置穿过图像像素的光线。每条射线都有起始点（摄像机位置）和方向。这些射线的集合构成了模型的输入。

**输出：** 模型的输出是在每条射线上各个点的颜色和密度。对于每个射线，模型输出的是沿着射线的各个位置的颜色和密度值。这些输出可以用来生成图像，即为每个像素着色，也可以用来还原三维场景的结构。

**实现过程：** NeRF 通过训练一个深度神经网络，该网络以射线作为输入，输出每个点的颜色和密度。训练数据通常是由带有深度信息的图像或者带有光照信息的图像生成的。

**效果呈现：** NeRF 可以生成`高质量的、逼真的`三维场景重建结果，包括了光照效果。由于其对场景的高度表达能力，可以捕捉复杂的几何结构和光照情况。

**效果好坏：** NeRF 的效果`通常非常好`，特别是在对场景的高保真度要求较高的情况下。然而由于模型的复杂性，训练和推理的`计算成本相对较高`。

# 模型生成挖掘机


我们使用 [`NeRF测试数据`](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz)来展示本次的任务，压缩包里面是 `106 张`不同角度拍摄的挖掘机玩具照片，另外还包含图像对应的`相机姿势`和`焦距`等相机参数。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e02d56e7d7644ae382e0ae9189ca1a32~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=640&h=480&s=22757&e=png&b=fdfbfb)

我们通过运行 tensorflow 实现的 NeRF 模型，通过不断训练，学习如何进行三维渲染，我们展示了某个相机角度的训练过程动图如下，可以看出重建效果不断清晰，说明 NeRF 产生了效果。

![training.gif](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/abb933398866470cafb4255a590d977a~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=2000&h=500&s=8501300&e=gif&f=94&b=ffffff)

我大约训练了 `100 个 epoch` ，然后经过处理可以看到渲染后的 `360 度场景视图`，效果虽然有点糊，主要是因为模型简单和数据不够精细的原因，但是还是能展现出这个模型的威力。可以看出该模型仅用了 100 个 epoch 就通过稀疏图像数据集成功学习了整个体积空间并进行了展示。

<p align=center><img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/50ed1bce705248fd94e4a92b8fc0b12a~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=100&h=100&s=690155&e=gif&f=115&b=040404" alt="rgb_video.gif"  width="50%"/></p>

# 工具生成狐狸

instant-ngp 是 GitHub 上面的一个开源项目，可以轻松实现三维模型的生成，只需要上传一个目标实体的 360 度视频或者是图片集合集合，渲染出的三维效果惊人，并且可以把结果保存为 obj 模型。

根据下图中官网说明，看自己显卡是哪个系列的，我的 window 主机是 4090 所以直接下载第一个压缩包。

-   [**RTX 3000 & 4000 series, RTX A4000–A6000**, and other Ampere & Ada cards](https://github.com/NVlabs/instant-ngp/releases/download/continuous/Instant-NGP-for-RTX-3000-and-4000.zip)
-   [**RTX 2000 series, Titan RTX, Quadro RTX 4000–8000**, and other Turing cards](https://github.com/NVlabs/instant-ngp/releases/download/continuous/Instant-NGP-for-RTX-2000.zip)
-   [**GTX 1000 series, Titan Xp, Quadro P1000–P6000**, and other Pascal cards](https://github.com/NVlabs/instant-ngp/releases/download/continuous/Instant-NGP-for-GTX-1000.zip)

解压之后直接双击 instant-ngp.exe 即可自动启动，然后将项目中自带的 data\nerf\fox 文件夹直接拖入到工具页面中，就会自动进行三维渲染。


![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/af0ce1960ff543de9f7e359d5254bde8~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1394&h=913&s=2493433&e=png&b=836a48)



等到训练结束，我们就能查看三维效果，下面展示的就是奇迹的时刻，渲染出来的狐狸相当逼真。

 
<p align=center><img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/533cf46abcfb423c9c95199b603925cb~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=338&h=338&s=1973864&e=gif&f=50&b=7a6148" alt="fox.gif"  /></p>

我们还能将三维模型保存成 obj 模型，可以在许多三维建模软件中进一步进行编辑修改。


<p align=center><img src="https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ae709df79f0649159c18b05a3690e054~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=484&h=709&s=266645&e=png&b=454545" alt="image.png"  /></p>

感兴趣的同学可以自己试试，如果有视频也可以，只不过需要将视频帧切成图片集，原理本质上都是一样的。

# 参考

- [NeRF 原论文](https://arxiv.org/abs/2003.08934)
- [NeRF 介绍博文](https://zhuanlan.zhihu.com/p/569843149)
- [tensorflow 实现 NeRF](https://keras.io/examples/vision/nerf/) 
- [NeRF 测试数据](http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz)
- [instant-ngp 仓库](https://github.com/NVlabs/instant-ngp?tab=readme-ov-file)