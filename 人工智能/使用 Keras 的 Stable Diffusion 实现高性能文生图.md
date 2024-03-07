# 前言
在本文中，我们将使用基于 KerasCV 实现的 [Stable Diffusion](https://github.com/CompVis/stable-diffusion) 模型进行图像生成，这是由 `stable.ai` 开发的文本生成图像的多模态模型。

`Stable Diffusion` 是一种功能强大的开源的文本到图像生成模型。虽然市场上存在多种开源实现可以让用户根据文本提示轻松创建图像，但 KerasCV 有一些独特的优势来加速图片生成，其中包括 `XLA 编译`和`混合精度支持`等特性。所以本文除了介绍如何使用 KerasCV 内置的 `StableDiffusion` 模块来生成图像，另外我们还通过对比展示了使用 KerasCV 特性所带来的图片加速优势。
# 准备

- N 卡，建议 `24 G` ，在下文使用 KerasCV 实际生成图像过程中至少需要 `20 G`
- 安装 `python 3.10` 的 anaconda 虚拟环境
- 安装 `tensorflow gpu 2.10 `
- 一颗充满想象力的大脑，主要是用来构建自己的文本 prompt 

这里有一个工具函数 `plot_images` ，主要是用来把模型生成的图像进行展示。
```
def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()
```
# 模型工作原理

`超分辨率工作`可以训练深度学习模型来对输入图像进行去噪，从而将其转换为更高分辨率的效果。为了实现这一目的，深度学习模型并不是通过恢复低分辨率输入图像中丢失的信息做到的，而是模型使用其训练数据分布来填充最有可能的给定输入的视觉细节。

然后将`这个想法推向极限`，在纯噪声上运行这样的模型，然后使用该模型不断去噪最终产生一个全新的图像。这就是潜在扩散模型的关键思想，在 2020 年的 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 中提出。

![flowers.gif](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e7628fb913c14826815cf2cbb7ed318f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=768&h=384&s=10096006&e=gif&f=81&b=d31dc7)

现在要从潜在扩散过渡到文本生成图像的效果，需要添加`关键字控制生成图像的能力`，简单来说就是将一段文本的向量加入到到带噪图片中，然后在数据集上训练模型即可得到我们想要的`文生图模型 Stable Diffusion` 。这就产生了 Stable Diffusion 架构，主要由三部分组成：

- `text encoder`：可将用户的提示转换为向量。
- `diffusion model`：反复对 64x64 潜在图像进行去噪。
- `decoder`：将最终生成的 64x64 潜在图像转换为更高分辨率的 512x512 图像。

基本模型架构图如下：


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4e2838282aa643c58eee345202a0de88~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1590&h=792&s=432983&e=png&b=fcfbfb)

#  benchmark
我们使用 `keras_cv` 中的` StableDiffusion` 模块构造一个文生图基准模型 model ，在对模型进行基准测试之前，先执行一次 ` text_to_image` 函数来预热模型，以确保 `TensorFlow graph`已被跟踪，这样在后续使用模型进行推理时候的速度测试才是准确的。可以从日志中看到第一次运行的时间是 22 s ，这个不用去管他，我们只看第二个时间。

我这里的提示词是`“There is a pink BMW Mini at the exhibition where the lights focus”` ，生成 `3` 张图像，耗时 `10.32 s` 。

执行结束之后运行 keras.backend.clear_session() 清除刚刚运行的模型，以保证不会影响到后面的试验。
```
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=False)
model.text_to_image("warming up the model", batch_size=3)
start = time.time()
images = model.text_to_image("There is a pink BMW Mini at the exhibition where the lights focus", batch_size=3)
print(f"Standard model: {(time.time() - start):.2f} seconds")
plot_images(images)
keras.backend.clear_session()
```
日志打印：
```
25/25 [==============================] - 22s 399ms/step
25/25 [==============================] - 10s 400ms/step
Standard model: 10.32 seconds
```
![319f63da759ac3c6d2b850d9465fef9.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/7197d10b96cb476086df7bec375d5a69~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=628&h=192&s=266901&e=png&b=f7eeed)



# benchmark + Mixed precision

正如日志中打印的信息可以看到，我们这里构建的模型现在使用`混合精度计算`，利用 `float16` 运算的速度进行计算，同时以 `float32` 精度存储变量，这是因为 `NVIDIA GPU` 内核处理同样的操作，使用 float16 比 float32 要快得多。


我们这里和上面一样先将模型`预热加载`，然后针对我的提示词`“There is a black BMW Mini at the exhibition where the lights focus”`生成了 `3` 张图像，耗时 `5.30s` ，可以看到在 benchmark 基础上使用混合精度生成速度提升将近一倍。
```
keras.mixed_precision.set_global_policy("mixed_float16")
model = keras_cv.models.StableDiffusion(jit_compile=False)
print("Compute dtype:", model.diffusion_model.compute_dtype)
print("Variable dtype:",  model.diffusion_model.variable_dtype)
model.text_to_image("warming up the model", batch_size=3)
start = time.time()
images = model.text_to_image( "There is a black BMW Mini at the exhibition where the lights focus", batch_size=3,)
print(f"Mixed precision model: {(time.time() - start):.2f} seconds")
plot_images(images)
keras.backend.clear_session()
```

日志打印：
```
Compute dtype: float16
Variable dtype: float32
25/25 [==============================] - 9s 205ms/step
25/25 [==============================] - 5s 202ms/step
Mixed precision model: 5.30 seconds
```
![179ce83c7bb1e25e5958d3c8a9dda51.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f4d95f6adef04a87921578f0eb9741bb~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=624&h=188&s=251287&e=png&b=f4f1ef)


# benchmark + XLA Compilation

`XLA（加速线性代数）`是一种用于机器学习的开源编译器。XLA 编译器从 PyTorch、TensorFlow 和 JAX 等常用框架中获取模型，并优化模型以在不同的硬件平台（包括 GPU、CPU 和机器学习加速器）上实现高性能执行。

TensorFlow 和 JAX 附带 XLA ， keras_cv.models.StableDiffusion 支持开箱即用的 `jit_compile` 参数。 将此参数设置为 True 可启用 XLA 编译，从而显著提高速度。

从日志中可以看到，在 benchmark 基础上使用 XLA 生成时间减少了 `3.34 s` 。

```
keras.mixed_precision.set_global_policy("float32")
model = keras_cv.models.StableDiffusion(jit_compile=True)
model.text_to_image("warming up the model", batch_size=3)
start = time.time()
images = model.text_to_image("There is a black ford mustang at the exhibition where the lights focus", batch_size=3, )
print(f"With XLA: {(time.time() - start):.2f} seconds")
plot_images(images)
keras.backend.clear_session()
```
日志打印：
```
25/25 [==============================] - 34s 271ms/step
25/25 [==============================] - 7s 271ms/step
With XLA: 6.98 seconds
```
![0fe51809c822d71ad91d8a770dc517f.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a73d9adbdd3846d9a56cc1095b497c1e~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=625&h=194&s=259035&e=png&b=e8dad3)


#  benchmark + Mixed precision + XLA Compilation 

最后我们在 benchmark 基础上同时使用混合精度计算和 XLA 编译，最终生成同样的 3 张图像，时间仅为 `3.96s` ，与 benchmark 相比生成时间减少了 `6.36 s` ，生成时间大幅缩短！

```
keras.mixed_precision.set_global_policy("mixed_float16")
model = keras_cv.models.StableDiffusion(jit_compile=True)
model.text_to_image("warming up the model", batch_size=3, )
start = time.time()
images = model.text_to_image( "There is a purple ford mustang at the exhibition where the lights focus", batch_size=3,)
print(f"XLA + mixed precision: {(time.time() - start):.2f} seconds")
plot_images(images)
keras.backend.clear_session()
```
日志打印：
```
25/25 [==============================] - 28s 144ms/step
25/25 [==============================] - 4s 152ms/step
XLA + mixed precision: 3.96 seconds
```
![630d45a4d883874517055b22ff61dce.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5c435922e58b45e9b88c707ca8c1731d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=627&h=194&s=277097&e=png&b=f9f3f1)


# 结论

四种情况的耗时对比结果，展示了使用 KerasCV 生成图片确实在速度方面有特别之处：
- benchmark : 10.32s
- benchmark + Mixed precision ：5.3 s
- benchmark + XLA Compilation : 6.98s
- benchmark + Mixed precision + XLA Compilation : 3.96s
