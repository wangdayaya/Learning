
# 介绍

Stability AI 刚发布 Stable Diffusion 3 模型进行公测。该模型采用 diffusion transformer 架构，显著提高了在多主题提示、图像质量和拼写能力方面的性能。



# 特点

## spelling abilities

就是可以将提示词中所需要绘制的文本展现在图片上。如下案例：

`Prompt: cinematic photo of a red apple on a table in a classroom, on the blackboard are the words "go big or go home" written in chalk`

`提示词：教室桌子上红苹果的照片，黑板上用粉笔写着“go big or go home”`

可以看出提示词中的 `go big or go home` 完整的展示在黑板上面，又准又狠！

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a1261f6f94ae4d1fb3d2ae5941211aa4~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=680&h=389&s=373899&e=png&b=c1c691)
##  multi-subject prompts
就是可以将用户提示词中的所提到的所有要素都展现出来。如下案例：

`prompt：Resting on the kitchen table is an embroidered cloth with the text 'good night' and an embroidered baby tiger. Next to the cloth there is a lit candle. The lighting is dim and dramatic`

`提示词：厨房的桌子上放着一块绣花布，上面写着“晚安”和一只绣着的小老虎。 布旁边有一支点燃的蜡烛。 灯光昏暗而戏剧性`

可以从图片中看出，桌子上的绣花布，晚安的字样，绣出来的小老虎，布旁边的点燃的蜡烛，昏暗的等黄，这些要素都完整的体现了出来！Perfect！
![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f2662cb6f0e54835bd49fa2cc16b57cb~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=680&h=389&s=552639&e=png&b=2c1506)
## image quality

把图像质量又提升了一个台阶，从此高清写真不在话下！

`Prompt: studio photograph closeup of a chameleon over a black background`

`提示词：黑色背景中变色龙的工作室照片特写`
![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a6b3e00910ad4b179e4c78681ab126e6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=680&h=389&s=552885&e=png&b=030303)

# 原理概要

`Stable Diffusion 3 `模型的参数范围在在 `800M 到 8B` 之间，整个模型的实现结合了 `diffusion transformer` 架构和 `flow matching` 机制。该模型的技术报告目前还未公布，不过不难推测，主要的模型还是 `DiT` 和 `flow matching 机制`。Stable Diffusion 3 模型致力于打造安全、负责任的目标，并且防止滥用，从开始训练模型时，持续到测试、评估和部署的整个过程，都增加了很多安全措施。

`DiT` 的主要贡献在于扩散模型可以成功地用 transformer `替换 U-Net` 主干，它不仅继承了 Transformer 模型类的优秀扩展特性，性能还优于先前使用 U-Net 的模型。Paper 入口请看文末参考部分。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4b5e668434cd42b9996c03cb33e814ce~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=679&h=554&s=882865&e=png&b=f3eae7)

`flow matching` 机制提出了基于连续归一化流（CNFs）的`生成模型新范式`，以及 flow matching 的概念，这是一种基于回归固定条件概率路径的矢量场的免模拟 CNFs 的方法。结果发现使用带有扩散路径的 flow matching ，可以使得训练出来的模型`更稳定`。Paper 入口请看文末参考部分。

![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dacab22eaa304329b413a8bb415ecde6~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=662&h=497&s=163053&e=png&b=fffdfd)



# 与 DALLE-3、MJ6 效果对比

`Prompt: a painting of an astronaut riding a pig wearing a tutu holding a pink umbrella, on the ground next to the pig is a robin bird wearing a top hat, in the corner are the words "stable diffusion"`

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/5434282821bb482f81d608b07800ed69~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=680&h=389&s=548436&e=png&b=57f6bc)

使用 `DALLE-3` 输入相同的 Prompt 生成的图像，虽然在关键的图像内容都生成了，但是可以看出在文本生成方面略输一筹，需要显示的“`stable diffusion`”拼写发生了错误。
![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/44d111fd118344d490d0d0e018c0d9b0~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=680&h=680&s=1232284&e=png&b=087090)

使用 ` midjourneyv6`   输入相同的 Prompt 生成的图像，可以看出和 `DALLE-3` 有同样的问题，可以看出 ` stable diffusion 3`  完胜。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d1edf9c594c943469cb5b2fc201e3a66~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1456&h=816&s=3061943&e=png&b=8f806f)

 




# 公测入口

- https://stability.ai/stablediffusion3

# 参考
- [Scalable Diffusion Models with Transformers](https://arxiv.org/pdf/2212.09748.pdf)
- [FLOW MATCHING FOR GENERATIVE MODELING](https://arxiv.org/pdf/2210.02747.pdf)
- https://blog.csdn.net/weixin_44839084/article/details/128634432
- https://www.qbitai.com/2024/02/122881.html
