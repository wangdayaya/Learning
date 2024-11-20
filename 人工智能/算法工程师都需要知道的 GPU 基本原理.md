# 为什么要使用 GPU

在一台主机上面通常会有 CPU 和 GPU 两种硬件设备，CPU 强调灵活性和单任务性能。 GPU 强调并行性和高吞吐性能 。两者各有所长，实际应用中经常共同使用， CPU 控制任务，GPU 执行数据密集型计算。

有一个很形象的例子，包工队里面有十个包工头和一万个工人，包工头就是 CPU 负责整体的工程进展和力量调度，工人就是 GPU 就负责简单的搬砖任务即可，哪里需要就去哪里搬。包工头肯定也能搬砖，但是他的更出色的能力是管理、控制和调度，工人只需要执行简单的海量任务即可。

下表是两者之间的主要区别：

| **对比维度**  | **CPU（中央处理器）**                 | **GPU（图形处理器）**               |
| --------- | ------------------------------ | ---------------------------- |
| **核心数量**  | 核心数量较少，通常为 4-24核，支持多线程，但并行能力有限 | 核心数量非常多，上万个处理单元，适合并行计算       |
| **任务类型**  | 适合处理复杂、逻辑密集型的任务（如系统调度、算法控制）    | 适合处理简单、数据并行的海量任务（如矩阵运算、图像处理） |
| **适用场景**  | 操作系统、日常办公、多任务调度                | 深度学习、图像处理、科学计算、高性能计算         |
| **延迟和吞吐** | 低延迟，高响应，快速完成单个任务               | 高吞吐量，适合大批量简单计算任务             |

在深度学习、科学计算、图形计算等领域有海量的并行计算任务，所以目前执行这些任务都需要用到 GPU 。

# GPU 相关的两个重要推动产业

游戏：游戏画面需要进行实时渲染，有海量计算任务需要执行

深度学习：模型内部需要大量的矩阵需要并行运算

# GPU 原理

## 算力计算单位

显卡算力单位常用的有 TFLOPS（每秒万亿次浮点运算）和 GFLOPS（每秒十亿次浮点运算）。

## 主流显卡介绍

这里有个概念需要解释一下，我们嘴里常说的 GPU ，严格来说只是显卡中的最核心的芯片，显卡一般还包括风扇、主板、电容等其他元器件，它是一个完整的产品形态。我们在具体的语境下要注意概念转换。

下面是 A100 的主要参数，我们可以看到对于对 FP64 Tensor Core 能达到 19.5TFLOPS，它的计算公式如下，先知道如何计算，等学习了后面的知识再回过来很容易搞懂了。

```
频率 * SM 数量 * 每个 SM 的 FP64 Tensor Cores 数量 * 2 (后面乘 2 是因为乘加视作两次浮点运算)
=1.41 GHz * 108 * 64 * 2
=19491.84 GFLOPS
=19.5 TFLOPS
```

<p align=center><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b3f33a1ed8da442eb047d79a32075dc1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732180431&x-orig-sign=xN%2BRyaXs5xiFz7qTByJRUYQbq2c%3D" alt=""  /></p>

## GPU 架构

如何计算海量简单的计算任务呢？简单的回答就是通过并行 ，通过大量的核心去并行计算。从下表中我们可以看出来 GPU 的核心和线程很多，它的架构就是专门针对执行海量计算任务而设计的。

在它的内部使用叫 SIMT 技术来管理这么多线程的执行。SlMT (single instruction multiple threads，单一指令多线程执行) 规定32个线程一组，叫做一个Warp 。Warp是GPU里调度任务的最小单元。

| 参数                            | CPU(Intel 酷睿 i9 14900K参数) | GPU（A100）        |
| ----------------------------- | ------------------------- | ---------------- |
| 核心数量                          | 24个                       | 6912 个 FP32 Core |
| 3456 个 FP64 Core              |                           |                  |
| 6912 个 Mixed INT32/FP32 cores |                           |                  |
| 线程数量                          | 32                        | 13824            |

下面是[ A100 显卡的架构图](https://hl8xut0wpb.feishu.cn/docx/B4oXdRbq7o8p59xEIwycNsOVnlJ#share-SWzYddDbSo99AcxyhElcO1GTnhd)，从下图我们可以看出来，中间占据了几乎大部分的就是 GA100 芯片，总共有 128 个 SM 。A100 基于 GA100 ，但是稍微有点不同，只有具有 108 个 SM。中间还有所有 SM 共享的 40MB 的 L2 Cache 。图片两边边缘的部分才是我们常说的芯片外的显存，也就是 HBM ，常见的有 40G 和 80G 两种规格。上边缘是 PCIE 接口负责和主板通讯，下边缘是 NVLink 接口负责多卡通讯。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/9887e4ecc6dd48b2b43c4694e6485590~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732180432&x-orig-sign=60nD2Ua3Ope1%2BK4ErUPO%2B2Yv39s%3D)

不同位置的带宽和计算强度如下图所示。可以看到 L1 传输速度最快，PCIe 传输速度慢的离谱，所以英伟达才会做出来多卡之间相互访问的 NVLink 加速传输数据，这对分布式训练很有用。基本上我们最理想状态就是从 L1 缓存中读取数据，充分利用 GPU 进行计算。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b57c73bfdd53450986f666b2d53678a4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732180431&x-orig-sign=q%2F5KB3d%2FbJCw08cOxW%2B9ql%2B9Lmc%3D)

## SM 结构

SM 就是 SM Streaming Multiprocessor，又叫流式多处理器，从下面的结构图中我们可以总结出来一些关键要素：

1.  每个 SM 包含 4 个区块， 它们共用 L1 Instruction Cache（一级指令缓存）、192KB L1 Data Cache（一级数据缓存）、Tex（纹理缓存，Texture cache）
1.  每个 SM 包含 4 个 L0 Instruction Cache
1.  每个 SM 总共有 4 个 Warp Scheduler ，负责调度 Warp ，确保 GPU 资源得到充分利用
1.  每个 SM 总共有 4 个 Dispatch Unit ， 负责接受 Warp Scheduler 准备好的 Warp 指令分配给执行单元
1.  每个 SM 中包含 4 个 16384 x 32bit 寄存器，也就是每个 SM 中包含寄存器文件大小为 256 KB
1.  每个 SM 总共有 64 个 INT32 Core
1.  每个 SM 总共有 64 个 FP32 Core
1.  每个 SM 总共有 32 个 FP64 Core
1.  每个 SM 总共有4 个 Tensor Core ，专为深度学习优化的核心，可以用于矩阵运算和混合精度计算
1.  每个 SM 中包含 16 个 LD/ST ，又叫 Load/store Units ，负责数据的加载和存储。
1.  每个 SM 中包含 4 个 SFU ，又叫 Special Function Units，负责处理特殊数学函数和复杂的计算操作，例如三角函数(sin、cos)、指数函数(exp)、对数函数(log)以及平方根(sqrt)等

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/80f715dacce448279988d99db6b5bd19~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732180431&x-orig-sign=euCBvy4zj%2FbWTkEVgswPRzwHGNk%3D)

## GPU 计算强度

  


一般情况下，在执行简单的图形渲染或者深度学习任务的时候，显卡不总是完全跑满，而是在等待数据，等待数据从显存加载到计算单元。我们看下图中的计算速度与内存带宽的比值为计算强度（ Compute Intensity ），说人话就是计算处理数据的速度与传输数据的速度的比值。

一般认为，当任务的计算强度高于此值的时候，也就是程序被认为是计算受限（compute-bound）；如果低于此值，则为内存受限（memory-bound） 。如下图所示 A100 的计算强度为 100 （19500 * 8 /1555 =100） ，也就是说任务的计算强度低于这个数 GPU 就会空闲，不过好在现在的大模型需要的计算量很大，一般只会不够用，而不是跑不满。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/01be26da032d47f6a563f1e86dc7f36e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732180432&x-orig-sign=V7sous0D28lybLxZqCje0c%2FraNs%3D)

  


假设对于两个 N*N 矩阵进行矩阵乘法运算，结果矩阵里的每个元素，都会经历 N 次乘和N次加，共2N次计算，最后总共计算结束计算量为 2N^3 ，传输数据也就是两个矩阵本身，就是 2N^2 ，计算量是传输量的 N 倍。这样我们就可以根据计算强度来调整 N 来充分利用GPU的计算能力。当数据传输较慢的时候我们可以传输较大的矩阵，数据传输较快的时候我们可以传输较小的矩阵。

  


## Tensor Core

当前显卡相较于以前最大的改动就是加入了 Tensor Core ，它是专为深度学习设计的张量核心，它的设计初衷就是为了解决深度学习中大量矩阵的乘法和加法计算，目的就是用于矩阵运算和混合精度计算。可以在一个时钟周期内完成一个矩阵乘法和一个矩阵加法，如下图所示，计算乘法的两个矩阵 A 和 B 要求是 FP16 ，而加法的两个矩阵 C 和 D可以是 FP16 ，也可以是 FP32 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d7e63da6517643a3a556ce7b371f2cad~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732180432&x-orig-sign=pB3iocpcVKG49Z4aW0XkXGhIlFQ%3D)

具体混合精度内部的计算原理如图所示，两个 FP16 的输入进行全精度乘法，然后使用 FP32 累加器与一个 FP32 输入进行累加计算，最后输出一个 FP32 结果。使用 FP32 累加计算应该是为了保证精度不损失。

  


![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/dfb30bbfa21f4cfa95d4891b3f80490a~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1732180431&x-orig-sign=5tmKOL4VKsQzeyBS%2F5U0hIjd6JY%3D)