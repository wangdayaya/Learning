 

  
 
 

https://zhuanlan.zhihu.com/p/660662864

[huggingface.co](https://huggingface.co/spaces/WildVision/vision-arena)

https://arxiv.org/pdf/2308.12966

2023年10月13日

## 贡献

1.  这是一套大型视觉语言模型(LVLMs)，旨在感知和理解文本和图像，可以实现图像理解、定位、文本阅读等。
1.  包括Qwen-VL和Qwen-VL-Chat两个模型：

-   Qwen-VL是一个预训练模型，能够感知和理解视觉输入，根据给定的提示生成所需的响应，并完成各种视觉语言任务，如图像描述、问题回答、基于文本的问题回答和视觉定位。
-   Qwen-VL-Chat则是基于Qwen-VL的指令调优视觉语言聊天机器人。Qwen-VL-Chat 能够与用户交互，并根据用户的意图感知输入图像。

3.  引入一种新的位置感知视觉-语言适配器，包括语言对齐的视觉编码器和位置感知适配器，来增强LLM的视觉能力。

3.  整个模型架构以及输入输出接口都非常简洁，详细设计了一个三阶段的训练管道，以在大量的图像-文本语料库上优化整个模型。

3.  Qwen-VL系列模型的主要特点包括：

    1.  领先的性能: 在多个评估基准（包括零样本图像描述、视觉问答、文档视觉问答和定位）上，它明显优于现有的开源大型视觉-语言模型（LVLMs）。
    1.  多语言: Qwen-VLs 支持英语、中文和中英文语言指令。
    1.  多图像:在训练阶段，我们允许任意交错的图像-文本数据作为Qwen-VL的输入。这个功能使Qwen-Chat-VL能够在给出多个图像时进行比较、理解和分析上下文。
    1.  精细视觉理解：由于我们在训练中使用了更高分辨率的输入大小和精细语料库，Qwen-VLs表现出高度竞争的精细视觉理解能力。与现有的视觉语言模型相比，我们的Qwen-VLs具有更好的细节文本识别、文档问答和边界框检测和精细对话性能。

<p align=center><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e327d6c984ee45d3978bb9fbde89765e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=uBwcdrAShRdgW5PIhMy07DjORIU%3D" alt=""  /></p>

## 模型架构

1.  Qwen-VL的整体网络架构由三个组件组成:

<p align=center><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d290dd6cc621441ebd91c0f5d6057f9f~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=hZz63ugqymRUHybdWaIpNJgKktk%3D" alt=""  /></p>

-   大型语言模型:Qwen-VL采用大型语言模型作为其基础组件。该模型由Qwen-7B预训练的权值初始化。
-   视觉编码器:Qwen-VL的视觉编码器使用了VisionTransformer(ViT) 架构，使用 Openclip 的 ViT-bigG （14*14 大小的 patch） 中预先训练的权重初始化。在训练和推理期间，输入图像被调整到特定的分辨率。视觉编码器通过将图像分成14*14的 patch 处理图像，生成一组图像特征。
-   2D位置感知视觉-语言适配器:为了缓解长图像特征序列带来的效率问题，使用视觉-语言适配器用于压缩图像特征。该适配器包含一个单层交叉注意力模块，初始化为随机值。该模块使用一组可训练的向量作为 query，并使用来自视觉编码器的图像特征作为交叉注意力操作的 key 。该机制将视觉特征序列压缩到固定长度为256。此外，考虑到为精细图像理解提供位置信息的重要性，2D绝对位置编码被整合到交叉注意力机制的 query-key 对中，以减轻压缩过程中位置细节的潜在损失。随后，长度为256的压缩图像特征序列被馈送到大型语言模型中。

## 输入和输出

1.  图像输入:图像通过视觉编码器和适配器进行处理，产生固定长度的图像特征序列。为了区分图像特征输入和文本特征输入，分别在图像特征序列的开头和结尾添加两个特殊标记(<img>和</img>)，表示图像内容的开始和结束。
1.  边界盒输入和输出:为了增强模型的细粒度视觉理解能力，Qwen-VL的训练涉及以区域描述、问题和检测形式的数据。与涉及图像-文本描述或问题的传统任务不同，该任务需要模型准确理解并生成指定格式的区域描述。对于任何给定的限界框，应用一个规范化过程(在[0,1000)范围内)并转换为指定的字符串格式:"(X_topleft,Y_topleft),(X_bottomright,Y_bottomright)".字符串标记为文本，不需要额外的位置词汇表。为了区分检测字符串和常规文本字符串，在限界框字符串的开头和结尾添加了两个特殊的标记(<box>和</box>)。此外，为了恰当地将限界框与其对应的描述性单词或句子相关联，引入了另一组特殊标记(<ref>和</ref>)，标记限界框所引用的内容。
1.  下面是 7 种任务的数据处理展示，黑色文本是前缀输入，蓝色文本是真实标签。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/cc72f6a516e7430e870522a56531f497~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=TqUWtWsXANY0Icun%2BwX36RtM9FA%3D)

## 训练

1.  一共分三个阶段，前两个是预训练，最后一个是 SFT 。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/cbb6bb7e40f941c59890a34270e6869e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=04P%2Fa1IWo2Arcrr4S9WiXCCAzck%3D)

2.  各个阶段的超参数介绍

<p align=center><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ca2a10aca8d04e2e9699ea32a3bee200~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551548&x-orig-sign=DZmyO6C53Ny1t%2B7%2BCbdJMPG%2FBJU%3D" alt=""  /></p>

### 第一阶段 Pre-training

1.  我们主要利用一个大规模、弱标签、网络爬取的图像-文本对数据集。我们的预训练数据集由几个公共可访问的来源和一些内部数据组成。原始数据集包含总共50亿对图像-文本，经过清理后，剩下14亿数据，其中77.3%(1092.4M)为英文(文本)数据，22.7%(325M)为中文(文本)数据。
1.  我们冻结大型语言模型，仅优化该阶段的视觉编码器和视觉适配器。输入图像被调整为 224x224 。训练目标是最小化文本标记的交叉熵。具体来说，模型的任务是尽量减小文本标记的预测与真实标记之间的差距，以提高文本生成的准确性。消耗了约15亿个图像-文本样本，预计 5000 亿个图像-文本 token。

<p align=center><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/fa5e6d2fb42b4f77b3a6ccfd85124b67~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=NBm5jBwHCK3LZotQH5s7T5VGEZY%3D" alt=""  /></p>

### 第二阶段 Mutil-task Pre-training

1.  引入了高质量和精细的视觉语言标注数据，具有更大的输入分辨率和交错的图像-文本数据。我们在7个任务上同时训练Qwen-VL，如下图所示。分别有： 图像描述（Captioning）、视觉问答（VQA）、定位任务（Grounding）、参考定位（Ref Grounding ）、定位描述（Grounded Cap.）、光学字符识别（OCR）、文本生成（Text Generation）。
1.  为了改进面向文本的任务，从Common Crawl中收集PDF和HTML格式的文本数据，并使用自然风景背景生成英语和中文的合成OCR数据。最后，我们简单地通过将相同任务数据打包成长度为2048的序列来构建交错的图像-文本数据。
1.  我们将视觉编码器的输入分辨率从224x224增加到448x448，从而减少了图像下采样引起的信息损失。我们进一步解锁了大型语言模型，并训练了整个模型。训练目标与预训练阶段相同。
1.  总计数据量76.8M。

<p align=center><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/cb95695b0b27467ea23bdcb610ea40d1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551548&x-orig-sign=U0FDm0ILc9T8xOjllrGDNt0p2Is%3D" alt=""  /></p>

### 第三阶段 SFT

1.  在这个阶段，我们通过指令微调对Qwen-VL预训练模型进行了微调，以增强其指令遵循和对话能力从而产生了交互式Qwen-VL-Chat模型。

1.  多模态指令微调数据主要来自字幕数据或通过LLM自我指令生成的对话数据，这些数据通常仅针对单图像对话和推理，并且仅限于图像内容理解。我们通过人工注释、模型生成和策略拼接构建了一套额外的对话数据，将定位能力和多图像理解能力融入Qwen-VL模型。

1.  我们在训练过程中将多模态和纯文本对话数据混合在一起，以确保模型在对话能力上的通用性。指令调优数据量为350k。在这个阶段，我们冻结了视觉编码器，并优化了语言模型和适配器模块。下面演示了这一阶段的数据格式。

    1.  ![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c39779d50c2d44fe998ee34bd4f510e8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=pse%2F23w8bL966h%2BFHrca5ofq98w%3D)

1.  为了更好地适应多图像对话和多图像输入，在不同图像前添加了字符串“Picture id:”，该字符串对应于图象输入对话的顺序。

1.  在对话格式方面，我们使用ChatML(Openai)格式构建了我们的指令调优数据集，其中每个交互的语句被标记为两个特殊的标记(<im_start>和<im_end>)，以便于对话的终止。

1.  在训练过程中，只对答案和特殊标记（上例中为蓝色）进行监督训练，而不监督角色名称或问题提示。

## 数据处理细节

论文中关于每种数据的处理细节做了介绍。

## 一些结论

1.  视觉语言适配器的输出特征越少，初始损失越低，但是过多或者变少都会导致收敛速度变慢。考虑到第二阶段的训练使用了 448*448分辨率，其中ViT输出的序列长度为(448/14)^2=1024。过少的查询会导致更多信息丢失。我们最终选择在Qwen-VL中为视觉语言适配器使用256作为特征长度。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0fccfeb8cf3042e1846a9c18b52e3748~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=aLPpVzd4elCRfmpytMkBs3a3E34%3D)

2.  在模型中使用高分辨率的Vision Transformer将显著增加计算成本。进行了消融实验，比较了使用全局注意力和窗口注意力进行ViT时的模型性能，以分析计算效率和模型收敛之间的权衡。 如图所示，当使用 Window Attention ，模型的损失会显著更高。而且他们俩的训练速度相似。因此，我们决定在训练Qwen-VL时使用 448x448 Global Attention 。我们之所以不使用896x896分辨率的 Window Attention，是因为它的训练速度对我们来说太慢了，训练时间几乎是448x448 Global Attention 输入模型的2.5倍。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4c69b97af04a478e8266bc4c4d35e76c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=UiG5ZRi4tZAGQtebJomldqlSyU4%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e491d4cecf7c48ebbeab6d663348acc8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=SanR9H8LiryYByTms0KDwYouZiI%3D)

3.  为了研究多模式训练对纯文本能力的影响，在下图展示了Qwen-VL与开源 LLM 的纯文本任务表现。测试时候Qwen-VL使用Qwen-7B的中间检查点而不是最终版本作为LLM初始化。此外，在多任务训练和SFT阶段，Qwen-V不仅利用视觉和语言相关数据，还纳入纯文本数据进行训练。这样做的目的是防止灾难性崩溃。结果表明 Qwen-VL模型在纯文本能力方面没有表现出任何退化，甚至在多任务训练后显示出改进。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d6141c38392242e6b65fe8a854e7a1e4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=n%2FhCzQk8XrWrmvByQy9tih65cl4%3D)
