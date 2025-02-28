
https://arxiv.org/pdf/2409.12191  2024年10月3日



## 引言
![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4863159286ec40a18f45052d4116bc11~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=UGSaVbEsQK0KiUVuTVIUOJJ7EJk%3D)
### 背景

1.  当前的大型视觉语言模型(LVLMs)受到固定图像输入大小的限制，通常通过降采样或升采样图像或使用缩放然后填充的方法。这种一刀切的策略尽管方便用一致的分辨率处理图像，但它也限制了模型在不同尺度图片捕捉信息的能力，特别是在高分辨率图像中导致大量细节信息的丢失。因此，这些模型无法像人类视觉那样对尺度和细节敏感地感知视觉信息。
1.  当前大多数LVLMs依赖于冻结的CLIP风格的视觉编码器，无法对于复杂的推理任务和图像中细节的处理，实验发现通过微调视觉变换器(ViT)在LVLM训练过程中可以解决这些限制。

### 贡献

1.  Qwen2-VL系列包括三个开放权重模型分别为 2B、7B和72B，语言模型使用的是不同的 Qwen2 的 LLM ，但是视觉编码器都是使用同一款 675M 的ViT。另外并开放了代码。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4a4a2c4b179f43d1bdf27f5adcef0238~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=js1wCl9DXv9h67NoAD8T9HTwHw4%3D)

2.  为了探索大型多模态模型的潜力，Qwen2-VL研究了大型视觉语言模型(LVLMs)的缩放规律。
2.  采用了一个统一的范式处理图像和视频，提高了模型的视觉感知能力。
2.  该模型还集成了多模态旋转位置嵌入(M-ROPE)，促进了文本、图像和视频之间位置信息的有效融合。它使用独立的组件来表示时空信息。
2.  Qwen2-VL引入了Naive动态分辨率机制，使模型能够动态地将不同分辨率的图像处理成不同数量的视觉标记，在各种分辨率和长宽比上最先进的理解能力。
2.  Qwen2-VL能够理解长度超过20分钟的视频，支持基于视频的问题回答、对话、内容创作等任务的能力。
2.  Qwen2-VL可以与手机、机器人等边缘设备集成。
2.  除了英语和中文，Qwen2-VL现在还支持图像内的多语言上下文理解，包括大多数欧洲语言、日语、韩语、阿拉伯语、越南语和其他语言。
2.  Qwen2-VL支持的功能包括，多语言图像文本理解、数学推理、视频分析、实时聊天、agent、定位、OCR等。大部分任务效果超过现有模型。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1a46c0194d5c4328bc2d50d4dbabf08d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=jo8r8rmMdi4%2F6y%2Fk2wvElPiV8qE%3D)

## 架构

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/a4ca2599238a4832a1142aeda6ac3ffc~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=HLfrLXeCcHJ%2BcfhGzx8yxfpw4Wg%3D)

1.  一个视觉转换器(ViT)，具有大约6.75亿个参数，擅长处理图像和视频输入。
1.  在语言处理方面，选择了更强大的Qwen2系列语言模型。
1.  Qwen2-VL的一个关键架构改进是引入Naive Dynamic Resolution支持。与Qwen-VL不同，Qwen2-VL现在可以处理任何分辨率的图像，动态地将它们转换为可变数量的视觉标记。为了支持这一功能修改了ViT原始的绝对位置嵌入，并引入了2D-RoPE来捕捉图像的二维位置信息。此外，为了减少每个图像的视觉标记，在ViT之后使用一个简单的MLP层两层线性层，将相邻的2x2 token 压缩为一个token ，加上特殊的<|vision start|>和<|vision end|>标记放置在压缩视觉token 的开头和结尾。因此，使用 patch size=14 进行编码的分辨率 224x224 的图像在进入 LLM 之前将被压缩到 66 个标记。Qwen2-VL并没有采用当下流行的大图切分方式（比如LLava-Next ），而是直接对图像进行patch化，然后直接过image encoder进行特征提取，最后对齐到LLM之前，对视觉token数压缩与提取特征。
1.  另一个关键的架构增强是多模态旋转位置嵌入(Multimodal Rotary Position Embedding，M-ROPE)的创新。与LLMs中的传统1D-ROPE不同，后者仅限于编码一维位置信息，M-ROPE有效地建模了位置信息。多模态输入的信息是通过将原始旋转嵌入分解为三个部分(时间、高度和宽度)来实现的。对于文本输入，这些组件使用相同的位置ID，使M-RoPE在功能上与1D-RoPE等效。在处理图像时，每个视觉标记的时间ID保持不变，而高度和宽度组件根据标记在图像中的位置分配不同的ID。对于视频，作为帧序列处理，每个帧的时间ID递增，而高度和宽度组件遵循与图像相同的ID分配模式。在模型输入包含多种模态的情况下，每种模态的位置编号通过递增前一种模态的最大位置ID来初始化。M-ROPE如下图所示。M-ROPE不仅增强了位置信息的建模，还减少了图像和视频的位置ID值，使模型在推理过程中能够扩展到更长的序列。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/82f6404f047d44cca6a505060dcf8eea~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551548&x-orig-sign=cw7BPvO93Ym4%2BPyWDOavJ9%2FMqoo%3D)

5.  Qwen2-VL采用统一混合训练方案（Unified lmage and Video Understanding），结合图像和视频数据，确保在图像理解和视频理解方面的熟练度。为了尽可能完整地保留视频信息，我们对每个视频进行每秒两帧的采样。此外，我们结合了深度为2的3D卷积用于处理视频输入，使模型能够处理 3D tubes 而不是 2D patches ，从而能够在不增加序列长度的情况下处理更多的视频帧。为了保持和视频每秒的处理方式一致，每个图像都被视为两个相同的帧。为了平衡长视频处理的计算需求与整体训练效率，我们动态调整每个视频帧的分辨率，将每个视频的总token数限制为16384。

  


## 数据格式

1.  和 Qwen-VL 一样，Qwen2-VL也使用特殊标记来区分视觉和文本输入。标记<|vision_startl>和<|vision_end|>插入在图像特征序列的开头和结尾，以区分图像内容。
1.  对话数据。在对话格式方面，我们使用ChatML格式构建我们的指令调整数据集，其中每个交互的语句都标记了两个特殊标记(<|im_start|>和<|im_end|>)，以方便对话终止。蓝色标记的部分是监督部分。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7256d42a57084f2fb889c0cb72afebf5~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=o27ejcC0uuIK63L8zyBnsK%2B%2Brso%3D)

3.  视觉定位。为了赋予模型可视化的定位能力，限界框坐标在[0,1000)内被归一化，并表示为字符串 “(X_topleft,Y_topleft),(X_bottomright, Y_bottomright) "。<|box_start|>和<|box_end|> 两个 token 用于限定限界框文本。为了准确地将限界框与文本描述链接起来，我们引入了标记 <|object_ref_start|> 和<|object_ref_end|> 来表示限界框引用的内容，从而允许模型有效地解释和生成特定区域的精确描述。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/caff295c426b4002a601009b83811154~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551548&x-orig-sign=57aoPBxPmu%2F%2F2tBxLFUEJQ%2F4D6s%3D)

4.  视觉agent。为了将Qwen2-VL开发为通用的VL-Agent，我们将各种代理任务视为顺序决策问题，使Qwen2-VL能够通过多步骤动作执行来完成任务。对于每个任务，我们首先为函数调用定义一组允许的操作和关键字模式（下图中带下划线的内容）。Qwen2-VL然后分析观测结果，进行推理和规划，执行选定的操作，并与环境交互以获取新的观测结果。这个循环会反复迭代，直到任务成功完成。通过整合各种工具并利用大型视觉语言模型(LVLMS)的视觉感知能力，Qwen2-VL能够反复执行涉及现实世界视觉交互的越来越复杂的任务。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/84e67ad03d1e40109e5e49a92f4b81f4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551548&x-orig-sign=1uuAa08MzO5jtHuGMshzc1DqqUE%3D)

## 训练

1.  沿用 Qwen-VL 中采用了三阶段训练方法，前两个是预训练阶段，后面是 SFT
1.  数据集包括图像-文字对，光学字符识别(OCR)资料，交错的图像-文字的文章，视觉问答数据集，视频对话，和图像知识数据集。我们的数据源主要包括清理过的网页、开源数据集和合成数据。我们的数据知识截止日期为2023年6月，这种多样化的数据组成有助于发展强大的多模态理解能力。
1.  在第一阶段，我们只专注于训练 ViT 部分（和线性层），利用大量的图像-文本对语料库来对齐大型语言模型(LLM)中的语义理解能力。在初始预训练阶段，Qwen2-VL使用了约6000亿个token的语料库。Qwen2-VL的LLM组件使用来自Qwen2 的参数初始化，而 Qwen2-VL 的视觉编码器使用从 DFN 衍生的 ViT 初始化，然而原始DFN的ViT中的固定位置嵌入被 RoPE-2D 取代。这个预训练阶级主要专注于学习图像-文本关系，通过 OCR 识别图像中的文本内容，以及图像分类任务。这种基础训练对于使模型能够对核心的视觉-文本相关性和对齐形成强大的理解至关重要。
1.  在第二阶段，我们训练所有参数，并用更广泛的数据进行训练，以便更全面地学习，第二个预训练阶段标志着一个显著的进步，涉及额外的8000亿个与图像相关的token。这个阶段引入了更多量的混合图像-文本内容，有助于更细致地理解视觉和文本信息之间的相互作用。视觉问题回答数据集的整合提高了模型回答与图像相关问题的能力。此外多任务数据集的纳入对于发展模型同时处理多种任务的能力至关重要。同时，纯文本数据在保持和提高模型的语言能力方面继续发挥着关键作用。
1.  在两个预训练阶段中，Qwen2-VL总共处理了1.4万亿个标记。具体来说，这些标记不仅包括文本标记，还包括图像标记。然而，在训练过程中，我们只对文本标记提供监督。这种对广泛和多样化的语言和视觉场景的接触确保了模型对视觉和文本信息之间复杂关系的深入理解，从而为各种多模态任务奠定了坚实的基础。
1.  最后阶段，我们冻结ViT参数，并使用指令数据集对LLM进行微调。在指令微调阶段，我们采用ChatML格式来构造指令跟踪数据。该数据集不仅包括纯文本对话数据，还包括多模态会话数据。多态组件包括图像问题回答、文档解析、多图像比较、视频理解、视频流对话和基于代理的交互。我们全面的数据构建方法旨在增强模型理解和执行各种不同模式的指令的能力。通过融合不同的数据类型，我们寻求开发一个更通用和稳健的语言模型，能够处理复杂的多模态任务，除传统的基于文本的交互外。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/fdf21c823b104581a5c2c148cd8f4784~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=ehiog7evNNaOJ8zZSECmRWwyh54%3D)

## 基础设施

1.  Qwen2-VL模型在阿里云上进行了训练。
1.  存储。我们使用阿里云的超高速CPFS构建Owen2-VL预训练和后期训练的存储系统。我们将文本数据和视觉数据存储解耦。我们直接在CPFS上存储文本数据，并使用mmap进行高效访问。我们使用阿里云的OSS对象存储服务进行视觉数据持久存储。在训练过程中，我们通过OSS的python-client并发访问视觉数据，并调整并发和重试参数以避免达到OPS 限制。我们还发现视频数据的解码是一个主要瓶颈，尤其是对于长视频。在尝试了开源和内部软件后，由于多次失败，我们选择了缓存解码技术。检查点将每个GPU的优化器和模型状态保存到CPFS上。
1.  并行性。我们使用3D并行性，结合数据并行性(DP)、张量并行性(TP)和管道并行性(PP)来扩展Qwen2-VL模型的训练。我们还利用deepspeed的zero-1冗余优化器来分片状态以节省内存。序列并行性(SP)和选择性检查点激活相结合以减少内存使用。在启用 TP 训练时，我们总是将视觉编码器和大语言模型一起分割，但不分割vision merger，因为它的参数相对较少。我们发现TP训练会导致不同的模型共享权重，因为卷积操作的非确定性行为。我们通过离线减少共享权重来解决这个问题，从而避免了额外的 all-reduce 通信步骤。这种方法仅对性能产生最小影响。我们利用1F1BPP 进行Qwen2-VL72B训练。我们将视觉编码器、视觉适配器和几个LLM的解码器层组合成一个阶段，并将剩余的解码器层平均分配。请注意，对于每个数据点，视觉和文本序列长度是动态的。在启动 1F1B 过程之前，我们广播动态序列长度，并使用批索引访问形状信息。我们还实现了交错的 1F1BPP ，但发现它比标准的1F1B设置慢。
1.  软件。我们使用 PyTorch 2.1.2 与 CUDA11.8 用于训练。此外，我们利用 flash-attention用于视觉编码器和LLM的高效训练。我们还使用融合运算符，如LayerNorm 、RMSNomm 和Adam 。除此之外，我们在训练过程中的矩阵乘法中使用了通信和计算重叠技术。

## 实验

1.  Qwen2-VL在相同规模下表现出极具竞争力的性能，大部分任务取得了最新最先进的(SOTA)成果。 值得注意的是，它在文档理解任务中表现出显著的优势。然而，在MMMU等基准测试中，我们的模型在GPT-4o的某些方面仍然落后，这表明Qwen2-VL-72B在处理更复杂和具有挑战性的问题集时仍有改进空间。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5e1619693f494119bc89dd37f0ffaa67~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551548&x-orig-sign=BfTxBIQiKn%2BuMvPt8Wc4iTBWNvA%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/adebbd9ede9c4a69b5870d91255ddd63~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=uPPqLv5RGyeYBJuzWORRJj9DzPo%3D)

2.  如表7所示，我们比较了动态分辨率和固定分辨率的性能。动态分辨率方法更有效率，动态分辨率方法始终能实现顶级性能，同时平均消耗的特征token更少。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/322deec9c1334e0cbafb04baf1a11146~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=Hiyn8CDaO1XM7oUjKx3b9ta8aqs%3D)

3.  仅仅增加图像大小并不总是能带来性能的提升。选择适合不同图像的分辨率更为重要。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6ab36249e88b439c8740fd100b70d0b1~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551548&x-orig-sign=YkYRzui%2F%2FDdEZEHc2WU5HWWiLoo%3D)

4.  我们展示了M-ROPE的有效性。首先，我们验证其在各种下游任务上的能力。我们使用Qwen2-1.5B和ViT-L作为骨干并报告预训练模型的结果。与1D-RoPE相比，使用M-ROPE在下游图像和视频各个任务中，特别是在视频基准上，取得了更好的性能。此外，我们评估了M-RoPE在Video-MME中长视频上的长度外推能力。下图还显示了Qwen2-VL-72B在不同推理长度下的性能。利用M-ROPE在各种视频推理长度上表现出稳健的结果。值得注意的是，尽管在训练过程中将每个视频的最大token数量限制为16K，但该模型在最大推理长度为80K 时仍表现出卓越的性能。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8cea119489414f3e98315cf9ca0173c4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=Z%2F0Z2eyxMDBFL9kdtSc834wkwrQ%3D)

5.  我们评估了不同规模的模型在多个能力维度上的表现。如下图左边所示，随着模型大小的增加，性能不断改善，特别是在数学能力方面，数学能力与模型参数的数量呈正相关。如下图右边所示，我们可视化Qwen2-VL-7B在预训练第二阶段中模型性能与训练token数量之间的关系。随着训练token数量的增加，模型性能提高;然而，在视觉问答(VQA)任务上的表现出现了一些波动。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b770e194dfd649429342e0a1ca344549~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740551547&x-orig-sign=kMFRe3JUNBwRBm6sAzwTcvVzHTI%3D)