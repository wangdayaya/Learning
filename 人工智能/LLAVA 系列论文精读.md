# LLAVA

https://blog.csdn.net/qq_27590277/article/details/136035115

https://zhuanlan.zhihu.com/p/624928279

https://arxiv.org/pdf/2304.08485

2023年12月11日

## 主要贡献

1.  GPT4生成多模态指令数据。本文提出了一个数据重组方式解决视觉与语言组成的指令数据匮乏问题，使用 ChatGPT/GPT-4 将图像-文本对转换为适当的指令格式；
1.  大型多模态模型。通过连接 CLIP 的开源视觉编码器和语言解码器 LLaMA 组成多模态模型 LLaVA，并在生成的视觉 - 语言指令数据上进行端到端微调。
1.  公布两个多模态基准测试。作者提出了 LLaVA-Bench 两个具有挑战性的基准，包括各种配对图像、指令和详细注释。
1.  公众发布了以下资产：GPT4生成的多模式指令数据、模型、代码。
1.  极大简化了VLM的训练方式：Pre-training + Instruction Tuning 。
1.  训练成本得到简化：1M量级数据+ 8卡A100 在一天可以完成训练。
1.  LLaVA它的网络结构简单、微调成本比较低，任何研究组、企业甚至个人都可以基于它构建自己的领域的多模态模型。

## GPT辅助生成的视觉指令数据

使用 COCO 图像并生成三种类型的指令数据。对于每种类型先人工标注一些数据，作为 In-context-learning 的 few-shot examples ，送给 GPT-4，共计 158K 。

利用两种格式的数据，用于提示 GPT-4 的上下文内容：

-   Captions
-   Bounding boxes

三类指令数据：

-   对话格式：58K 多轮对话，助手与人之间关于图像的具体、事实性问题和答案。
-   详细描述：23K 单轮对话，基于随机从提前准备的问题列表中抽取的问题，生成的全面、详尽的图像描述。
-   复杂推理：77K 单轮对话，需要严格逻辑和深入分析来回答的图像相关问题

用于从 ChatGPT/GPT-4 生成基于图像的对话的提示词。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6f914118b19242dcaf515bf4fb703ea3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724586&x-orig-sign=sAI3aOefPQ5NIMJgbQzYImuRkIg%3D)

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5badccd971b3451184232d953b932327~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=IQEssnNQ4cJqVtdFNWjr%2BOj5XqM%3D)

举例展示指令遵循的数据，不包括右上方的图片。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/2e63b4e611c342bbaa3cff57ad87a38d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=ATXnuH%2BVg8Rx5wWv1muJz0R5ERM%3D)

简短的图像描述指令，制作数据的时候会随机抽取一个当作指令。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0d0f88ff2d4b45ce900f338df8790a5b~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=qmMCy%2BifZqZimYHPbnGhBIAj1iw%3D)

详细的图像描述指令，制作数据的时候会随机抽取一个当作指令。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/aaafd080013b44ae830c78c826ce0818~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=M6tEHrIxg75%2F0LknDW76vMzYTHA%3D)

## 架构

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7910b9ba3aca483bac237aa6ec9b993c~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=82iJ%2F%2FB4ctBzvBJVSqO137OXdZI%3D)

1.  架构主要目标是有效利用预训练好的语言模型和图像模型进行视觉语言生成，完成通用视觉和语言理解。
1.  论文中选择了语言模型 Vicuna 13B，使用其他的如 LLAMA 也可以。对于输入图像 X_v，本文使用预训练的 CLIP 视觉编码器 ViT-L/14-224px（LLaVA 1.5用的是 CLIP ViT-L/14-336px ）进行处理，得到视觉特征 Z_v=g (X_v)。实验中使用的是最后一个 Transformer 层之前和之后的网格特征。本文使用一个简单的线性层来将图像特征连接到token嵌入空间中。具体而言，应用可训练投影矩阵 W 将 Z_v 转换为语言嵌入标记 H_v，H_v 具有与语言模型中的单词嵌入空间相同的维度：

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/32c1c299d2b848898c2741b4437352a0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=MO%2B6PQOD3auYcSH1038lhLIYHbI%3D)

3.  这种简单投影方案具有轻量级、成本低等特点，能够快速迭代以数据为中心的实验。也可以考虑连接图像和语言特征的更复杂（但昂贵）的方案，例如 Flamingo 中的门控交叉注意力机制和 BLIP-2 中的 Q-former 等。

## 训练

对于同一张图片的多轮问答训练数据组织形式如下图所示。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d99bcd74c86a4061af51a82105cf7120~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=lB2EZSTO9lkQVOmKizBo8DbAfwo%3D)

### 训练的两个阶段总结图

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/17bca452160f4b37bd76258ca02eeee9~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=s%2B6GK91dIAsLsE1MvtVC6oCluXo%3D)

在微调过程中，使用 FSDP（FullShard Data Parallel）和梯度检查点来节省 GPU 内存，并且不使用卸载。启用 BF16 和 TF32 以实现速度和精度之间的平衡。我们使用 8× A100 训练所有模型。在 CC-595K 上进行预训练在 4 小时内完成。在 Instruct-158K 上进行微调在 10 小时内完成。在 ScienceQA 上进行微调在 4 小时内完成。

### Stage 1: Pre-training for Feature Alignment

使用 CC3M 过滤出来的 595K 个图像文本对。使用上面描述的扩展方法将图像文本对转换为指令跟随数据。每个样本都可以被视为单轮对话。为了构造输入 X_instruct，对于图像 X_v，随机抽取一个问题 X_q，这是一个语言指令用于要求助手简要描述图像。真实预测答案 X_a 是原始标题。在训练中，我们保持视觉编码器和 LLM 权重冻结，并仅训练投影层 W 最大化预测真实答案的可能性。这样图像特征 Hv 可以与预训练的 LLM 词嵌入对齐。

### Stage 2: Fine-tuning End-to-End

保持视觉编码器权重冻结，并继续更新 LLaVA 中投影层和 LLM 的预训练权重。我们考虑两个特定的用例场景进行不同方式的微调：

• 多模态聊天机器人。我们通过对 158K 语言图像指令遵循数据进行微调来开发聊天机器人。在三种类型的响应中，对话是多轮的，而其他两种是单轮的。它们在训练中被统一采样。

• 科学问答。我们在 ScienceQA 基准上研究我们的方法，这是第一个大规模多模态科学问题数据集，它用详细的讲座和解释来注释答案。每个问题都用自然语言或图像的形式提供上下文。助手以自然语言提供推理过程，并在多个选项中选择答案。训练中我们将数据组织为单轮对话，问题和上下文为 X_instruct，推理和答案为 X_a。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ac25709ed7cd4cffb4e55e1e80fb4f16~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=LNrRECzQRspVFmI%2B1xYL%2B9TavGY%3D)

## 测试效果

1.  分别在上面提到的多模态聊天机器人和科学问答进行测试。
1.  多模态聊天机器人测试结果中，尽管 LLaVA 是用一个小的多模态指令数据集（约 80K 的不重复图像）训练的，但它在不是 LLaVA 的数据集范围内，展示了与多模态模型 GPT-4 非常相似的推理结果，LLaVA 能够理解场景并按照问题说明进行回答。指令回答能力出色，相比之下，BLIP-2 和 OpenFlamingo 专注于描述图像，而不是按照用户指令以适当的方式进行回答。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/42f437cfd8ec44f0a005af49453ee1eb~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724586&x-orig-sign=6qRi%2BjwfmGvS7cmRqnAp98OKEr0%3D)

3.  Flamingo 可以看作是多模态领域的 GPT-3 时刻，因为它在零样本任务迁移和上下文学习方面表现优异。但是在多模态任务中的表现通常与纯语言任务相比有所欠缺。
3.  科学问答测试结果中，其实没超过SOTA MM-CoT_Large，但是结合上GPT-4的ensembling 模型略微有所提升。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/bac2bf157ab04d5ba2dacd27362cafa5~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724587&x-orig-sign=zIZ2zGjcQ0F0nGbw5rD%2B3i17lWw%3D)

5.  文章同时做了一些ablation study，一些值得关注的结论：

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e26a4d104f134debb81ff2d5d5fe028d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724587&x-orig-sign=f5isvRYPSmdfmot32EKkS7CtaL0%3D)

-   **视觉特征提取**：使用ViT倒数第二层的Features更有利，这可能因为 CLIP 的最后一层特征可能比它之前的层更关注全局和抽象的图像属性，而它之前的层可以更关注有助于理解特定图像细节的局部属性
-   **思维链CoT**：发现“先生成reason再生成answer” 相比“先生成answer再生成reason” 仅对模型快速收敛有帮助，对最终的性能提升没帮助。
-   **Pre-train**：证明了pre-train的有效性，pre-train+scienceQA finetune 相对比直接在ScienceQA train from scratch 会提升5.11%，因为在保留大量预训练知识的同时对齐多模态特征。
-   LLM**模型大小**：7B比13B的低1.08%，印证了越大的LLM对整体的性能越有利。
-   **涌现能力**：LLaVA 的一个有趣的涌现行为是它能够理解训练中未涵盖的视觉内容。此外LLaVA 还展示了令人印象深刻的 OCR（光学字符识别）能力。

# LLAVA 1.5

https://arxiv.org/pdf/2310.03744

2024年5月15日

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3ef5798961ea41e89f4226904f993bc9~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=hP8c7AcXChov6wKyjoGb6Bc7GrE%3D)

  


1.  LLaVA-1.5在11个基准测试中达到了SoTA水平，仅通过对原始LLaVA进行简单修改，利用约120万的数据，超越了使用十亿规模数据的方法。
1.  使用 CLIP-ViT-L-336px 视觉编码器替换原先的 CLIP-ViT-L/14。
1.  将原先的一层线性层替换为两层线性层
1.  LLM 换成了更大的 13B 。
1.  使用类似“残差连接”的方式将高分辨率图像的视觉特征对齐传入 LLM ，囊括了原图缩略图和原图切分局部图的所有特征。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/28792ceb56c34b29a26ff0328bfd260f~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=%2F3mPYgFMsE30rHwVFpExfq2nafs%3D)

6.  实验显示了使用 Format Prompt（指令后加一个后缀Answer the question using a single word or phrase）、Projection Layer从单个 Linear 改为MLP（两层Linear+Activation）、更丰富的数据（VQA数据，OCR和region-level的数据）、更大的LLM 模型（从7B到13B）、更大的ViT分辨率（从224px到336px）都有利于提高性能。

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/a2f58a2251cf4b31b6e6393718e688d4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=FQ4GXoN4e7OwSzapn3BRYLG%2Bl68%3D)

7.  训练详情如下，和 LLAVA 稍有不同，如 epoch、优化器等

![](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d2ac516ec2b04250a234d91143690cf4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1740724585&x-orig-sign=4ronDye0lAVP29fRaMdVfF190yM%3D)

# LLAMA VISION

https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/

1.  有11B 和 90B 两款模型。
1.  为了添加图像输入支持，我们训练了一组适配器权重，将预训练的图像编码器集成到预训练的语言模型中。适配器由一系列交叉注意层组成，这些层将图像编码器表示输入到语言模型中。我们在文本-图像对上训练了适配器，以使图像表示与语言表示对齐。在适配器训练期间，我们还更新了图像编码器的参数，但有意不更新语言模型参数。通过这样做，我们保持了所有纯文本功能不变，为开发人员提供了 Llama 3.1 模型的直接替代品。
1.  我们的训练流程由多个阶段组成，从预训练的 Llama 3.1 文本模型开始。首先，我们添加图像适配器和编码器，然后在大规模噪声（图像、文本）对数据上进行预训练。接下来，我们在中等规模的高质量领域内和知识增强的（图像、文本）对数据上进行训练。
1.  在后期训练中，我们使用与文本模型类似的方法，在监督微调、拒绝采样和直接偏好优化方面进行多轮对齐。我们利用 Llama 3.1 模型生成合成数据，在域内图像的基础上过滤和扩充问题和答案，并使用奖励模型对所有候选答案进行排名，以提供高质量的微调数据。我们还添加了安全缓解数据，以生成具有高安全水平的模型，同时保留模型的有用性
1.  最终结果是一组可以同时接收图像和文本提示并深入理解和推理两者组合的模型。