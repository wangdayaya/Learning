# 前文
我们常见在使用 `agent` 理解参考文件的时候，只能使用 `RAG` 或者 `BM25` 的方式召回 `docs` 来填充 `prompt` ，这种方式的效果不尽如人意，说白了就是如果单纯靠文本`语义向量`或者`相似度`很难召回恰当的文档，而 `Qwen-agent` 提出了一种新的方式来解决这一问题。


# 准备

首先可以参考官方给出的[`文档`](https://github.com/QwenLM/Qwen-Agent/blob/main/README_CN.md)，先把 `Qwen-agent` 部署到本地，然后可以使用自己的`本地模型`或者`线上大模型的 api-key `。

# 解决方案介绍
### level-1 常规检索
如下图所示，其实这种方式就是常见的 `RAG` 技术路线，流程如下：
- 先将参考的文档切分成 `512 token` 的 `docs` 
- 同时使用`大模型`提取出问题中的 `instruction` 和 `information`，然后再使用`大模型`进一步将 `information` 提取出`问题关键字` 
- 然后根据问题`关键字`，结合`语义向量`或者 `BM25 相似度`从切分好的 `docs` 中来计算召回最相关的内容
- 将召回的 docs 经过裁剪之后填充入大模型的 `prompt` 中，保证不会超出大模型的上下文限制，现在有了问题和召回的 `docs` ，让大模型回答出合适的答案即可。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7876cbe6321d4e91bfecd531b370950f~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723168912&x-orig-sign=xGDf7rLkbbkf6FXmC2fnKw1pAlw%3D)

这种方式的优点是速度快，但是缺点是召回的 docs 相关性较差，不足以提供足够的上下文供大模型解答问题，实际使用中效果有限。如果想尝试效果，可以访问使用[此案例](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/assistant_rag.py)感受一下，或者 bebug 研究内部源码。


### level-2 全文理解

如下图所示，这里在召回文档的部分做出了重要的改进，流程如下，有些步骤是查看源码之后根据我的理解添加的，图中没有展示：
- 先将参考的文档切分成 `512 token` 的 `docs` 
- 同时使用`大模型`提取出问题中的 `instruction` 和 `information`，然后再使用`大模型`进一步将 `information` 提取出`问题关键字` 
- 使用大模型并行去阅读所有的 docs 并完成一个小任务，任务就是让大模型去理解当前的 doc 是否能解决用户的问题，或者是否与问题相关，如果是就让大模型总结并返回`“相关的句子”`，如果不是就直接返回`“不相关”`。因为经过了大模型的理解，所以此时召回的内容肯定是和问题强相关的。
- 将召回的所有句子拼接成长字符串进一步进行解析，使用大模型分解出关键字。
- 然后根据上面的所有`关键字`，结合`语义向量`或者 `BM25 相似度`从所有 docs 中来找回计算排序，召回最相关的 `docs`。
- 将召回的 `docs` 裁剪之后填充入大模型的 `prompt` 中，保证不会超出大模型的上下文限制，现在有了问题和召回的 `docs` ，让大模型回答出合适的答案即可。



![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/2821a9e69999466b9046a52d93cb4899~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723169582&x-orig-sign=yEyYqHFYliZejoaxUr7ZpxUlOUs%3D)

这种方式因为加入了大模型对于所有 docs 的并行理解，所以召回的内容是和问题强相关的，足以帮助大模型结合问题生成合适的答案。如果想尝试效果，可以访问使用[此案例](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/parallel_doc_qa.py)感受一下，或者 bebug 研究内部源码。但是需要注意的是此种方式的缺点就是成本较高。

### level-3 高级智能体

在基于文档的问题回答中，一个典型的挑战是`多跳推理`。例如，考虑回答问题：“与第五交响曲创作于同一世纪的交通工具是什么？”模型首先需要确定子问题的答案，“第五交响曲是在哪个世纪创作的？”即19世纪。然后，它才可以意识到包含“自行车于19世纪发明”的信息块实际上与原始问题相关的。

`Qwen-agent` 中已经定义好了 `ReAct` 智能体作为此种问题的解决方案，它们内置了`问题分解、逐步推理、工具调用`等能力。因此，我们将前述的 `level-2` 智能体封装为一个工具，由 `level-3` 智能体（也就是 `ReAct` ）调用。工具调用智能体进行多跳推理的流程如下：

```
向 level-3 智能体提出一个问题。
while (level-3 智能体无法根据其记忆回答问题) {
    level-3 智能体提出一个新的子问题待解答。
    level-3 智能体向 level-2 智能体提问这个子问题。
    将 level-2 智能体的回应添加到 level-3 智能体的记忆中。
}
level-3 智能体提供原始问题的最终答案。
```


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/eb05e37af3dc4279b2bc9c22a26e6995~tplv-73owjymdk6-jj-mark:0:0:0:0:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1723171471&x-orig-sign=DqOBrU70BiwbNSUneRr4KIQx6RA%3D)

可以访问使用[此案例](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/react_data_analysis.py)感受一下，或者 bebug 研究内部源码。 

# 参考

- https://qwenlm.github.io/blog/qwen-agent-2405/
- https://github.com/QwenLM/Qwen-Agent/blob/main/README_CN.md
- https://github.com/QwenLM/Qwen-Agent/blob/main/examples/react_data_analysis.py
- https://github.com/QwenLM/Qwen-Agent/blob/main/examples/parallel_doc_qa.py
- https://github.com/QwenLM/Qwen-Agent/blob/main/examples/assistant_rag.py