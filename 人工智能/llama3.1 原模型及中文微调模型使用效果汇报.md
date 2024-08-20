# 前文

llama3.1 强势来袭，我也是第一时间就下载使用了，下面给各位领导汇报一下最新的成果和使用体验感受。下面是官方的性能图，请欣赏开源的强大力量，闭源估计在瑟瑟发抖。


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/e8962aed5a2e4ca7aaf68db99be572c5~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722047026&x-orig-sign=mF3%2FVH1Job%2BrNMJheTrPSW966i8%3D)

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1c69b035c0a84ba487ec310656559058~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722047022&x-orig-sign=eVrZgv2rKYFAX00LZouBiqH4wos%3D)
# 准备

因为我是第一天就直接使用，在 [huggingface llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) 上面的模型还没法直接使用，需要提交申请，只有通过之后才能使用，我这里已经过去两天了还没有任何消息，果断放弃了。

现在最方便的是使用 `Ollama` 框架平台来拉取 `llama 3.1` 的模型，分别有 `8B 、70B、405B` ，我这里的硬件只能支撑 `8B` 。首先你要先保证安装好 `Ollama` ，如果不会可以看我这个[教程](https://juejin.cn/post/7395404699599224870) ，包教包会。然后在命令行使用下面的命令拉取模型。

    ollama run llama3.1
    
如果拉取成功，我们可以通过下面的命令看到本地的模型列表。


    ollama list

<p align=center><img src="https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/1611667d4aad4f67a4247685897c8440~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722045149&x-orig-sign=GdFfI1iKxcKa%2B3gkh%2F22ip7Wt%2FI%3D" alt="image.png"  /></p>

然后我们通过命令 `ollama ps` 查看模型是否已经运行，如果没有运行起来我们使用下面的代码运行模型。

    ollama run llama3.1 

到此为止原始的 llama3.1-8b 的模型已经运行起来了，显存占 7G 左右，正常情况命令行就可以进行互动交流，让我们看下效果吧。

# llama3.1-8B 效果展示


回答的速度相当快，简单的互动都可以完成。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0c72baba5da148a88ca42384da559cb2~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722045619&x-orig-sign=OPogH6nkjFxAdhRTr3Qqq6Sm5vw%3D)

询问有难度的问题也可以快速响应，效果符合预期。


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/2b5449a8a9b24e46be6646b510600e5a~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722046010&x-orig-sign=pAAQscZFEFR9B21eJ6D2Y4ovehU%3D)

# llama3.1-8B 缺点

在和业内人士交流的时候发现这个模型对于中文其实还不友好，有时候会出现乱答甚至不答的情况，所以为了能更好支持中文任务，我找到了经过中文微调数据的模型 [Llama-3.1-8B 中文 DPO 模型](https://www.modelscope.cn/models/LCoder/llama3.1-8B-Chinese-Instruct-DPO) 开始部署使用。

# 准备 Llama-3.1-8B 中文 DPO 模型 

其实就是直接去[官方页面](https://www.modelscope.cn/models/LCoder/llama3.1-8B-Chinese-Instruct-DPO/files)
把所有文件都下载下来就可以了。不得不感叹开源的力量，真的是太大了，时隔一天就能出来微调版本，我相信后面应该还能继续出来效果更好适合中文的模型。

# 体验 Llama-3.1-8B 中文 DPO 模型 

因为之前我自己搭了一个开源的大模型
聊天界面，所以我直接就使用自己的这个工具进行聊天，如果想学习的同学可以看我这篇[教程](https://juejin.cn/post/7394444427584208907)。我将自己下载好的模型放到 `text-generation-webui\models` 目录之下。如果你已经安装好，那么使用下面的命令启动即可：

    python .\server.py
    
启动成功之后，直接访问下面的页面：

    http://127.0.0.1:7860
    
 接下来我们还要做一下简单的配置，就是下图所示的几个步骤：
 
1.  切换到 `Model` 页面
2.  选择我们自己的模型 `llama3.1-8B-Chinese-Instruct-DPO`
3.  点击 `Load` 按钮加载模型
4.  如果显示 `Successfully` 就说明成功了，成功跑起来大约需要 `16G` 的显存。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/f2bb05ac5a0e4d93bcc4e2bac3bdad32~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722046643&x-orig-sign=dkJ7pkIfPfKpXuOEpSdVt0Dzb9Q%3D)

然后切换到 `Chat` 页面就可以进行对话了，我直接问的就是比较有难度的业务问题，看起来回答的也比较符合预期。


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/01c98c5bee304a52bbdbcd535b7aa6fc~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1722046929&x-orig-sign=Wk7xhsnrX8eOhvIBjLBu0LDq7dQ%3D)


# 总结

- 开源的力量是越来越强大，`Meta` 这条路子总算是走对了，未来 `LLAMA` 成为大模型一个指日可待，干翻闭源模型也指日可待。
- 其实不管怎么说 `8B` 的模型大小也就是能随便玩玩，想正式进入商用阶段还得使用 `70B` 、甚至 `405B` ，如果真的有一天 `8B` 的模型能够商用，也就是小模型能在消费级显卡上面支持商用，那真的是人工智能的到来之日。
- 就在发文的功夫，最强开源的位置易主了，已经是 `Mistral Large 2` ，`LLAMA3.1` 的霸主之位只坐了一天，再次感叹开源力量，太卷了。

