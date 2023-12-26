# 前言

本文使用最新的大模型 chatglm3-6b ，调用自定义工具计算两个地址的经纬度之间的距离。
 
# 安装 11.8 的 cuda

在 `https://pytorch.org/get-started/locally/` 中可以查看 pytorch 最常见的是支持 `11.8 cuda` 版本，然后进入 `https://developer.nvidia.com/cuda-toolkit-archive` 找到 `CUDA Toolkit 11.8` 进行下载，下载结束之后双击基本上是傻瓜式下一步按钮即可，不懂的可以见参考中的链接。此时重新打开命令行，查看 `nvcc -V` 已经变成了 11.8 版本：

```
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
    Cuda compilation tools, release 11.8, V11.8.89
    Build cuda_11.8.r11.8/compiler.31833905_0
```
# 搭建 conda 虚拟环境

创建 3.10 的 python 版本来生成虚拟环境。

```
conda create -n torch-2.x-py-3.10 python=3.10
```
激活虚拟环境

```
conda activate torch-2.x-py-3.10
```



在浏览器 <https://pytorch.org/get-started/locally/> 页面中找到支持 CUDA 11.8 的 pytorch2.1 命令在虚拟环境中进行安装。

```
pip3.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# 拉取项目 ChatGLM3 到本地


    git clone https://github.com/THUDM/ChatGLM3
    
# 安装项目所需要的库

```
pip.exe install -r requirements.txt
```

# 修改关键代码

从 https://huggingface.co/THUDM/chatglm2-6b 将完整的模型下载到本地，将 langchain_demo/main.py 文件改为下面的代码，其中主要修改的部分是 model_path 是模型的绝对路径，另外使用  run_tool 调用 Calculator 。


```
from typing import List
from ChatGLM3 import ChatGLM3
from langchain.agents import load_tools
from Tool.Calculator import Calculator
from langchain.agents import initialize_agent
from langchain.agents import AgentType


def run_tool(tools, llm, prompt_chain: List[str]):
    loaded_tolls = []
    for tool in tools:
        if isinstance(tool, str):
            loaded_tolls.append(load_tools([tool], llm=llm)[0])
        else:
            loaded_tolls.append(tool)
    agent = initialize_agent(
        loaded_tolls, llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    for prompt in prompt_chain:
        agent.run(prompt)


if __name__ == "__main__":
    model_path = "D:\chatglm3-6b"
    llm = ChatGLM3()
    llm.load_model(model_name_or_path=model_path)

    # calculator: 单个工具调用示例 3
    run_tool([Calculator()], llm, [
        "复兴城市家园和清河家园经的经纬度是120.15235、30.210432 和 120.1531、30.20865，计算他们之间的距离？",
    ])
```



将 langchain_demo/Tool/Calculator.py 改成下面的代码，主要实现了一个计算两个经纬度之间的直线距离的函数。



```
import abc
from typing import Any
from langchain.tools import BaseTool
from math import radians, sin, cos, sqrt, atan2

class Calculator(BaseTool, abc.ABC):
    name = "Calculator"
    description = "Useful for when you need to answer questions about math"
    def __init__(self):
        super().__init__()
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        # 用例中没有用到 arun 不予具体实现
        pass

    def haversine_distance(self,  lon1, lat1, lon2, lat2 ):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        R = 6371.0 * 1000
        distance = R * c
        return distance

    def _run(self, para: str) -> str:
        A,B = para.split('-')
        lon1, lat1 = A.split(",")
        lon2, lat2 = B.split(",")
        return str(self.haversine_distance(float(lon1), float(lat1), float(lon2), float(lat2))) + "米"


if __name__ == "__main__":
    calculator_tool = Calculator()
    result = calculator_tool.run("120.15235,30.210432-120.1531,30.20865")
    print(result)
```

启动 main.py ，可以看到打印下面的结果，大模型从输入中提取出来了两个经纬度，并且传入到了自定义的计算距离的工具，可以返回结果为 210.849 米 ，下面是整个思考和结论过程：

    Loading checkpoint shards: 100%|██████████| 7/7 [00:03<00:00,  2.21it/s]
    > Entering new AgentExecutor chain...
    ======
    System: Respond to the human as helpfully and accurately as possible. You have access to the following tools:

    Calculator: Useful for when you need to answer questions about math, args: {{'para': {{'title': 'Para', 'type': 'string'}}}}

    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

    Valid "action" values: "Final Answer" or Calculator

    Provide only ONE action per $JSON_BLOB, as shown:

    ```
    {
      "action": $TOOL_NAME,
      "action_input": $INPUT
    }
    ```

    Follow this format:

    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "Final response to human"
    }
    ```

    Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
    Thought:
    Human: 复兴城市家园和清河家园经的经纬度是120.15235、30.210432 和 120.1531、30.20865，计算他们之间的距离？


    ======
    Action: 
    ```
    {"action": "Calculator", "action_input": "120.15235,30.210432-120.1531,30.20865"}
    ```
    Observation: 210.84897095947352米
    Thought:======
    System: Respond to the human as helpfully and accurately as possible. You have access to the following tools:

    Calculator: Useful for when you need to answer questions about math, args: {{'para': {{'title': 'Para', 'type': 'string'}}}}

    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

    Valid "action" values: "Final Answer" or Calculator

    Provide only ONE action per $JSON_BLOB, as shown:

    ```
    {
      "action": $TOOL_NAME,
      "action_input": $INPUT
    }
    ```

    Follow this format:

    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "Final response to human"
    }
    ```

    Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
    Thought:
    Human: 复兴城市家园和清河家园经的经纬度是120.15235、30.210432 和 120.1531、30.20865，计算他们之间的距离？

    This was your previous work (but I haven't seen any of it! I only see what you return as final answer):

    Action: 
    ```
    {"action": "Calculator", "action_input": "120.15235,30.210432-120.1531,30.20865"}
    ```
    Observation: 210.84897095947352米
    Thought:
    ======

    Action: 
    ```
    {"action": "Final Answer", "action_input": "复兴城市家园和清河家园之间的距离约为210.849米。"}
    ```

    > Finished chain.


# 参考

- https://blog.csdn.net/ziqibit/article/details/131435252
- https://github.com/THUDM/ChatGLM3
 