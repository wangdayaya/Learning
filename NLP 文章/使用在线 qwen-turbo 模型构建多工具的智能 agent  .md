# 前言
    
本文以 chatchat 项目为框架，以线上大模型 `qwen-turbo` 的基础构建智能 `agent` ，在已有知识库的基础上，如果用户提问知识库相关的内容则调用 `知识库查询工具` 获取相关知识进行回复。如果用户提问其他的天气相关问题，则调用 `天气查询工具`（[前文](https://juejin.cn/post/7301942327818977320)已有详细过程，此处不再赘述）来获取相关在线信息。

# 环境搭建和项目拉取

见[前文](https://juejin.cn/post/7298927211994333234) 中有详细部署过程，此处不做赘述。

# qwen-turbo api-key 获取

1. 先注册阿里云账号并进行登陆
2. 访问然后按照[申请教程](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key)获取 api-key
3. 访问 [安装教程](https://help.aliyun.com/zh/dashscope/developer-reference/install-dashscope-sdk) ，安装环境，如下命令：

        pip.exe install dashscope

4. 使用代码，将` dashscope.api_key` 换成你申请的 KEY 运行代码进行测试。

    ```
    from http import HTTPStatus
    import dashscope

    def call_with_messages():
        dashscope.api_key = '你申请的KEY'
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': '如何做炒西红柿鸡蛋？'}]

        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_turbo,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            print(response)
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))

    if __name__ == '__main__':
        call_with_messages()
    ```

    如果正常返回内容如下，说明一切正常，可以调用：
      ```
        {"status_code": 200, "request_id": "0c2cd38b-f02c-9407-b891-ff73432317ae", "code": "", "message": "", "output": {"text": null, "finish_reason": null, "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "首先，你需要准备以下材料：两个新鲜的西红柿、三个鸡蛋、一小勺盐和适量油。然后，把西红柿切成小块，把鸡蛋打散。接着，在锅里放少量油，将西红柿炒至出汁后加入鸡蛋，继续翻炒，最后撒上少许盐即可完成。"}}]}, "usage": {"input_tokens": 12, "output_tokens": 65, "total_tokens": 77}}
      ```
# 修改配置文件 model_config.py

设置` model_config.py` 的` "qwen-api"` 字典内容改为如下, 其中 api_key 改成和上面一样你申请的 KEY 值： 

    {
        "version": "qwen-turbo",  # 可选包括 "qwen-turbo", "qwen-plus"
        "api_key": "你申请的KEY",  # 请在阿里云控制台模型服务灵积API-KEY管理页面创建
        "provider": "QwenWorker",
    }
    
# 修改配置文件 server_config.py

在 `FSCHAT_MODEL_WORKERS` 字典中加入下内容，这里是为了要给每个运行的在线 API 设置不同的端口：

```
"qwen-api": {
    "port": 21006,
},
```

# 修改配置文件 prompt_config.py

在 `PROMPT_TEMPLATES["knowledge_base_chat"]` 字典中新增下面键值对，这里主要是在配置知识库问答的时候的 prompt 模板：

```
"knowledge_base_chat":
    """
    <指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>
    <已知信息>{{ context }}</已知信息>、
    <问题>{{ question }}</问题>
    """,
```


修改 `PROMPT_TEMPLATES["agent_chat"]["default"]` 的内容改为如下：

```
"""
Answer the following questions as best you can by using some tools appropriately. Please use Chinese for the answer.
If you cannot get an answer from tools, please say "无法解答该问题"， No fabrication is allowed in the answer.
You have access to the following tools:

{tools}

请注意，“天气查询工具”只能用来回答询问城市天气的问题；
请注意，除了天气相关的问题外，其他问题都是用“知识库查询工具”用来所有的专业性问题，如果从知识库获取不到相关信息则回答”无法解答该问题“；

Use the following format:
Question: the input question you must answer1
Thought: you should always think about what to do and what tools to use.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


Begin!
history:
{history}
Question: {input}
Thought: {agent_scratchpad}
"""
```

# 修改 search_all_knowledge_once.py


将` _PROMPT_TEMPLATE `替换为下面的字符串内容。这个提示模板主要是引导大模型将用户的问题进行理解并按照格式解析出相应的参数。

````
_PROMPT_TEMPLATE = """
用户会提出一个需要你查询知识库的问题，你应该按照我提供的思想进行思考
例如 用户提出的问题是: "请从知识库 samples 中查询复兴城市家园的地址"，则按照格式提取的数据库名字和用户提问分别是:  "samples,复兴城市家园的地址"
例如 用户提出的问题是: "请从知识库 address 中查询复兴城市家园的guid"，则按照格式提取的数据库名字和用户提问分别是:  "address,复兴城市家园的guid"
这些是你能访问的知识库名称：[{database_names}]

Question: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。

```text
${{知识库名字}}
```

```output
${{知识库查询的结果}}
```

答案: ${{答案}}

这是一个例子：

问题: 请从知识库samples中查询复兴城市家园的地址

```text samples,复兴城市家园```

```output
复兴城市家园是一个小区，它位于杭州市上城区南星街道玉皇山社区。

Answer: 复兴城市家园是一个小区，地址是杭州市上城区南星街道玉皇山社区。

现在，这是我的问题：
问题: {question}

"""
````
在 `_PROMPT_TEMPLATE` 下面一行我们将加入下面代码，否则报错` 'nonetype' object has no attribute 'items'`：


```
kb_list = {x["kb_name"]: x for x in get_kb_details()}
model_container.DATABASE = {name: details['kb_info'] for name, details in kb_list.items()}
```


同时因为上面的 prompt 模式中会将数据库和问题用“,”隔开，所以后面为了提取相关的模型返回信息，将 `database = text_match.group(1).strip()` 修改为 `database = text_match.group(1).strip().split(",")[0]` 。

# 修改 tool_select.py 文件

这里将我们定义的两个工具 天气查询工具（具体配置见[前文](https://juejin.cn/post/7301942327818977320)） 、知识库查询工具都列出来，方便调用。

```
from langchain.tools import Tool
from server.agent.tools import *
tools = [
    Tool.from_function(
        func=weathercheck,
        name="天气查询工具",
        description="访问互联网，使用这个工具查询中国各城市未来24小时的天气",
    ),
    Tool.from_function(
        func=knowledge_search_once,
        name="知识库查询工具",
        description="除去天气相关的问题，其他问题都优先访问知识库来获取答案",
    ),
]

tool_names = [tool.name for tool in tools]
```

# 启动项目

首先将准备好的数据切分成小文件，导入` samples` 知识库中。



![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/2b44489b838f4c03b71476f8a225fe9f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1307&h=922&s=91180&e=png&b=fcfcfc)


然后选择好左边的参数，对话模式为`“自定义Agent问答”`，LLM 模型为配置的 `“qwen-api(Running)” `, Temperature 禁止设置为零，否则会报错 `"Temperature should be greater than 0.0"` ，所以设置为 `0.05` 。

然后提问`“请根据samples知识库查询西湖花苑的纬度”` ，可以发现调用了`“知识库查询工具”`，并经过了`“思考”`，完成了回答`“西湖花苑的纬度是30.263473855231503”`，答案经过查证是完全正确的 。


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/dcf39215c3d84258b200f44d2215b8d5~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1313&h=926&s=77271&e=png&b=fdfdfd)

然后提问`“杭州天气如何”`，发现调用了“天气查询工具” ，通过调用外部接口得到数据，并进行思考总结出来回答`“杭州现在的天气晴朗，气温为19度。建议您出门携带雨具以备不时之需，同时注意防晒和保暖。”` 。，答案经过查证是完全正确的

![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/4cfa65ccf17b4d3b8e4758bbec100782~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1324&h=898&s=80132&e=png&b=fdfdfd)
 
至此两个工具可以自由调用，自动根据用户问题进行答案的回复。
 
# 参考

- https://juejin.cn/post/7298927211994333234
- https://juejin.cn/post/7301942327818977320
- https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key
- https://help.aliyun.com/zh/dashscope/developer-reference/install-dashscope-sdk
- https://github.com/chatchat-space/Langchain-Chatchat

