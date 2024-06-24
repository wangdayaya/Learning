# 前文
在业务中如果使用大模型的 tools 方式来解决问题，肯定会遇到一个问题，就是 tools 的数量太多了，tools 本身的描述加上参数的描述整个 token 的数量会超出大模型的输入 token 限制，这种情况需要我们想办法去解决。本文就是讨论遇到这类问题的解决方案。

# 基本认识
首先我们要有以下几点基本的认识：
- 大模型的最强的最本质的能力是`理解能力`，理解能力是个很强的能力，理论上大模型什么都可以干，就像`大脑`一样。
- tools 本质上就是将遇到的问题进行`分类`，再往简单说就是将业务上可能碰到的问题进行 `if-else` 划分，每个 tool 会写好具体的`实现逻辑、入参、返回结果`，我们只需要让大模型干两件事情：第一步就是先让大模型理解用户的问题可以使用哪个 tool 来解决，第二个问题就是使用大模型从用户的问题中抽取出适合该 tool 的入参。这样可以保证在遇到 tools 可以解决的问题集合的时候，输出都是稳定的。避免了原生的大模型的幻觉等原生缺点。


# 基础思路

如果是通常情况下 tools 的数量比较少，其中的参数及其描述也比较少，使用最常见的方式进行即可，我这里用 qwen 的 api 展示效果。可以看到我的 tools 中只有两个：
- 一个是获取当前的时间 `get_current_time` ，其工具描述为`“当你想知道现在的时间时非常有用”`,这个 tool 没有参数，直接返回结果即可。
- 另一个是获取某地的天气 `get_current_weather` ，其工具描述为`“当你想查询指定城市的天气时非常有用”`，这个 tool 需要从问题中获取参数 `location`。

```
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },  
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {  # 查询天气时需要提供位置，因此参数设置为location
                        "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                }
            },
            "required": [
                "location"
            ]
        }
    }
]

messages = [
        {
            "content": '杭州天气如何',
            "role": "user"
        }
]

response = Generation.call(  model='qwen-max',  messages=messages,  tools=tools,  seed=random.randint(1, 10000),   result_format='message' )
```

我们这里问的问题是`杭州天气如何`，那么大模型就会同时去 tools 中查看，自己通过理解问题选择合适的工具为` get_current_weathe`r ，并且提取出了参数`“杭州”`。日志结果如下：

```
{
    "status_code": 200,
    "request_id": "bd803417-56a7-9597-9d3f-a998a35b0477",
    "code": "",
    "message": "",
    "output": {
        "text": null,
        "finish_reason": null,
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_current_weather",
                                "arguments": "{"properties": {"location": "杭州"}, "type": "object"}"
                            },
                            "id": "",
                            "type": "function"
                        }
                    ]
                }
            }
        ]
    },
    "usage": {
        "input_tokens": 222,
        "output_tokens": 27,
        "total_tokens": 249
    }
}
```

# 进阶思路

从上面的日志结果中我们看到，里面有一项是 `input_tokens` ，这个表示我们向大模型中传入的 prompt 的 token 数量，包括了 tools 中的所有文字描述以及你的问题描述的 token 总和。 `qwen-max` 的上限是 `6k` ，而我在业务中需要用到的 tools 数量很大，而且参数的数量及其介绍也很多，这样上面的基础思路就明显无法满足我们的需求，因为一旦将所有的 tools 传入大模型，肯定会报错超出大模型的输入限制。我们可以换一个思路，大模型接收所有的 tools 内容自动做了选 tool 和提参数这两件事情，我可以手动控制分步做这两件事：

1. 第一步就是使用一个简略的 tools ，这个里面只有工具的`名字和描述`，没有参数及其描述，先将这个 tools 传入大模型让大模型能够先选择出正确的 tool 。
2. 第二步再将这个 tool 改造成长度只有 1 的 tools ，传入大模型，让其按照这个 tool 的预定参数来从问题中抽取入参即可。

这样分两步即可解决 tools 太多的问题。其实还可以扩展开来，我们将所有的 tools 进行多级分类，这样前面的几级只负责对问题识别和路由的过程，到最后的某个局部分子树或者叶子节点的时候对应的少量的 tools ，再解决问题就很简单了。

我这里简单举例，大家懂逻辑即可。首先我先简单写了一个 tool_choose_prompt 来让大模型针对问题选择合适的 tool 名字，然后根据名字再使用大模型做具体的任务。

```
tool_choose_prompt = [system_message(
    "你是一个擅长解析问题的助手，你可以根据问题解析出解决这个问题要用哪一类工具，具体有以下几种工具："
    "gxssln_tool、 dxtdpy_tool、 plan_tool，"
    "其中，gxssln_tool 用于解决管线问题、 dxtdpy_tool 用于解决地下通道问题、 plan_tool 用于解决计划、项目、工程的问题。"
    "再返回结果时你只需要返回工具名即可。"),
    user_message(f"我现在的问题是【{question}】")
]
first_response = dashscope.Generation.call(model="qwen-max", messages=tool_choose_prompt, result_format='message')
content = str(first_response["output"]["choices"][0]["message"]["content"])
response = ""
if content == "gxssln_tool":
    response = dashscope.Generation.call(model="qwen-max", messages=tool_prompt, result_format='message', tools=gxssln_tool)
elif content == "dxtdpy_tool":
    response = dashscope.Generation.call(model="qwen-max", messages=tool_prompt, result_format='message', tools=dxtdpy_tool)
elif content == "plan_tool":
    response = dashscope.Generation.call(model="qwen-max", messages=tool_prompt, result_format='message', tools=plan_tool)
print(response['output']['choices'][0]['message']['tool_calls'])
```