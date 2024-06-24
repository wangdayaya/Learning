# 前文

大家肯定对使用大模型的函数回调或者说 Tools 已经耳熟能详了，那么他们具体内部是如何运作的呢，本文就此事会详细给大家介绍具体的细节。

# tools

首先是大家最熟悉的环节，定义两个 tool 的具体实现，其实就是两个函数：

*   `get_current_weather`：基于给定的城市获取天气
*   `get_current_time`：查询当前时间

我这里为了演示说明，所以只是简单实现了功能，里面的逻辑并不复杂，大家可以看下面的代码，然后将这两个函数的具体功能描述以及对参数的定义使用 `json` 格式存放在列表 `tools` 中。tools 中的这些对于函数的功能描述或者参数的定义描述等等信息后面会传入到大模型中，大模型会解析改写成它能理解的 `prompt` 。所以这里按照固定的格式写 tools ，就是为了源代码中能够解析出相关的内容。具体可以看继续后面。

```

def get_current_weather(location):
    if '上海' in location.lower():
        return json.dumps({'location': '上海', 'temperature': '10度'}, ensure_ascii=False)
    elif '杭州' in location.lower():
        return json.dumps({'location': '杭州', 'temperature': '12度'}, ensure_ascii=False)
    else:
        return json.dumps({'location': location, 'temperature': 'unknown'}, ensure_ascii=False)


def get_current_time():
    current_datetime = datetime.now()
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    return f"当前时间：{formatted_time}。"

tools = [
    {
        'name': 'get_current_weather',
        'description': '基于给定的城市获取天气',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': '城市名称',
                },
            },
            'required': ['location'],
        },
    },
    {
        "name": "get_current_time",
        "description": "获取当前时间",
        "parameters": {}
    }
]

```

# 查询天气

首先演示查询天气的 tool ，此时我们先给大模型定义一个 `system` 级别的角色定义，将其设定为一个可以使用工具的帮手，可以从我在 tools 定义的两个工具中挑出合适的工具完成任务。理论上我提问“`上海天气如何`”，然后大模型能够自动挑选出 `get_current_weather` 这个工具来完成天气查询任务。我这里使用的是 `qwen-max` 模型，大家可以换成自己的，大模型选用不是本文重点，大家不必特别关心。

整个过程分为以下几步：

1.  给大模型传入问题 `上海天气如何`
2.  大模型自己从 `tools` 中挑选可以解决用户问题的工具  `get_current_weather`
3.  使用工具 `get_current_weather` 和大模型提取出来的参数 `上海` 获取天气情况
4.  将天气情况传给大模型
5.  大模型结合用户的问题，以及得到的天气情况，总结返回最后的结果。

具体第 2 和 3 步是如何实现的呢，其实大模型在拿到我们定义的存放函数信息的 `json` 格式的 `tools` 之后，将里面所有的 `tools` 内容拼接成了一个很长的 `prompt`，通过 `debug` ，我这里拿到了使用 `qwen-max` 的中间结果如下，可以看出其实就是将我们最原始的 `system` 定义的内容填充入了 `tools` 的内容供大模型理解，并按照固定的参数定义从用户的问题中提取参数，这些对原有的对话信息进行修改的过程普通用户是没有感觉的，也不会改变用户真实的对话信息，因为已经封装入了 `api` 中直接供大家使用了。

```
[Message({'role': 'system', 'content': '你是一个有用的帮手，可以使用合适的工具解决我的我问题
# 工具
## 你拥有如下工具：
### get_current_weather
get_current_weather: 基于给定的城市获取天气 输入参数：{"type": "object", "properties": {"location": {"type": "string", "description": "城市名称"}}, "required": ["location"]}
### get_current_time
get_current_time: 获取当前时间 输入参数：{}
## 你可以在回复中插入零次、一次或多次以下命令以调用工具：
✿FUNCTION✿: 工具名称，必须是[get_current_weather,get_current_time]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果，需将图片用![](url)渲染出来。
✿RETURN✿: 根据工具结果进行回复'}),

Message({'role': 'user', 'content': '上海天气如何'})]
```
代码如下：
 
```

def get_weather():
    llm = get_chat_model({'model': 'qwen-max',  'model_server': 'dashscope', 'api_key': 'sk-c69985b9a3c94cd5a56f8cd003a3cf08'})
    messages = [{'role': 'system', 'content': "你是一个有用的帮手，可以使用合适的工具解决我的我问题"}, {'role': 'user', 'content': "上海天气如何"}]
    response = llm.chat(messages=messages, functions=tools, stream=False)
    print(f'# Assistant 回复 1:\n{response}')
    messages.extend(response)
    if messages:
        last_response = messages[-1]
        if last_response.get('function_call', None):
            function_name = last_response['function_call']['name']
            if function_name == 'get_current_weather':
                function_args = json.loads(last_response['function_call']['arguments'])
                function_response = get_current_weather(function_args.get('location'))
                print(f'# Function 回复:\n{function_response}')
                messages.append({'role': 'function', 'name': function_name, 'content': function_response, })
                print(f'# All messages:\n {messages}')
                response = llm.chat(messages=messages,  functions=tools, stream=False)
                print(f'# Assistant 回复 2:{response}')

```

日志打印：

```

# Assistant 回复 1:

\[{'role': 'assistant', 'content': '', 'function\_call': {'name': 'get\_current\_weather', 'arguments': '{"location": "上海"}'}}]

# Function 回复:

{"location": "上海", "temperature": "10度"}

# All messages:

\[{'role': 'system', 'content': '你是一个有用的帮手，可以使用合适的工具解决我的我问题'},
{'role': 'user', 'content': '上海天气如何'},
{'role': 'assistant', 'content': '', 'function\_call': {'name': 'get\_current\_weather', 'arguments': '{"location": "上海"}'}},
{'role': 'function', 'name': 'get\_current\_weather', 'content': '{"location": "上海", "temperature": "10度"}'}]

# Assistant 回复 2:

\[{'role': 'assistant', 'content': '上海现在的天气是10度。'}]

```


# 查询时间

相信大家看了上面的解释已经对整个内部的细节有了一个新的认识，这里我在简单举例，让大模型自动挑选 tools 中的工具完成查看时间的任务。过程和上面一样，不再具体赘述。

大模型结合 tools 改写的 prompt 如下：

```

\[Message({'role': 'system', 'content': '你是一个有用的帮手，可以使用合适的工具解决我的我问题

# 工具

## 你拥有如下工具：

### get\_current\_weather

get\_current\_weather: 基于给定的城市获取天气 输入参数：{"type": "object", "properties": {"location": {"type": "string", "description": "城市名称"}}, "required": \["location"]}

### get\_current\_time

get\_current\_time: 获取当前时间 输入参数：{}

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是\[get\_current\_weather,get\_current\_time]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果，需将图片用![转存失败，建议直接上传图片文件](<转存失败，建议直接上传图片文件 url>)渲染出来。
✿RETURN✿: 根据工具结果进行回复'}),

Message({'role': 'user', 'content': '当前时间几点了'})]

```



代码如下：
```

def get_time():
    llm = get_chat_model({'model': 'qwen-max',  'model_server': 'dashscope', 'api_key': '你的key'})
    messages = [{'role': 'system', 'content': "你是一个有用的帮手，可以使用合适的工具解决我的我问题"}, {'role': 'user', 'content': "当前时间几点了"}]
    response = llm.chat(messages=messages, functions=tools, stream=False)
    print(f'# Assistant 回复 1:\n{response}')
    messages.extend(response)
    if messages:
        last_response = messages[-1]
        if last_response.get('function_call', None):
            function_name = last_response['function_call']['name']
            if function_name == 'get_current_time':
                function_response = get_current_time()
                print(f'# Function 回复:\n{function_response}')
                messages.append({'role': 'function', 'name': function_name, 'content': function_response, })
                print(f'# All messages:\n {messages}')
                response = llm.chat(messages=messages,  functions=tools, stream=False)
                print(f'# Assistant 回复 2:{response}')

```
日志打印：
```

# Assistant 回复 1:

\[{'role': 'assistant', 'content': '', 'function\_call': {'name': 'get\_current\_time', 'arguments': '{}'}}]

# Function 回复:

当前时间：2024-06-12 19:00:48。

# All messages:

\[{'role': 'system', 'content': '你是一个有用的帮手，可以使用合适的工具解决我的我问题'},
{'role': 'user', 'content': '当前时间几点了'},
{'role': 'assistant', 'content': '', 'function\_call': {'name': 'get\_current\_time', 'arguments': '{}'}},
{'role': 'function', 'name': 'get\_current\_time', 'content': '当前时间：2024-06-12 19:00:48。'}]

# Assistant 回复 2:

\[{'role': 'assistant', 'content': '当前时间是2024年6月12日19点00分48秒。'}]

```

# 总结

所以从上面的例子可以看出来，本质上大模型理解 tools 还是在拼写 prompt ，我们如果不使用 api ，直接自己拼写可以使用的函数信息，其实也是可以实现上述的功能。
