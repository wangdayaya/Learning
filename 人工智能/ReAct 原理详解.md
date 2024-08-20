# 前文

我一直对 ReAct 的内部执行流程感到好奇，为何如此神奇能够执行复杂任务，通过阅读源码，终于揭开了面纱，让我们一探究竟。我这里参考的是 Qwen-Agent 的源码。

# prompt 模板

最关键的就是我们的 `prompt` 模板，这是指导大模型如何工作的关键，这里面需要着重介绍是`两部分`：

*   使用 `TOOL_DESC` 模板来填充当前可用的所有的 `tools` 的名字、描述、参数等，然后转换成字符串拼接入完整的 `prompt` 中。
*   另外就是强制大模型在解答问题的过程中要在理解 `Question`  的情况下，`一次或者多次`重复按照 `Thought、Action、Action Input、Observation` 的步骤来调用合适的工具，得到结果来指导下一步的进行，最后使用 `Final Answer` 标识返回最终的答案。

<!---->

    TOOL_DESC = (
        '{name_for_model}: Call this tool to interact with the {name_for_human} API. '
        'What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} {args_format}')

    PROMPT_REACT = """Answer the following questions as best you can. You have access to the following tools:

    {tool_descs}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {query}
    Thought: """

# 将工具信息加入 prompt 模板

这里就是具体能够填充 prompt 中所有工具的详细过程，这里隐去了很多细节，因为在定义工具的时候就会要求填充 `name 、name_for_human 、name_for_model 、args_format 、description 、parameters` ，所以这里获取到所有工具的时候已经有了这些信息，只需要按照下面的代码逻辑将所有的工具按照 TOOL\_DESC 模板要求填充入 `PROMPT_REACT` 中即可，当然为了大模型能够顺利执行，最后都将完整的 `prompt` 都封装成了`Message` 类。

    def _prepend_react_prompt(self, messages: List[Message], lang: Literal['en', 'zh']) -> List[Message]:
        tool_descs = []
        for f in self.function_map.values():
            function = f.function
            name = function.get('name', None)
            name_for_human = function.get('name_for_human', name)
            name_for_model = function.get('name_for_model', name)
            assert name_for_human and name_for_model
            args_format = function.get('args_format', '')
            tool_descs.append(
                TOOL_DESC.format(name_for_human=name_for_human,
                                 name_for_model=name_for_model,
                                 description_for_model=function['description'],
                                 parameters=json.dumps(function['parameters'], ensure_ascii=False),
                                 args_format=args_format).rstrip())
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool.name for tool in self.function_map.values())
        text_messages = [format_as_text_message(m, add_upload_info=True, lang=lang) for m in messages]
        text_messages[-1].content = PROMPT_REACT.format(
            tool_descs=tool_descs,
            tool_names=tool_names,
            query=text_messages[-1].content,
        )
        return text_messages

# 抽取关键信息

这是一个抽取关键信息的函数，因为我们的 prompt 对于输出的格式是有明确的规定的，所以可以通过关键字，提取出来并最后返回, `func_name` 就是 `Action` 、`func_args` 就是 `Action Input`，`text` 就是 `Thought` 。

    def _detect_tool(self, text: str) -> Tuple[bool, str, str, str]:
        special_func_token = '\nAction:'
        special_args_token = '\nAction Input:'
        special_obs_token = '\nObservation:'
        func_name, func_args = None, None
        i = text.rfind(special_func_token)
        j = text.rfind(special_args_token)
        k = text.rfind(special_obs_token)
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`, then it is likely that `Observation` is ommited by the LLM,  # because the output text may have discarded the stop word.
                text = text.rstrip() + special_obs_token  # Add it back.
            k = text.rfind(special_obs_token)
            func_name = text[i + len(special_func_token):j].strip()
            func_args = text[j + len(special_args_token):k].strip()
            text = text[:i]  # Return the response before tool call, i.e., `Thought`
        return (func_name is not None), func_name, func_args, text

# React 执行

将上面的所有过程都串联起来，就是下面的代码，大体流程如下：

1.  先使用  `_prepend_react_prompt` 将所有工具和问题都拼接成完整的 prompt 模板，并封装成 Message
2.  调用大模型输出响应，然后使用  `_detect_tool`，解析出关键的 `Action` 、 `Action Input`，  `Thought`
3.  如果没有检测到  `Action` 那就直接终止函数运行，如果有则调用 `_call_tool` 函数调用相应的工具进行执行，并拿到执行结果作为  `Observation` 并打印出来。
4.  最后将当前所有的 “`thought + f'\nAction: {action}\nAction Input: {action_input}' + observation`”消息，拼接到 `prompt` 模板末尾，形成新的全历史的 `Message`
5.  重复进行 `2-5` 步，如果没有工具可调用或者超出 `MAX_LLM_CALL_PER_RUN` 次数就停止 `ReAct` 执行。

<!---->

    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', **kwargs) -> Iterator[List[Message]]:
        text_messages = self._prepend_react_prompt(messages, lang=lang)

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response: str = 'Thought: '
        while num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            # Display the streaming response
            output = []
            for output in self._call_llm(messages=text_messages):
                if output:
                    yield [Message(role=ASSISTANT, content=response + output[-1].content)]

            # Accumulate the current response
            if output:
                response += output[-1].content

            has_action, action, action_input, thought = self._detect_tool(output[-1].content)
            if not has_action:
                break

            # Add the tool result
            observation = self._call_tool(action, action_input, messages=messages, **kwargs)
            observation = f'\nObservation: {observation}\nThought: '
            response += observation
            yield [Message(role=ASSISTANT, content=response)]

            if (not text_messages[-1].content.endswith('\nThought: ')) and (not thought.startswith('\n')):
                # Add the '\n' between '\nQuestion:' and the first 'Thought:'
                text_messages[-1].content += '\n'
            if action_input.startswith('```'):
                # Add a newline for proper markdown rendering of code
                action_input = '\n' + action_input
            text_messages[-1].content += thought + f'\nAction: {action}\nAction Input: {action_input}' + observation

# 例子

这个例子使用 `ReAct` 一共执行了`两步`，第一步是使用 `code_interpreter 工具`查看了数据，第二步是使用  `code_interpreter 工具`画图。

    def mytest( ):
        bot = ReActChat(llm={'model': 'qwen-max', 'model_server': 'dashscope', 'api_key':'sk-********************'},
                        name='任务小助手',
                        description='一个可以使用工具的智能体助手',
                        function_list=['code_interpreter', 'image_gen'])
        messages = []
        query = '使用 pd.head 函数查看文件前几行，然后帮我绘制一个折线图展示数据'
        file = os.path.join(os.path.dirname(__file__), 'stock_prices.csv')
        messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})
        for response in bot.run(messages):
            pprint(response, indent=2)

### 第一轮最初的 prompt 输入

这里主要是将我需要的两个工具 `code_interpreter` 和 `image_gen` 的信息以及`问题`都加入到了 `prompt` 模板中。

```

Answer the following questions as best you can. You have access to the following tools:

code\_interpreter: Call this tool to interact with the code\_interpreter API. What is the code\_interpreter API useful for? Python代码沙盒，可用于执行Python代码。 Parameters: \[{"name": "code", "type": "string", "description": "待执行的代码", "required": true}] 此工具的输入应为Markdown代码块。

image\_gen: Call this tool to interact with the image\_gen API. What is the image\_gen API useful for? AI绘画（图像生成）服务，输入文本描述和图像分辨率，返回根据文本信息绘制的图片URL。 Parameters: \[{"name": "prompt", "type": "string", "description": "详细描述了希望生成的图像具有什么内容，例如人物、环境、动作等细节描述，使用英文", "required": true}, {"name": "resolution", "type": "string", "description": "格式是 数字*数字，表示希望生成的图像的分辨率大小，选项有\[1024*1024, 720*1280, 1280*720]"}] 此工具的输入应为JSON对象。

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of \[code\_interpreter,image\_gen]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: （上传了 [文件](stock_prices.csv)）

使用 pd.head 函数查看文件前几行，然后帮我绘制一个折线图展示数据
Thought:

```

### 第二轮的 prompt 输入

这里给出了第一次执行的思考过程、调用工具、参数、以及执行结果，这一步的功能主要是完成浏览数据。

    Answer the following questions as best you can. You have access to the following tools:

    code_interpreter: Call this tool to interact with the code_interpreter API. What is the code_interpreter API useful for? Python代码沙盒，可用于执行Python代码。 Parameters: [{"name": "code", "type": "string", "description": "待执行的代码", "required": true}] 此工具的输入应为Markdown代码块。

    image_gen: Call this tool to interact with the image_gen API. What is the image_gen API useful for? AI绘画（图像生成）服务，输入文本描述和图像分辨率，返回根据文本信息绘制的图片URL。 Parameters: [{"name": "prompt", "type": "string", "description": "详细描述了希望生成的图像具有什么内容，例如人物、环境、动作等细节描述，使用英文", "required": true}, {"name": "resolution", "type": "string", "description": "格式是 数字*数字，表示希望生成的图像的分辨率大小，选项有[1024*1024, 720*1280, 1280*720]"}] 此工具的输入应为JSON对象。

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [code_interpreter,image_gen]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: （上传了 [文件](stock_prices.csv)）

    使用 pd.head 函数查看文件前几行，然后帮我绘制一个折线图展示数据
    Thought: 我需要先用代码解释器API来加载数据，并使用pandas的head函数查看数据前几行。
    Action: code_interpreter
    Action Input: 
    ```py
    import pandas as pd
    # Load the data
    df = pd.read_csv("stock_prices.csv")
    # Display the first few rows of the dataframe
    df.head()
    ```
    Observation: execute_result:

    ```
       Unnamed: 0      Date   Open   High    Low  Close    Adj   Close.1    Volume
    0           0  2020/1/3  74.13  74.31  73.60  73.91  73.91  17423000  36237000
    1           1  2020/1/4  73.91  74.20  73.68  74.08  74.00  17376000  36206000
    2           2  2020/1/5  74.08  74.29  73.82  73.93  73.93  17353000  36184000
    3           3  2020/1/6  73.93  74.03  73.71  73.73  73.73  17341000  36184000
    4           4  2020/1/7  73.73  73.86  73.62  73.70  73.68  17338000  36184000
    ```
    Thought: 

### 第三轮 prompt 

这里给出了第一次和第二次的所有执行的思考过程、调用工具、参数、以及执行结果，主要是根据数据画图。


    Answer the following questions as best you can. You have access to the following tools:

    code\_interpreter: Call this tool to interact with the code\_interpreter API. What is the code\_interpreter API useful for? Python代码沙盒，可用于执行Python代码。 Parameters: \[{"name": "code", "type": "string", "description": "待执行的代码", "required": true}] 此工具的输入应为Markdown代码块。

    image\_gen: Call this tool to interact with the image\_gen API. What is the image\_gen API useful for? AI绘画（图像生成）服务，输入文本描述和图像分辨率，返回根据文本信息绘制的图片URL。 Parameters: \[{"name": "prompt", "type": "string", "description": "详细描述了希望生成的图像具有什么内容，例如人物、环境、动作等细节描述，使用英文", "required": true}, {"name": "resolution", "type": "string", "description": "格式是 数字*数字，表示希望生成的图像的分辨率大小，选项有\[1024*1024, 720*1280, 1280*720]"}] 此工具的输入应为JSON对象。

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of \[code\_interpreter,image\_gen]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: （上传了 [文件](stock_prices.csv)）

    使用 pd.head 函数查看文件前几行，然后帮我绘制一个折线图展示数据
    Thought: 我需要先用代码解释器API来加载数据，并使用pandas的head函数查看数据前几行。
    Action: code\_interpreter
    Action Input:

    ```py
    import pandas as pd
    # Load the data
    df = pd.read_csv("stock_prices.csv")
    # Display the first few rows of the dataframe
    df.head()
    ```

    Observation: execute\_result:

           Unnamed: 0      Date   Open   High    Low  Close    Adj   Close.1    Volume
        0           0  2020/1/3  74.13  74.31  73.60  73.91  73.91  17423000  36237000
        1           1  2020/1/4  73.91  74.20  73.68  74.08  74.00  17376000  36206000
        2           2  2020/1/5  74.08  74.29  73.82  73.93  73.93  17353000  36184000
        3           3  2020/1/6  73.93  74.03  73.71  73.73  73.73  17341000  36184000
        4           4  2020/1/7  73.73  73.86  73.62  73.70  73.68  17338000  36184000

    Thought: 现在我了解了数据格式。我需要使用code\_interpreter API来绘制数据的折线图。由于DataFrame中有多列，我们先确定要基于哪一列或几列来绘制折线图。通常，时间序列数据中，我们会以日期为x轴，某一价格（如收盘价Close）为y轴来绘制。让我们继续进行这个操作。
    Action: code\_interpreter
    Action Input:

    ```py
    import matplotlib.pyplot as plt

    # Assuming we want to plot the 'Close' column over time, which seems to be the 'Date' column
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is in datetime format
    df.set_index('Date', inplace=True)  # Set the 'Date' column as the index for plotting

    plt.figure(figsize=(14,7))
    plt.plot(df['Close'])
    plt.title('Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.show()
    ```

    Observation: ![fig-001转存失败，建议直接上传图片文件](<转存失败，建议直接上传图片文件 workspace\tools\code_interpreter\4966e43d-aa9b-4458-a2cd-1dc24b41afc1.png>)
    Thought:

### 最终完整执行结果打印

这里是最终所有的结果信息打印。这里只显示两次的调用工具和执行工具的过程，以及最后的答案。

```
[ { 'content': 'Thought: 我需要先用代码解释器API来加载数据，并使用pandas的head函数查看数据前几行。\n'
               'Action: code_interpreter\n'
               'Action Input:\n'
               '```py\n'
               'import pandas as pd\n'
               '# Load the data\n'
               'df = pd.read_csv("stock_prices.csv")\n'
               '# Display the first few rows of the dataframe\n'
               'df.head()\n'
               '```\n'
               '\n'
               'Observation: execute_result:\n'
               '\n'
               '```\n'
               '   Unnamed: 0      Date   Open   High    Low  Close    Adj   '
               'Close.1    Volume\n'
               '0           0  2020/1/3  74.13  74.31  73.60  73.91  73.91  '
               '17423000  36237000\n'
               '1           1  2020/1/4  73.91  74.20  73.68  74.08  74.00  '
               '17376000  36206000\n'
               '2           2  2020/1/5  74.08  74.29  73.82  73.93  73.93  '
               '17353000  36184000\n'
               '3           3  2020/1/6  73.93  74.03  73.71  73.73  73.73  '
               '17341000  36184000\n'
               '4           4  2020/1/7  73.73  73.86  73.62  73.70  73.68  '
               '17338000  36184000\n'
               '```\n'
               'Thought: 现在我了解了数据格式。我需要使用code_interpreter '
               'API来绘制数据的折线图。由于DataFrame中有多列，我们先确定要基于哪一列或几列来绘制折线图。通常，时间序列数据中，我们会以日期为x轴，某一价格（如收盘价Close）为y轴来绘制。让我们继续进行这个操作。\n'
               'Action: code_interpreter\n'
               'Action Input:\n'
               '```py\n'
               'import matplotlib.pyplot as plt\n'
               '\n'
               "# Assuming we want to plot the 'Close' column over time, which "
               "seems to be the 'Date' column\n"
               "df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' "
               'column is in datetime format\n'
               "df.set_index('Date', inplace=True)  # Set the 'Date' column as "
               'the index for plotting\n'
               '\n'
               'plt.figure(figsize=(14,7))\n'
               "plt.plot(df['Close'])\n"
               "plt.title('Stock Price Over Time')\n"
               "plt.xlabel('Date')\n"
               "plt.ylabel('Closing Price')\n"
               'plt.show()\n'
               '```\n'
               '\n'
               'Observation: '
               '![fig-001](workspace\\tools\\code_interpreter\\4966e43d-aa9b-4458-a2cd-1dc24b41afc1.png)\n'
               'Thought: 我已经成功绘制了股票收盘价随时间变化的折线图。\n'
               'Final Answer: '
               '折线图成功展示了股票价格随时间的变化趋势。从图上我们可以观察到股票收盘价在时间段内的波动情况。如果您需要进一步的分析或者不同数据列的图表，请告知我。',
    'name': '任务小助手',
    'role': 'assistant'}]

```

# 总结

看完整个细节是不是对 ReAct 的内部流程了解了很多，其实说白了这无非是巧妙地将大模型与工具回调函数，借用一个精妙地 prompt 模板，来实现自动循环调用工具来完成任务的过程，没有很复杂。

# 参考

<https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/agents/react_chat.py>
