# 前言

本文使用 Qwen-7B-Chat 大模型，通过 Agent 调用自定义工具 weathercheck 来获取实时天气状况，还能给出出行穿戴的建议哦。


# 搭建虚拟环境并且安装 11.8 cuda 

之前的文章中都介绍了如何完成这一步，请看这里https://juejin.cn/post/7298927211994333234#heading-21 。

# 拉取 chatchat 项目

按照之前的文章搭建 chatchat 项目环境 https://juejin.cn/post/7298927211994333234#heading-21


# 拉取 Qwen-7B-Chat 模型

使用下面命令从 huggingface 上将模型下载到本地，可能需要梯子请自行解决，或者使用迅雷某些时候也可以下载：

    git clone https://huggingface.co/Qwen/Qwen-7B-Chat
    
# 拉取  m3e 词向量模型

使用下面的命令从 huggingface 上将模型下载到本地，可能需要梯子请自行解决，或者使用迅雷某些时候也可以下载：

    git clone https://huggingface.co/moka-ai/m3e-base
    
# 修改配置和关键代码

将使用到的大模型绝对路径和词向量模型的绝对路径进行配置即可，因为这里要使用 agent 对话模型，所以将强烈建议设置 TEMPERATURE 为接近 0 或者 0 。

    MODEL_PATH['embed_model']['m3e-base'] 改为自己存放 m3e 的绝对路径 'D:\\m3e-base' 
    MODEL_PATH['llm_model']['chatglm2-6b'] 改为自己存放 chatglm2-6b 的绝对路径 'D:\\Qwen-7B-Chat' 
    TEMPERATURE = 0.1

# 注册和风天气的账号

这里需要使用和风天气的 API ，所以到 https://dev.qweather.com/docs/start/ 进行注册获取 API 所需要的 KEY ，注册账号很简单，进入控制台创建新项目，注意要选择`免费订阅`，这样才能免费使用 API 接口。

![05ad148d57e8b54e988e01ec906f017.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/37f97cdab705466ebec8528d29f38b0b~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1903&h=927&s=71333&e=png&b=fcfcfc)

然后进入项目即可看到 API 所需要的 KEY 。

![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/c0d81cc82fdf4da1ae62d6ce485e9f9f~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1917&h=931&s=79252&e=png&b=fbfbfb)

在使用 API 的时候，比如下方使用查询天气的 API ，我们是免费订阅的账户，所以要将 API Host更改为`devapi.qweather.com` 。


![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/22b2b812d7624762bfaae3bac001d380~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1905&h=928&s=160181&e=png&b=fefefe)

# 修改关键代码

将 Langchain-Chatchat-torch2\server\agent\tools\weather.py 中的 `_PROMPT_TEMPLATE` 提示词模板变量改为如下：

````

_PROMPT_TEMPLATE = """
用户会提出一个关于天气的问题，你的目标是拆分出用户问题中的城市， 并按照我提供的工具回答。
例如 用户提出的问题是: 上海天气情况？
则 提取的市和区是: 上海
请注意以下内容:
1. 如果你没有找到城市的内容,则一定要使用 None 替代，否则程序无法运行
2. 如果用户没有指定市 则直接返回缺少信息

问题: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。
```text
${{城市}}
```
... weathercheck(市)...
```output
${{提取后的答案}}
```
答案: ${{答案}}



这是一个例子：
问题: 上海未来天气情况？


```text上海```
...weathercheck(上海)...

```output
预报时间: 1小时后
温度: 6°C
天气: 晴
预报时间: 24小时后
温度: 7°C
天气: 晴

Answer: 现在天气晴，温度 6 度, 未来 24 小时天气晴，温度 7 度。

现在，这是我的问题：

问题: {question}
"""
````

然后将获取天气情况的工具源代码都去掉，换成以下我写的代码，首先是要下载 https://github.com/qwd/LocationList/blob/master/China-City-List-latest.csv 文件，主要是为了可以通过城市名字获取所对应的 ID ，因为获取天气情况的参数是城市 ID 。

```

def getLocationId(city):
    d = collections.defaultdict(str)
    try:
        df = pd.read_csv("D:\Langchain-Chatchat-torch2\server\agent\tools\China-City-List-latest.csv", encoding='utf-8')
    except Exception as e:
        print(e)
    for i, row in df.iterrows():
        d[row['Location_Name_ZH']] = row['Location_ID']
    return d[city] if city in d else ''
```
然后调用 API 来获取城市未来 24 个小时的天气情况。

```
def get_weather(location):
    key = "你的 KEY"
    id = getLocationId(location)
    if not id:
        return "没有这个城市"
    base_url = 'https://devapi.qweather.com/v7/weather/24h?'
    params = {'location': id, 'key': key, 'lang': 'zh'}
    response = requests.get(base_url, params=params)
    data = response.json()
    if data["code"] != "200":
        return "没有这个城市的天气情况"
    return get_weather_info(data)
```
将获取的天气情况进行抽取，因为会返回每个小时的天气情况，所以这里只获取当前时刻的温度和天气描述文本，以及与当前距离未来第 24h 的温度和天气描述文本，组织成字符串进行返回。
```
def get_weather_info(info):
    if info["code"]!="200":
        return "没有这个城市的天气情况"
    result = f'现在天气{info["hourly"][0]["text"]}，温度 {info["hourly"][0]["temp"]} 度, 未来 24 小时天气{info["hourly"][-1]["text"]}，温度 {info["hourly"][-1]["temp"]} 度。'
    return result


def weather(query):
    key = KEY
    if key == "":
        return "请先在代码中填入和风天气API Key"
    try:
        weather_data = get_weather(query)
        return weather_data  + "，并根据天气的不同，给出贴心的开车出行建议，或者行人出行穿戴建议\n"
    except KeyError:
        return "输入的地区不存在，无法提供天气预报"
```
# 启动项目进行对话

启动项目后，设置好如下左侧的参数，然后进行对话训话，发现我们的 agent 对话时候可以调用天气的工具进行外部天气的获取，不仅可以看到天气情况，还能给出贴心的出行穿戴建议（虽然有的建议很扯淡），折叠框中间还能看到中间的思考过程，到此大功告成！。

    python.exe .\startup.py -a



![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/cade695a601b428fb7c5da5ae2192542~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1322&h=929&s=109134&e=png&b=fdfdfd)

# 参考

- https://dev.qweather.com/docs/api/weather/weather-hourly-forecast/
- https://github.com/qwd/LocationList/blob/master/China-City-List-latest.csv
- https://github.com/chatchat-space/Langchain-Chatchat/wiki/%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5#%E5%AE%9E%E9%99%85%E4%BD%BF%E7%94%A8%E6%95%88%E6%9E%9C
- https://huggingface.co/Qwen/Qwen-7B-Chat
- https://huggingface.co/moka-ai/m3e-base