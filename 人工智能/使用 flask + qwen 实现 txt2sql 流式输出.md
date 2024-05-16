# 前言
一般的大模型提供的 api 都是在提问之后过很久才会返回对话内容，可能要耗时在 `3 秒`以上了，如果是复杂的问题，`大模型在理解和推理的耗时会更长`，这种展示结果的方式对于`用户体验是很差`的。

其实大模型也是可以进行`流式输出`，也就是像 chatgpt 一个字一个字往出崩，这样用户可以一直追踪输出的内容，而不是枯燥的没有止境的等待，本文以我的 `txt2sql` 实际项目为例，简单介绍使用`通义千问 api +  flask 框架`搭建出一个可以流式输出结果的服务。


# txt2sql 任务
我的 txt2sql 任务是基于我的业务数据库内容，用户会提出相关的业务问题，我会让大模型在理解数据库内容的情况下，输出对于问题的理解和思考过程，并最终返回正确的 sql 。

# 准备

- 需要阿里云 [开通DashScope并创建API-KEY](https://help.aliyun.com/zh/dashscope/developer-reference/activate-dashscope-and-create-an-api-key?spm=a2c4g.11186623.0.0.693246c1Izr8bq)，正常情况下通义千问系列的每个大模型有 `100 万`的免费 token 可以白嫖。
- 在实际 python 开发的时候需要安装通义千问特有的库  `DashScope ` ，并将  `api-key ` 设置为 `环境变量 `
- 熟悉通义千问的[流式输出模式](https://help.aliyun.com/zh/dashscope/developer-reference/api-details?spm=a2c4g.11186623.0.0.4c7246c1zEq9vJ#8d583410d7so6)
- 需要安装并熟悉 flask 框架

# 服务

这里的代码虽然很长，但是内容不多，这里需要关心的点有以下几个：

1. flask 的路由函数 `getAnwser` 正常写即可，但是最后的返回为了支持流输出，需要另外封装定义一个函数 `getStream`，并在  `getAnwser`  最后使用下面方式调用 `getStream` 进行流式输出：
```
Response(stream_with_context(getStream()), content_type='text/event-stream')
```
2. 很多关于大模型的 tools 回调、 rag 框架细节、prompt 模板都被我封装了，剩下的就是使用 `get_llm_prompt` 获取最终的 prompt ，然后喂给通义千问最强模型 `qwen-max-longcontext`，设置到参数 `stream=True 和 incremental_output=True`，让通义千问进行流式输出，将获得的  `responses` 结果进行处理即可，结果要用 `yield` 生成输出流数据。
3. 其他的代码是日志管理和异常处理。

```
import logging
from http import HTTPStatus

import dashscope
from flask import request, Flask, Response, stream_with_context
from config import config
from llm import MyCustomLLM
from tools_imp import get_llm_prompt
from my_util import get_question_sql

app = Flask(__name__)
model = MyCustomLLM(config.DB_HOST, config.DB_PORT, config.DB_NAME, config.DB_USER, config.DB_PASS)
logging.basicConfig(level=logging.INFO, encoding="utf-8",
                    filename=config.LOG_PATH, filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
question_sql = get_question_sql()

@app.route('/getAnwser', methods=["POST"])
def getAnwser():
    def getStream():
        data = request.get_json()
        if 'question' not in data or not data['question']:
            yield "无法理解，请重新输入问题"
        question = data['question']
        try:
            prompt = get_llm_prompt(model, question, question_sql)
            dashscope.api_key = config.API_KEY
            llm_response = ""
            responses = dashscope.Generation.call(model="qwen-max-longcontext", messages=prompt, result_format='message', stream=True, incremental_output=True )
            r = None
            for r in responses:
                if r.status_code == HTTPStatus.OK:
                    info = r['output']['choices'][0]['message']['content']
                    llm_response += info
                    yield info
                else:
                    raise Exception("大模型执行报错")
            logging.info(f"llm_response: {llm_response}")
            logging.info(f"input_tokens: {r['usage']['input_tokens']}, output_tokens: {r['usage']['output_tokens']}")
        except BaseException as e:
            logging.error(f'question:{question}, Error: {e}')
            yield f"Error: {str(e)}\n\n".encode()

    return Response(stream_with_context(getStream()), content_type='text/event-stream')



if __name__ == '__main__':
    app.run(config.FLASK_HOST, config.FLASK_PORT, debug=True)
```





# 测试

另外写一个访问 post 请求的测试代码，请求我的服务接口，结果会持续地一点一点打印完整。

```
import requests

url = 'http://localhost:9001/getAnwser'
payload = {"question": "沈塘桥地铁站的信息"}
response = requests.post(url, json=payload, stream=True)
if response.status_code == 200:
    try:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(chunk.decode('utf-8'), end="")   
    except Exception as e:
        print(f"流处理过程中出现错误: {e}")
```


控制台中会一点点持续输出以下内容，就是流式输出样式，但是我没法使用 gif 动态展示，只能直接显示最后的整体内容：
```
您的问题是：沈塘桥地铁站的信息

思考过程：
- 用户想了解关于“沈塘桥地铁站”的具体信息。
- 关键点在于定位到名为“沈塘桥”的地铁站，这涉及到模糊匹配站名。
- 需要从dtzpt表中查询，因为该表存储了地铁站点的详细信息。
- 查询时，需确保返回所有字段信息，以便提供完整详情。

```sql
SELECT * FROM dtzpt WHERE name LIKE '%沈塘桥%'```
```