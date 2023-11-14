# 前言
本文参考官网和其他多方教程，将搭建  Langchain-Chatchat 的详细步骤进行了整理，供大家参考。
# 我的硬件
- 4090 显卡
- win10 专业版本
# 搭建环境使用 chatglm2-6b 模型
### 1. 创建虚拟环境 chatchat ，python 3.9 以上
```
conda create -n chatchat python=3.10
```
### 2. 激活环境
 ```
conda activate chatchat
```
### 3. 在自己选好的目录下拉取仓库
```
git clone https://github.com/chatchat-space/Langchain-Chatchat.git
```
### 4. 安装所需要的依赖
```
pip.exe install -r requirements.txt (使用清华源应该能加速下载 -i https://pypi.tuna.tsinghua.edu.cn/simple)
```
### 5. 安装 pytorch  ，进入 `https://pytorch.org/get-started/previous-versions/` 页面搜索 `11.6`（这是我的 cuda 版本），即可找到对应版本的下载命令
```
pip.exe install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
### 6. 拉取 chatglm2-6b 模型（需要配置好 VPN ）
```
git clone https://huggingface.co/THUDM/chatglm2-6b （如果网络稳定应该可以成功下载，如果不稳定，直接去页面点击各个模型进行下载，然后放到指定的目录中即可，反正原则就是将网页的文件都下载到本地目录即可）   
```
### 7. 拉取 m3e 模型（这个和上面同样的操作）
```
git clone https://huggingface.co/moka-ai/m3e-base
```
如果 git 拉取代码的时候报下面的错 Failed to connect to huggingface.co port 443 after 21045 ms: Couldn't connect to server ，将 git 的代理重新设置一下，然后尝试重新 clone 。
```
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

### 8. 将 Langchain-Chatchat/configs 下面的所有以 .example 结尾的文件都复制一份，将原文件名结尾的 .example 去掉，这样得到所有的新文件就是 py 文件，如图所示。下面的 model_config.py 文件需要特殊处理，其他的文件使用默认配置。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a6db45e67e0c43159d78efbbf3f3bdda~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=607&h=239&s=29883&e=png&b=fffefe)

### 9. model_config.py 中的配置需要修改，配置 m3e 和 chatglm2-6b 的模型的绝对路径
```
MODEL_PATH['embed_model']['m3e-base'] 改为自己存放 m3e 的绝对路径 'D:\\m3e-base'
MODEL_PATH['llm_model']['chatglm2-6b'] 改为自己存放 chatglm2-6b 的绝对路径 'D:\\chatglm2-6b'
TEMPERATURE 不建议过高，如果是 Agent 对话或者知识库问答，强烈建议设置为接近 0 或者 0
TEMPERATURE = 0.1
```
同样的道理，如果你使用其他的模型如 `chatglm3-6b` ，那么就提前将 huggingface 中的 chatglm3-6b 项目下载到本地，然后在 `MODEL_PATH["llm_model"]` 中仿照上下文的格式新增一行内容，配置好绝对路径即可。如果想在启动的时候使用 chatglm3-6b  要把 `LLM_MODEL` 参数改为 chatglm3-6b 即可。

 

### 10. 初始化知识库
```
python.exe init_database.py --recreate-vs
```
打印如下表示成功：
```
    database talbes reseted
    recreating all vector stores
    2023-11-08 19:08:33,030 - faiss_cache.py[line:75] - INFO: loading vector store in 'samples/vector_store' from disk.
    {}
    2023-11-08 19:08:33,100 - SentenceTransformer.py[line:66] - INFO: Load pretrained SentenceTransformer: D:\m3e-base
    2023-11-08 19:08:33,560 - loader.py[line:54] - INFO: Loading faiss with AVX2 support.
    2023-11-08 19:08:33,560 - loader.py[line:58] - INFO: Could not load library with AVX2 support due to:
    ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
    2023-11-08 19:08:33,560 - loader.py[line:64] - INFO: Loading faiss.
    2023-11-08 19:08:33,570 - loader.py[line:66] - INFO: Successfully loaded faiss.
    2023-11-08 19:08:33,580 - faiss_cache.py[line:75] - INFO: loading vector store in 'samples/vector_store' from disk.
    Batches: 100%|███████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.08s/it]
    2023-11-08 19:08:34,821 - utils.py[line:287] - INFO: UnstructuredFileLoader used for D:\Langchain-Chatchat\knowledge_base\samples\content\test.txt
```
### 11. 启动项目，因为我的 20000 端口已经被占了，而且杀不掉所以重启电脑再启动项目
```
python.exe .\startup.py -a
```
会打印如下信息：
```
==============================Langchain-Chatchat Configuration==============================
操作系统：Windows-10-10.0.19045-SP0.
python版本：3.10.13 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:24:38) [MSC v.1916 64 bit (AMD64)]
项目版本：v0.2.6
langchain版本：0.0.331. fastchat版本：0.2.31
当前使用的分词器：ChineseRecursiveTextSplitter
当前启动的LLM模型：['chatglm2-6b'] @ cuda
{'device': 'cuda',
 'host': '127.0.0.1',
 'infer_turbo': False,
 'model_path': 'D:\\chatglm2-6b',
 'port': 20002}
当前Embbedings模型： m3e-base @ cuda
==============================Langchain-Chatchat Configuration==============================
2023-11-08 20:20:45,665 - startup.py[line:626] - INFO: 正在启动服务：
2023-11-08 20:20:45,665 - startup.py[line:627] - INFO: 如需查看 llm_api 日志，请前往 D:\Langchain-Chatchat\logs
2023-11-08 20:20:48 | ERROR | stderr | INFO:     Started server process [6772]
2023-11-08 20:20:48 | ERROR | stderr | INFO:     Waiting for application startup.
2023-11-08 20:20:48 | ERROR | stderr | INFO:     Application startup complete.
2023-11-08 20:20:48 | ERROR | stderr | INFO:     Uvicorn running on http://127.0.0.1:20000 (Press CTRL+C to quit)
2023-11-08 20:20:48 | INFO | model_worker | Register to controller
2023-11-08 20:20:48 | INFO | model_worker | Loading the model ['chatglm2-6b'] on worker 928af55b ...
Loading checkpoint shards:   0%|                                                                 | 0/7 [00:00<?, ?it/s]
Loading checkpoint shards:  14%|████████▏                                                | 1/7 [00:01<00:06,  1.09s/it]
Loading checkpoint shards:  29%|████████████████▎                                        | 2/7 [00:02<00:05,  1.14s/it]
Loading checkpoint shards:  43%|████████████████████████▍                                | 3/7 [00:03<00:04,  1.11s/it]
Loading checkpoint shards:  57%|████████████████████████████████▌                        | 4/7 [00:04<00:03,  1.06s/it]
Loading checkpoint shards:  71%|████████████████████████████████████████▋                | 5/7 [00:05<00:02,  1.10s/it]
Loading checkpoint shards:  86%|████████████████████████████████████████████████▊        | 6/7 [00:06<00:01,  1.10s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 7/7 [00:07<00:00,  1.06it/s]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████| 7/7 [00:07<00:00,  1.03s/it]
2023-11-08 20:20:56 | ERROR | stderr |
2023-11-08 20:20:58 | INFO | model_worker | Register to controller
INFO:     Started server process [23280]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:7861 (Press CTRL+C to quit)
==============================Langchain-Chatchat Configuration==============================
操作系统：Windows-10-10.0.19045-SP0.
python版本：3.10.13 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:24:38) [MSC v.1916 64 bit (AMD64)]
项目版本：v0.2.6
langchain版本：0.0.331. fastchat版本：0.2.31
当前使用的分词器：ChineseRecursiveTextSplitter
当前启动的LLM模型：['chatglm2-6b'] @ cuda
{'device': 'cuda',
 'host': '127.0.0.1',
 'infer_turbo': False,
 'model_path': 'D:\\chatglm2-6b',
 'port': 20002}
当前Embbedings模型： m3e-base @ cuda
服务端运行信息：
    OpenAI API Server: http://127.0.0.1:20000/v1
    Chatchat  API  Server: http://127.0.0.1:7861
    Chatchat WEBUI Server: http://127.0.0.1:8501
==============================Langchain-Chatchat Configuration==============================
      Welcome to Streamlit!
      If you’d like to receive helpful onboarding emails, news, offers, promotions,
      and the occasional swag, please enter your email address below. Otherwise,
      leave this field blank.
```
到这里是个坑！！！！大家一定要注意！！！！终端停在这里只是在等待输入，我们还要按下`回车`才行，才会打印下面的成功信息，并自动跳出默认浏览器界面：

```
You can find our privacy policy at https://streamlit.io/privacy-policy
  Summary:
  - This open source library collects usage statistics.
  - We cannot see and do not store information contained inside Streamlit apps,
    such as text, charts, images, etc.
  - Telemetry data is stored in servers in the United States.
  - If you'd like to opt out, add the following to %userprofile%/.streamlit/config.toml,
    creating that file if necessary:
    [browser]
    gatherUsageStats = false
  You can now view your Streamlit app in your browser.
  URL: http://127.0.0.1:8501
```

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/51759f7674a84de5b9fc1caba61d6e68~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1909&h=919&s=73646&e=png&b=fefefe)

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ebf38eb141f349f1aa787ad69cc1f971~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1910&h=921&s=102329&e=png&b=fcfcfc)


### 12. 报错解决  ERROR: RemoteProtocolError: API通信遇到错误：peer closed connection without sending complete message body (incomplete chunked read)


![image.png](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/01b2326e8ed44a898f70417a91abc093~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=715&h=251&s=15425&e=png&b=fff7f7)
```
使用 openai==0.28.1 即可解决问题
```

### 13. 报错解决
上传除了 csv 文件的其他文件都会报错，这个问题在 https://github.com/chatchat-space/Langchain-Chatchat/issues/1642 也有，但是一直没有解决，目前只能处理 csv 文件了，看后续官方是否会解决：

    {'base_url': 'http://127.0.0.1:7861', 'timeout': 60.0, 'proxies': {'all://127.0.0.1': None, 'all://localhost': None, 'http://127.0.0.1': None, 'http://': None, 'https://': None, 'all://': None, 'http://localhost': None}}
    2023-11-10 17:36:15,387 - utils.py[line:287] - INFO: UnstructuredFileLoader used for D:\Langchain-Chatchat\knowledge_base\samples\content\sanguo.txt
    2023-11-10 17:37:15,386 - utils.py[line:95] - ERROR: ReadTimeout: error when post /knowledge_base/update_docs: timed out
    2023-11-10 17:37:15,405 - utils.py[line:287] - INFO: UnstructuredFileLoader used for D:\Langchain-Chatchat\knowledge_base\samples\content\sanguo.txt

### 14. 最终 python 库版本号
```
accelerate                    0.24.1
aiohttp                       3.8.6
aiosignal                     1.3.1
altair                        5.1.2
antlr4-python3-runtime        4.9.3
anyio                         3.7.1
async-timeout                 4.0.3
attrs                         23.1.0
backoff                       2.2.1
beautifulsoup4                4.12.2
blinker                       1.7.0
blis                          0.7.11
cachetools                    5.3.2
catalogue                     2.0.10
certifi                       2023.7.22
cffi                          1.16.0
chardet                       5.2.0
charset-normalizer            3.3.2
click                         8.1.7
cloudpathlib                  0.16.0
colorama                      0.4.6
coloredlogs                   15.0.1
confection                    0.1.3
contourpy                     1.2.0
cryptography                  41.0.5
cycler                        0.12.1
cymem                         2.0.8
dataclasses-json              0.6.1
distro                        1.8.0
effdet                        0.4.1
einops                        0.7.0
emoji                         2.8.0
et-xmlfile                    1.1.0
exceptiongroup                1.1.3
faiss-cpu                     1.7.4
fastapi                       0.104.1
filelock                      3.13.1
filetype                      1.2.0
flatbuffers                   23.5.26
fonttools                     4.44.0
frozenlist                    1.4.0
fschat                        0.2.31
fsspec                        2023.10.0
gitdb                         4.0.11
GitPython                     3.1.40
greenlet                      3.0.1
h11                           0.14.0
httpcore                      0.17.3
httpx                         0.24.1
huggingface-hub               0.17.3
humanfriendly                 10.0
idna                          3.4
importlib-metadata            6.8.0
iniconfig                     2.0.0
iopath                        0.1.10
Jinja2                        3.1.2
joblib                        1.3.2
jsonpatch                     1.33
jsonpointer                   2.4
jsonschema                    4.19.2
jsonschema-specifications     2023.7.1
kiwisolver                    1.4.5
langchain                     0.0.331
langchain-experimental        0.0.38
langcodes                     3.3.0
langdetect                    1.0.9
langsmith                     0.0.60
layoutparser                  0.3.4
lxml                          4.9.3
Markdown                      3.5.1
markdown-it-py                3.0.0
markdown2                     2.4.10
MarkupSafe                    2.1.3
marshmallow                   3.20.1
matplotlib                    3.8.1
mdurl                         0.1.2
mpmath                        1.3.0
msg-parser                    1.2.0
multidict                     6.0.4
murmurhash                    1.0.10
mypy-extensions               1.0.0
networkx                      3.2.1
nh3                           0.2.14
nltk                          3.8.1
numexpr                       2.8.7
numpy                         1.24.0
olefile                       0.46
omegaconf                     2.3.0
onnx                          1.14.1
onnxruntime                   1.15.1
openai                        0.28.1
opencv-python                 4.8.1.78
openpyxl                      3.1.2
packaging                     23.2
pandas                        2.0.3
pathlib                       1.0.1
pdf2image                     1.16.3
pdfminer.six                  20221105
pdfplumber                    0.10.3
peft                          0.6.0
Pillow                        9.5.0
pip                           23.3
pluggy                        1.3.0
portalocker                   2.8.2
preshed                       3.0.9
prompt-toolkit                3.0.39
protobuf                      3.19.0
psutil                        5.9.6
pyarrow                       14.0.0
pyclipper                     1.3.0.post5
pycocotools                   2.0.7
pycparser                     2.21
pydantic                      1.10.13
pydeck                        0.8.1b0
Pygments                      2.16.1
PyMuPDF                       1.23.6
PyMuPDFb                      1.23.6
pypandoc                      1.12
pyparsing                     3.1.1
pypdfium2                     4.23.1
pyreadline3                   3.4.1
pytesseract                   0.3.10
pytest                        7.4.3
python-dateutil               2.8.2
python-decouple               3.8
python-docx                   1.1.0
python-iso639                 2023.6.15
python-magic                  0.4.27
python-magic-bin              0.4.14
python-multipart              0.0.6
python-pptx                   0.6.21
pytz                          2023.3.post1
pywin32                       306
PyYAML                        6.0.1
rapidfuzz                     3.5.2
rapidocr-onnxruntime          1.3.8
referencing                   0.30.2
regex                         2023.10.3
requests                      2.31.0
rich                          13.6.0
rpds-py                       0.12.0
safetensors                   0.4.0
scikit-learn                  1.3.2
scipy                         1.11.3
sentence-transformers         2.2.2
sentencepiece                 0.1.99
setuptools                    68.0.0
shapely                       2.0.2
shortuuid                     1.0.11
simplejson                    3.19.2
six                           1.16.0
smart-open                    6.4.0
smmap                         5.0.1
sniffio                       1.3.0
soupsieve                     2.5
spacy                         3.7.2
spacy-legacy                  3.0.12
spacy-loggers                 1.0.5
SQLAlchemy                    2.0.19
srsly                         2.4.8
starlette                     0.27.0
streamlit                     1.28.1
streamlit-aggrid              0.3.4.post3
streamlit-antd-components     0.2.3
streamlit-chatbox             1.1.10
streamlit-option-menu         0.3.6
svgwrite                      1.4.3
sympy                         1.12
tabulate                      0.9.0
tenacity                      8.2.3
tensorflow-hub                0.15.0
tf2crf                        0.1.33
tf2onnx                       1.15.1
thinc                         8.2.1
threadpoolctl                 3.2.0
tiktoken                      0.5.1
timm                          0.9.10
tokenizers                    0.14.1
toml                          0.10.2
tomli                         2.0.1
toolz                         0.12.0
torch                         1.13.1+cu116
torchaudio                    0.13.1+cu116
torchvision                   0.14.1+cu116
tornado                       6.3.3
tqdm                          4.66.1
transformers                  4.35.0
transformers-stream-generator 0.0.4
typer                         0.9.0
typing_extensions             4.8.0
typing-inspect                0.9.0
tzdata                        2023.3
tzlocal                       5.2
unstructured                  0.10.29
unstructured-inference        0.7.11
unstructured.pytesseract      0.3.12
urllib3                       2.0.7
uvicorn                       0.23.2
validators                    0.22.0
wasabi                        1.1.2
watchdog                      3.0.0
wavedrom                      2.0.3.post3
wcwidth                       0.2.9
weasel                        0.3.4
websockets                    12.0
wheel                         0.41.2
xformers                      0.0.22.post7
xlrd                          2.0.1
XlsxWriter                    3.1.9
yarl                          1.9.2
zipp                          3.17.0
```

### 15. LLM 对话体验

这里使用的是 chatglm3-6b 模型，效果感觉一般。
![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8a79445151b344cfa92f2f94299f9912~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1911&h=917&s=148933&e=png&b=fefefe)

### 16. 知识库问答体验

我将自己的数据做成 `cvs` 格式，然后导入到项目中。因为是知识库问答，所以  `Temperature` 要尽量调整到 `0` 附近，避免模型自由发挥。可以看出每个问题都能回答准确，这是让我满意的一点。如果点开`“知识库匹配结果”`可以看到准确的原文引用内容。



![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/d08081b54da3482f8062fda303aa76a5~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1917&h=925&s=110730&e=png&b=fefefe)
### 16. 上传 csv 文件的时候报错  ERROR: RuntimeError: 从文件 samples/小区描述2_6.csv 加载文档时出错：Error loading D:\描述.csv

经过我“二分法”在文件中不断删除文件内容，最终定位到原来有一行数据是有一个柠檬的图像🍋，如下，将这个柠檬的图像删掉即可：

    “徐大柠🍋手打柠檬茶是一个餐饮服务...”

# 参考

- http://d262l52040.wicp.vip/
- https://github.com/chatchat-space/Langchain-Chatchat/wiki
- https://blog.csdn.net/weixin_43094965/article/details/133044128
- https://www.bilibili.com/video/BV1cj41187cX/?vd_source=66ea1dd09047312f5bc02b99f5652ac6
- https://github.com/chatchat-space/Langchain-Chatchat/wiki/%E5%BC%80%E5%8F%91%E7%8E%AF%E5%A2%83%E9%83%A8%E7%BD%B2#%E7%A1%AC%E4%BB%B6%E8%A6%81%E6%B1%82
- https://github.com/chatchat-space/Langchain-Chatchat/wiki/%E5%8F%82%E6%95%B0%E9%85%8D%E7%BD%AE
- https://blog.csdn.net/IRay21/article/details/116600397


======================================================

============================分割线====================

=====================================================

# 重新搭建环境使用 Baichuan2-13B-Chat-4bits 模型

### 1. 安装 11.8 的 cuda
在 `https://pytorch.org/get-started/locally/`  中可以查看 pytorch 最高支持 `11.8 cuda` 版本，然后进入 `https://developer.nvidia.com/cuda-toolkit-archive` 找到 `CUDA Toolkit 11.8` 进行下载，下载结束之后双击基本上是傻瓜式下一步按钮即可，不懂的可以见参考中的链接。此时重新打开命令行，查看 `nvcc -V` 已经变成了 11.8 版本：

```
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2022 NVIDIA Corporation
    Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
    Cuda compilation tools, release 11.8, V11.8.89
    Build cuda_11.8.r11.8/compiler.31833905_0
```


### 2. 创建支持 python=3.10 的虚拟环境  torch-2.x-py-3.10
先创建号支持 python 3.10 的虚拟环境，然后进行虚拟环境，在浏览器 https://pytorch.org/get-started/locally/ 页面中找到支持 CUDA 11.8 的 pytorch2.1 命令在虚拟环境中进行安装，然后安装项目所需的 requirements.txt 中的库，同上一样。
```
pip3.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 拉取 0.2.6 版本的项目，按照上面的步骤重新配置 Chatchat 项目中的各个文件中的配置

### 4. 初始化数据库


     python.exe .\init_database.py -r

如果遇到卡住不动的情况，直接结束进行，去下一步中找原因

### 5. 启动项目

    python.exe .\startup.py -a

进入浏览器中发现成功启动，可以开始交互。
![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a949770b2ab341178708824f2c736e79~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1913&h=924&s=84859&e=png&b=fefefe)

### 6. 解决启动时候报错 AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'

    安装 transformers==4.33.2 
    
### 7. 解决启动报错  ImportError: Needs import model weight init func to run quantize.

    安装 pip.exe install bitsandbytes==0.41.1

### 8. 解决启动时候报错 RuntimeError:
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:
        python -m bitsandbytes
        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
        
参考这里 https://github.com/TimDettmers/bitsandbytes/issues/869 中的答案。卸载 bitsandbytes 和 bitsandbytes-windows 库，然后编译包 bitsandbytes-0.41.1-py3-none-win_amd64.whl


    pip.exe uninstall bitsandbytes-windows bitsandbytes 
    pip.exe install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl

### 9. 最终 python 库版本号


    accelerate                    0.24.1
    aiohttp                       3.8.6
    aiosignal                     1.3.1
    altair                        5.1.2
    antlr4-python3-runtime        4.9.3
    anyio                         3.7.1
    async-timeout                 4.0.3
    attrs                         23.1.0
    backoff                       2.2.1
    beautifulsoup4                4.12.2
    bitsandbytes                  0.41.1
    blinker                       1.7.0
    blis                          0.7.11
    Brotli                        1.1.0
    cachetools                    5.3.2
    catalogue                     2.0.10
    certifi                       2022.12.7
    cffi                          1.16.0
    chardet                       5.2.0
    charset-normalizer            2.1.1
    click                         8.1.7
    cloudpathlib                  0.16.0
    colorama                      0.4.6
    coloredlogs                   15.0.1
    confection                    0.1.3
    contourpy                     1.2.0
    cryptography                  41.0.5
    cycler                        0.12.1
    cymem                         2.0.8
    dataclasses-json              0.6.2
    distro                        1.8.0
    effdet                        0.4.1
    einops                        0.7.0
    emoji                         2.8.0
    et-xmlfile                    1.1.0
    exceptiongroup                1.1.3
    faiss-cpu                     1.7.4
    fastapi                       0.104.1
    filelock                      3.9.0
    filetype                      1.2.0
    flatbuffers                   23.5.26
    fonttools                     4.44.0
    frozenlist                    1.4.0
    fschat                        0.2.32
    fsspec                        2023.10.0
    gitdb                         4.0.11
    GitPython                     3.1.40
    greenlet                      3.0.1
    h11                           0.14.0
    h2                            4.1.0
    hpack                         4.0.0
    httpcore                      1.0.2
    httpx                         0.25.1
    huggingface-hub               0.17.3
    humanfriendly                 10.0
    hyperframe                    6.0.1
    idna                          3.4
    importlib-metadata            6.8.0
    iniconfig                     2.0.0
    iopath                        0.1.10
    Jinja2                        3.1.2
    joblib                        1.3.2
    jsonpatch                     1.33
    jsonpointer                   2.4
    jsonschema                    4.19.2
    jsonschema-specifications     2023.7.1
    kiwisolver                    1.4.5
    langchain                     0.0.335
    langchain-experimental        0.0.40
    langcodes                     3.3.0
    langdetect                    1.0.9
    langsmith                     0.0.63
    layoutparser                  0.3.4
    lxml                          4.9.3
    Markdown                      3.5.1
    markdown-it-py                3.0.0
    markdown2                     2.4.10
    markdownify                   0.11.6
    MarkupSafe                    2.1.3
    marshmallow                   3.20.1
    matplotlib                    3.8.1
    mdurl                         0.1.2
    mpmath                        1.3.0
    msg-parser                    1.2.0
    multidict                     6.0.4
    murmurhash                    1.0.10
    mypy-extensions               1.0.0
    networkx                      3.0
    nh3                           0.2.14
    nltk                          3.8.1
    numexpr                       2.8.7
    numpy                         1.25.0
    olefile                       0.46
    omegaconf                     2.3.0
    onnx                          1.14.1
    onnxruntime                   1.15.1
    openai                        0.28.1
    opencv-python                 4.8.1.78
    openpyxl                      3.1.2
    packaging                     23.2
    pandas                        2.0.3
    pathlib                       1.0.1
    pdf2image                     1.16.3
    pdfminer.six                  20221105
    pdfplumber                    0.10.3
    peft                          0.6.1
    Pillow                        9.3.0
    pip                           23.3
    pluggy                        1.3.0
    portalocker                   2.8.2
    preshed                       3.0.9
    prompt-toolkit                3.0.40
    protobuf                      3.20.1
    psutil                        5.9.6
    pyarrow                       14.0.1
    pyclipper                     1.3.0.post5
    pycocotools                   2.0.7
    pycparser                     2.21
    pydantic                      1.10.13
    pydeck                        0.8.1b0
    Pygments                      2.16.1
    PyMuPDF                       1.23.6
    PyMuPDFb                      1.23.6
    pypandoc                      1.12
    pyparsing                     3.1.1
    pypdfium2                     4.24.0
    pyreadline3                   3.4.1
    pytesseract                   0.3.10
    pytest                        7.4.3
    python-dateutil               2.8.2
    python-decouple               3.8
    python-docx                   1.1.0
    python-iso639                 2023.6.15
    python-magic                  0.4.27
    python-magic-bin              0.4.14
    python-multipart              0.0.6
    python-pptx                   0.6.23
    pytz                          2023.3.post1
    pywin32                       306
    PyYAML                        6.0.1
    rapidfuzz                     3.5.2
    rapidocr-onnxruntime          1.3.8
    referencing                   0.30.2
    regex                         2023.10.3
    requests                      2.28.1
    rich                          13.6.0
    rpds-py                       0.12.0
    safetensors                   0.4.0
    scikit-learn                  1.3.2
    scipy                         1.11.3
    sentence-transformers         2.2.2
    sentencepiece                 0.1.99
    setuptools                    68.0.0
    shapely                       2.0.2
    shortuuid                     1.0.11
    simplejson                    3.19.2
    six                           1.16.0
    smart-open                    6.4.0
    smmap                         5.0.1
    sniffio                       1.3.0
    socksio                       1.0.0
    soupsieve                     2.5
    spacy                         3.7.2
    spacy-legacy                  3.0.12
    spacy-loggers                 1.0.5
    SQLAlchemy                    2.0.19
    srsly                         2.4.8
    starlette                     0.27.0
    streamlit                     1.27.2
    streamlit-aggrid              0.3.4.post3
    streamlit-antd-components     0.2.3
    streamlit-chatbox             1.1.11
    streamlit-feedback            0.1.2
    streamlit-option-menu         0.3.6
    strsimpy                      0.2.1
    svgwrite                      1.4.3
    sympy                         1.12
    tabulate                      0.9.0
    tenacity                      8.2.3
    tensorflow-hub                0.15.0
    tf2crf                        0.1.33
    tf2onnx                       1.15.1
    thinc                         8.2.1
    threadpoolctl                 3.2.0
    tiktoken                      0.5.1
    timm                          0.9.10
    tokenizers                    0.13.3
    toml                          0.10.2
    tomli                         2.0.1
    toolz                         0.12.0
    torch                         2.1.0+cu118
    torchaudio                    2.1.0+cu118
    torchvision                   0.16.0+cu118
    tornado                       6.3.3
    tqdm                          4.66.1
    transformers                  4.33.2
    transformers-stream-generator 0.0.4
    typer                         0.9.0
    typing_extensions             4.8.0
    typing-inspect                0.9.0
    tzdata                        2023.3
    tzlocal                       5.2
    unstructured                  0.10.30
    unstructured-inference        0.7.11
    unstructured.pytesseract      0.3.12
    urllib3                       1.26.13
    uvicorn                       0.23.2
    validators                    0.22.0
    wasabi                        1.1.2
    watchdog                      3.0.0
    wavedrom                      2.0.3.post3
    wcwidth                       0.2.9
    weasel                        0.3.4
    websockets                    12.0
    wheel                         0.41.2
    xformers                      0.0.22.post7
    xlrd                          2.0.1
    XlsxWriter                    3.1.9
    yarl                          1.9.2
    zipp                          3.17.0

# 参考

- https://blog.csdn.net/ziqibit/article/details/131435252
- https://zhuanlan.zhihu.com/p/647138388
- https://pytorch.org/get-started/locally/
- https://github.com/TimDettmers/bitsandbytes/issues/869 
