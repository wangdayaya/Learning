# 前文

最近微软新推出来的技术 graphRAG  相当火爆，我通过学习网上的资料总结出来在本地部署 Ollama+graphRAG 的教程，并且用《凡人修仙传》进行测试。

# windows 安装 Ollama
## 下载
访问[官网](https://ollama.com/download/windows)，我们能看到如下页面，我们选择 `Windows` 选项，然后点击` Download for Windows（Preview）` 下载安装包即可，这里要求 `Windows 10 或者之后`的系统版本。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/fa91bdc7d9754958ad6e8acb31f74dfc~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721891959&x-orig-sign=2q3G8QkVRs0wHboPaauPA0DuzHA%3D)



## 安装
然后打开安装包点击 `Install` 等待安装完成即可。


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/7ee6235bfe7c465f94c18ba3928379a6~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721891959&x-orig-sign=LB9pLkiUOEwhMsnBluOxUW4Mv%2BE%3D)

这里 Ollama 默认安装的位置在 `C:\Users\<用户>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs` 下面，
模型的默认位置在 `C:\Users\13900K\.ollama\models` ，如果 C 盘空间不足，可以通过新增系统环境变量 `OLLAMA_MODELS` 来指定位置，如下，`需要注意的是这个修改只能在重启电脑后生效`：


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/0527ecb102bf4103ab74d8915c61730c~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721891959&x-orig-sign=Cu6UpQ8w6%2B8Ily%2FJ9UV5Y0Loi4E%3D)


 ## 确认
 然后我们打开 power shell 之后，运行 ollma 命令，即可看到下面的提示出现，表示安装成功。
 
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/76f8f02195334899ac7ff1c518098974~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721891959&x-orig-sign=Ikt7IjhlI7B5uVLyIVjkzo91D0U%3D)


# Ollama 安装模型
我们可以在[官网模型库](https://ollama.com/library)中查找自己想用的模型，以及安装模型的命令，我这里直接安装 qwen2 7b 的模型，使用提供的命令，在 power shell 中执行即可。


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/103baf2741864c3abcec72d53a87954d~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721891959&x-orig-sign=dfScOgAhWGQibeZ0xebXg3c8fgo%3D)



成功安装完模型之后，`命令行提示词就会提示你输入文本可以进行交互了`。



 
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c2bf6eff723545829d993e8427b0f0c1~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721891959&x-orig-sign=Kd7Woh8oAHQb6H9IDwAuKCKV1jw%3D)


由于我们需要进行向量转换，所以需要安装 `embedding model` ，我这里使用的是 [quentinz/bge-large-zh-v1.5](https://ollama.com/quentinz/bge-large-zh-v1.5) ，直接在命令行使用下面的命令安装即可。

    ollama pull quentinz/bge-large-zh-v1.5
    
到目前为止，我安装了两个模型 `qwen` 和 `bge` ：

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/b8b4739d61f24f16ac22183dcb2412b1~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721891959&x-orig-sign=piZKSLixQqPWJN8LF3W%2Fsai%2BKV4%3D)




# 安装 graphrag

很简单的一个安装命令，只是要 `3.10 及以上的 python 版本`才能安装。

    pip install graphrag
    
然后创建一个目录 `D:\graphrag_data\input` 放入准备好的小说 `novel.txt` 。然后运行下面的命令初始化工作区：

    python -m graphrag.index --init --root D:\graphrag_data

初始化结束之后在 `D:\graphrag_data` 下会出现两个文件 `.env` and `settings.yaml`：

- `.env` ：包含运行 `GraphRAG pipeline` 所需的环境变量，能看到已定义的单个环境变量，GRAPHRAG_API_KEY=<API_KEY>。这是 OpenAI API 或 Azure OpenAI 端点的 API 密钥，我们可以将其替换为自己的 API 密钥。
- `settings.yaml` ：包含  `pipeline` 的设置，可以修改此文件以更改 `pipeline` 的设置。

# 构建索引

接下来需要修改我们的 `settings.yaml` 来构建我们的文件索引，我的如下，你们可以参考一下，注释掉的参数不需要管，只需要看放开注释的参数即可，最重要的就是`大语言模型`和`向量模型`的名称别写错了：

```

encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ollama
  type: openai_chat # or azure_openai_chat
  model: qwen2:latest
  model_supports_json: true # recommended if this is available for your model.
  max_tokens: 4000
  # request_timeout: 180.0
  api_base: http://localhost:11434/v1
  # api_version: 2024-02-15-preview
  # organization: <organization_id>
  # deployment_name: <azure_model_deployment_name>
  # tokens_per_minute: 150_000 # set a leaky bucket throttle
  # requests_per_minute: 10_000 # set a leaky bucket throttle
  # max_retries: 10
  # max_retry_wait: 10.0
  # sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
  concurrent_requests: 1 # the number of parallel inflight requests that may be made

parallelization:
  stagger: 0.3
  # num_threads: 50 # the number of threads to use for parallel processing

async_mode: threaded # or asyncio

embeddings:
  ## parallelization: override the global parallelization settings for embeddings
  async_mode: threaded # or asyncio
  llm:
    api_key: ollama
    type: openai_embedding # or azure_openai_embedding
    model: quentinz/bge-large-zh-v1.5:latest
    api_base: http://localhost:11434/api
    # api_version: 2024-02-15-preview
    # organization: <organization_id>
    # deployment_name: <azure_model_deployment_name>
    # tokens_per_minute: 150_000 # set a leaky bucket throttle
    # requests_per_minute: 10_000 # set a leaky bucket throttle
    # max_retries: 10
    # max_retry_wait: 10.0
    # sleep_on_rate_limit_recommendation: true # whether to sleep when azure suggests wait-times
    concurrent_requests: 1 # the number of parallel inflight requests that may be made
    # batch_size: 16 # the number of documents to send in a single request
    # batch_max_tokens: 8191 # the maximum number of tokens to send in a single request
    # target: required # or optional
  


chunks:
  size: 300
  overlap: 100
  group_by_columns: [id] # by default, we don't allow chunks to cross documents
    
input:
  type: file # or blob
  file_type: text # or csv
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"

cache:
  type: file # or blob
  base_dir: "cache"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

storage:
  type: file # or blob
  base_dir: "output/${timestamp}/artifacts"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

reporting:
  type: file # or console, blob
  base_dir: "output/${timestamp}/reports"
  # connection_string: <azure_blob_storage_connection_string>
  # container_name: <azure_blob_storage_container_name>

entity_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/entity_extraction.txt"
  entity_types: [organization,person,geo,event]
  max_gleanings: 0

summarize_descriptions:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  # enabled: true
  prompt: "prompts/claim_extraction.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 0

community_report:
  ## llm: override the global llm settings for this task
  ## parallelization: override the global parallelization settings for this task
  ## async_mode: override the global async_mode settings for this task
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

embed_graph:
  enabled: false # if true, will generate node2vec embeddings for nodes
  # num_walks: 10
  # walk_length: 40
  # window_size: 2
  # iterations: 3
  # random_seed: 597832

umap:
  enabled: false # if true, will generate UMAP embeddings for nodes

snapshots:
  graphml: false
  raw_entities: false
  top_level_nodes: false

local_search:
  # text_unit_prop: 0.5
  # community_prop: 0.1
  # conversation_history_max_turns: 5
  # top_k_mapped_entities: 10
  # top_k_relationships: 10
  max_tokens: 5000

global_search:
  max_tokens: 5000
  # data_max_tokens: 12000
  # map_max_tokens: 1000
  # reduce_max_tokens: 2000
  # concurrency: 32


```

保存好文件之后，执行下面的命令，开始构建文件的索引：

    python -m graphrag.index --root  D:\graphrag_data

然后我的 CPU 就开始了`冒烟模式` ，等待结束即可，因为我放的是`《凡人修仙传》前一百章`，所以肯定相当慢。

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/71509291e1a748db9ac125d1ded4d936~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=f64ab15b&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721891959&x-orig-sign=woVq1ExUmAgPuATgNwnFq1M6b3w%3D)

等了一个多小时还是没结束，果断关闭了进程，将文件中小说内容改成`前五章`，耗时了半个小时终于成功构建结束了。



# 修改 graphrag 源代码



 
因为我们没有使用 openai 的向量模型，而是自己的本地的 Ollama 平台中的向量模型，所以需要将源代码 `graphrag\llm\openai\openai_embeddings_llm.py` 中的代码进行调整如下，主要就是配置了一下自己使用的向量模型代码是 ：

    ollama.embeddings(model="quentinz/bge-large-zh-v1.5:latest",prompt=inp)
    
```

# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""

from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes
import ollama

class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        embedding_list = []
        for inp in input:
            embedding = ollama.embeddings(model="quentinz/bge-large-zh-v1.5:latest",prompt=inp)
            embedding_list.append(embedding["embedding"])
        return embedding_list 

```


另外需要修改在查询时候需要调用向量库的代码，位置在 `graphrag\query\llm\oai\embedding.py` ，大家可以参考我的代码：
```
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""OpenAI Embedding model implementation."""

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np
import tiktoken
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.base import OpenAILLMImpl
from graphrag.query.llm.oai.typing import (
    OPENAI_RETRY_ERROR_TYPES,
    OpenaiApiType,
)
from graphrag.query.llm.text_utils import chunk_text
from graphrag.query.progress import StatusReporter

from langchain_community.embeddings import OllamaEmbeddings

class OpenAIEmbedding(BaseTextEmbedding, OpenAILLMImpl):
    """Wrapper for OpenAI Embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        azure_ad_token_provider: Callable | None = None,
        model: str = "text-embedding-3-small",
        deployment_name: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,
        organization: str | None = None,
        encoding_name: str = "cl100k_base",
        max_tokens: int = 8191,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
        reporter: StatusReporter | None = None,
    ):
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,  # type: ignore
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )

        self.model = model
        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        self.retry_error_types = retry_error_types

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's sync function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        Please refer to: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
        """
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        chunk_embeddings = []
        chunk_lens = []
        for chunk in token_chunks:
            try:
                embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
                chunk_embeddings.append(embedding)
                chunk_lens.append(chunk_len)
            # TODO: catch a more specific exception
            except Exception as e:  # noqa BLE001
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: str(e)},
                )

                continue
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's async function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        """
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        chunk_embeddings = []
        chunk_lens = []
        embedding_results = await asyncio.gather(*[
            self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
        ])
        embedding_results = [result for result in embedding_results if result[0]]
        chunk_embeddings = [result[0] for result in embedding_results]
        chunk_lens = [result[1] for result in embedding_results]
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)  # type: ignore
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    def _embed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    embedding = (
                        OllamaEmbeddings(
                            model=self.model,
                        ).embed_query(text)
                        or []
                    )
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return ([], 0)
        else:
            # TODO: why not just throw in this case?
            return ([], 0)

    async def _aembed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    embedding = await (OllamaEmbeddings( model=self.model, ).embed_query(text) or [])
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return ([], 0)
        else:
            # TODO: why not just throw in this case?
            return ([], 0)

```


到这里基本的配置就完成了，下面进行使用阶段。

# 使用效果
通过上面你修改代码我们应该知道，想要运行后面的代码要确保 pip 已经安装了 `ollama` 和 `langchain_community` 两个库才可以。


graphrag 提供了两种查询方式，可以适用于提问涉及全文内容的问题和局部内容的问题两种情况，[详见官方介绍](https://microsoft.github.io/graphrag/posts/query/overview/):
-   `Global Search`：通过以 map-reduce 方式搜索所有 AI 生成的报告来生成答案。这是一种资源密集型方法，但通常可以回答需要了解整个数据集的问题。如“这个小说讲述了什么”。
-   `Local Search`：本地搜索方法通过将 AI 提取的知识图谱中的相关数据与原始文档的文本块相结合来生成答案。此方法适用于需要了解文档中提到的特定实体的问题,如韩立如何进入七玄门？。
  


```
python -m graphrag.query  --root D:\graphrag_data  --method global  "这个小说讲述了什么"
```


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/086c6d9120994778a20a9f6ae62f0ee1~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721374431&x-orig-sign=ok%2FGRXdW253nl%2FN7qjOVj89t5Ic%3D)

```
python -m graphrag.query  --root D:\graphrag_data  --method local  "韩立如何进入七玄门"
```

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/99f4751febcb47049b17126808e622aa~tplv-73owjymdk6-watermark.image?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1721374525&x-orig-sign=lpuIHS0ujEk%2FdE974iVx6dBOfm4%3D)

我个人感觉还是效果有点差强人意，因为我看过小说，知道答案不是这样子的，不过总的来说我们已经在本地跑起来了！

# 总结

使用过后有一些想法：

- 比较吃硬件性能，我的机器 CPU 是i9-13900K 的型号，结果针对小说前五章构建图谱和索引的过程竟然要半个小时才结束，是不能接受的。我看网上有人用 openai 的接口将小说跑完花了 10 美金，真的是成本太高了，我估计这也是微软将这个技术开源出来的原因，希望社区的力量来优化性能。
- 从使用便捷性来说确实很方便，如果使用 openai 的接口其实只需要安装 graphrag 的库即可，我们操作麻烦是因为想通过 Ollama 调用本地的大模型和向量模型，从而要改配置和源代码。
- 从效果来看不如预期，我觉得可能是大模型和向量模型的效果有待提升，或者是 rag 的配置需要优化。
 