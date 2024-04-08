# 读取文件
我这里使用的是随便找的一份 csv 文件，只是为了示范，没有什么实际的用处。需要注意的是内容中不要有特殊字符或者🤫之类的表情包，否则在处理的时候会报错，内容如下：

<p align=center><img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/8bed973eedb64cfd80caa65664192150~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1602&h=635&s=183858&e=png&b=fdfdfd" alt="image.png"  /></p>

# 基本的文档处理参数如下：

```
chunk_overlap = 50
chunk_size = 250
embed_model = 'bge-large-zh-v1.5'
vs_type = 'fassi'
zh_title_enhance = False
```

详细解释如下：

1.  `chunk_overlap = 50`: `chunk_overlap` 是指在进行文本分块时，每个块之间的重叠量。在处理文本时，通常将文本分成多个块以便更有效地处理，而重叠量可以确保在相邻的块之间不会丢失重要的信息。在这个例子中，重叠量为 50，表示相邻块之间会有 50 个字符的重叠。
1.  `chunk_size = 250`： `chunk_size` 是指每个文本块的大小。将长文本分成适当大小的块有助于更高效地处理文本数据。在这里每个文本块的大小为 250 个字符。
1.  `embed_model = 'bge-large-zh-v1.5'`： `embed_model` 是指用于文本嵌入（embedding）的模型。文本嵌入是将文本数据转换成向量的过程，通常用于表示文本数据。在这里，使用了名为 `'bge-large-zh-v1.5'` 的嵌入模型。
1.  `vs_type = 'fassi'`： `vs_type` 是向量数据库名称。
1.  `zh_title_enhance = False`： `zh_title_enhance` 是一个布尔值，用于指示是否要增强中文标题。当设置为 `True` 时，表示对中文标题进行增强处理；当设置为 `False` 时，表示不进行增强处理。

# 加载自定义的 Loader 处理 csv 文件


我们这里使用的是 `<class 'langchain.document_loaders.csv_loader.CSVLoader'>` 来处理 csv 文件内容，详细代码如下，将每一行的内容封装成给你一个 `Document 类` ，然后将所有行对应的 Document 添加到一个列表中即可完成对 csv 文件的内容处理，具体  `Document 类` 介绍如下：

```
Document(page_content=content, metadata=metadata)
```
- `page_content` 就是每一行的内容，其实就是将当前行的列名和内容使用 ":" 进行拼接，然后将所有的列的内容用"\n"拼接而成的字符串。
- `metadata` 记录了当前所在行以及 csv 文件的路径。

我这里以前两行为例列举内容如下：


     [
     Document(
         page_content=': 0\ntitle: 加油~以及一些建议\nfile: 2023-03-31.0002\nurl: https://github.com/imClumsyPanda/langchain-ChatGLM/issues/2\ndetail: 加油，我认为你的方向是对的。\nid: 0', 
         metadata={'source': 'D:\\Langchain-Chatchat-torch2-240402\\knowledge_base\\samples\\content\\test_files/langchain-ChatGLM_closed.csv', 'row': 0}
     ), 
     Document(
         page_content=': 1\ntitle: 当前的运行环境是什么，windows还是Linux\nfile: 2023-04-01.0003\nurl: https://github.com/imClumsyPanda/langchain-ChatGLM/issues/3\ndetail: 当前的运行环境是什么，windows还是Linux，python是什么版本？\nid: 1', 
         metadata={'source': 'D:\\Langchain-Chatchat-torch2-240402\\knowledge_base\\samples\\content\\test_files/langchain-ChatGLM_closed.csv', 'row': 1}
     )
     ]

```
def __read_file(self, csvfile: TextIOWrapper) -> List[Document]:
    docs = []

    csv_reader = csv.DictReader(csvfile, **self.csv_args)  # type: ignore
    for i, row in enumerate(csv_reader):
        try:
            source = (
                row[self.source_column]
                if self.source_column is not None
                else self.file_path
            )
        except KeyError:
            raise ValueError(
                f"Source column '{self.source_column}' not found in CSV file."
            )
        content = "\n".join(
            f"{k.strip()}: {v.strip()}"
            for k, v in row.items()
            if k not in self.metadata_columns
        )
        metadata = {"source": source, "row": i}
        for col in self.metadata_columns:
            try:
                metadata[col] = row[col]
            except KeyError:
                raise ValueError(f"Metadata column '{col}' not found in CSV file.")
        doc = Document(page_content=content, metadata=metadata)
        docs.append(doc)

    return docs
```
# 向量化

选择合适的向量化模型和向量化数据库，将得到的 docs 列表转为向量存入数据库中即可，后续即可完成基于 csv 的文档问答任务。



# 参考
- https://github.com/chatchat-space/Langchain-Chatchat


