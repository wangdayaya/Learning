# 读取文件

使用的 docx 文档是一个 `示例.docx` 文档，内容截图如下：

 
 
<p align=center><img src="https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/1d517f0fb7e74a459e946bcb5cc03380~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=545&h=760&s=140094&e=png&b=fefefe" alt="image.png"  /></p>

 

# 参数说明
基本的文档处理参数如下：

    chunk_overlap = 50
    chunk_size = 250
    embed_model = 'm3e-large'
    vs_type = 'fassi'
    zh_title_enhance = False
 
 详细解释如下：
 
 1.  `chunk_overlap = 50`: `chunk_overlap` 是指在进行文本分块时，每个块之间的重叠量。在处理文本时，通常将文本分成多个块以便更有效地处理，而重叠量可以确保在相邻的块之间不会丢失重要的信息。在这个例子中，重叠量为 50，表示相邻块之间会有 50 个字符的重叠。

1.  `chunk_size = 250`： `chunk_size` 是指每个文本块的大小。将长文本分成适当大小的块有助于更高效地处理文本数据。在这里每个文本块的大小为 250 个字符。

1.  `embed_model = 'm3e-large`： `embed_model` 是指用于文本嵌入（embedding）的模型。文本嵌入是将文本数据转换成向量的过程，通常用于表示文本数据。在这里，使用了名为 `m3e-large` 的嵌入模型。

1.  `vs_type = 'fassi'`： `vs_type` 是向量数据库名称。

1.  `zh_title_enhance = False`：  `zh_title_enhance` 是一个布尔值，用于指示是否要增强中文标题。当设置为 `True` 时，表示对中文标题进行增强处理；当设置为 `False` 时，表示不进行增强处理。
 
# 加载自定义的 Loader 处理 pdf 文件

这里我使用的是自定义的 `document_loaders.mydocloader.RapidOCRDocLoader` ，处理过程的核心代码如下：

```
def _get_elements(self) -> List:
    def doc2text(filepath):
        from docx.table import _Cell, Table
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.text.paragraph import Paragraph
        from docx import Document, ImagePart
        from PIL import Image
        from io import BytesIO
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        doc = Document(filepath)
        resp = ""

        def iter_block_items(parent):
            from docx.document import Document
            if isinstance(parent, Document):
                parent_elm = parent.element.body
            elif isinstance(parent, _Cell):
                parent_elm = parent._tc
            else:
                raise ValueError("RapidOCRDocLoader parse fail")

            for child in parent_elm.iterchildren():
                if isinstance(child, CT_P):
                    yield Paragraph(child, parent)
                elif isinstance(child, CT_Tbl):
                    yield Table(child, parent)

        b_unit = tqdm.tqdm(total=len(doc.paragraphs)+len(doc.tables),
                           desc="RapidOCRDocLoader block index: 0")
        for i, block in enumerate(iter_block_items(doc)):
            b_unit.set_description(
                "RapidOCRDocLoader  block index: {}".format(i))
            b_unit.refresh()
            if isinstance(block, Paragraph):
                resp += block.text.strip() + "\n"
                images = block._element.xpath('.//pic:pic')  # 获取所有图片
                for image in images:
                    for img_id in image.xpath('.//a:blip/@r:embed'):  # 获取图片id
                        part = doc.part.related_parts[img_id]  # 根据图片id获取对应的图片
                        if isinstance(part, ImagePart):
                            image = Image.open(BytesIO(part._blob))
                            result, _ = ocr(np.array(image))
                            if result:
                                ocr_result = [line[1] for line in result]
                                resp += "\n".join(ocr_result)
            elif isinstance(block, Table):
                for row in block.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            resp += paragraph.text.strip() + "\n"
            b_unit.update(1)
        return resp

    text = doc2text(self.file_path)
    from unstructured.partition.text import partition_text
    return partition_text(text=text, **self.unstructured_kwargs)
```

这里使用了一个叫 ```Document``` 的 python 库可以直接提取 docx 文件中的信息，Document 专门用于处理 Microsoft Word 文档。我们这里主要处理的只有两种类型的内容，分别对应 ```paragraphs``` 和 ```tables``` ，处理逻辑如下：

- paragraphs ： 直接将文本提取出来拼接到 resp 后面，如果有图片，则会使用 ocr 技术提取图片中的文字同样拼接到 resp 后面
- tables：将表格中的每一行文本，从左到右使用换行符 "\n" ，将每一列的数据拼接起来，如下图所示表格，最后拼接的字符串结果如下所示。

<p align=center><img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/327a396ae9b3427b8bb86c399ab5df87~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=562&h=220&s=19336&e=png&b=ffffff" alt="image.png"  /></p>

```
优点
缺点
GEOcoding & CSV export：类似于知识库问答，因为需要返回准确的经纬度
只能查询
Administrative layers&export to QGIS: 与app.ageospatial.com进行数据访问
无法对结果进行操作交互
Population data：与app.ageospatial.com进行数据访问人口数据分布
输入数据格式有限
Sentinel-2 imagery and NDVl(Normalized Difference Vegetation Index) ：与app.ageospatial.com进行数据访问卫星影像
依赖于自己的数据，因为都是专业涉密数据，准确性也高
Building data&export to QGIS
```

最终将所有```paragraphs``` 和 ```tables``` 中的字符串都拼接起来形成一个长字符串，最后使用一个 ```partition_text``` 函数进行了一定的切分，将得到的字符串列表返回即可，其实这一步感觉没啥用处，因为后边其实还使用了`ChineseRecursiveTextSplitter` 来对长文本进行了递归拆分。




# 封装

将得到文本进行拆分之后，以方便后续的内容向量化，将上面的结果包装成一个包含了许多 Document 列表，，这些 Document 有利于后续向量化入库，每个 Document 中有 `pagecontent` 和 `metadata` ，前者存放部分文本内容，后者存放该内容的元数据，比如文件位置等等，部分内容展示如下图。



![image.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/34d88d02391f43cdb436901bb680f198~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1740&h=506&s=175754&e=png&b=3c3f41)
 




# 存入向量库
随便找一个可以使用的向量模型，我这里使用的是 `m3e-large` ，另外还有找自己合适的向量数据库，我这里使用的是 `fassi` ，将上面处理好的内容都经过向量化存入 fassi 中，后面结合大模型即可即可进行文档的问答和检索。这里展示了使用我这个文档进行的问答过程。


 
<p align=center><img src="https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/27362af2cbd743f384550a34addffd4d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1401&h=871&s=143632&e=png&b=fdfdfd" alt="image.png"  /></p>



# 参考
- https://github.com/chatchat-space/Langchain-Chatchat