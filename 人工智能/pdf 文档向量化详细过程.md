# 读取文件

使用的 pdf 文档是一个 `地址树模型的中文地址提取方法.pdf` 文档，内容截图如下：

 
 

<p align=center><img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a96deaf6edc04138abe7f4575fe03479~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=478&h=732&s=219063&e=png&b=fefcfc" alt="image.png"  /></p>


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

因为我使用的是一份 pdf 文件，所以可以根据 .pdf 后缀可以选择使用自定义的 `RapidOCRPDFLoader 类`来作为我们的 Loader ，日志打印如下：

    RapidOCRPDFLoader used for D:\Langchain-Chatchat-torch2-240402\knowledge_base\samples\content\test_files/DRL.pdf


下面的代码看起来很长，其实关键的逻辑就是在 pdf2text 函数中，先通过 `doc = fitz.open(filepath)` 提取出每一页 `pdf` 的信息 `doc` 列表 ，然后遍历  `doc` 列表中每一个 的 `page` 信息。这里的 `page` 里面包含了可能出现的`文本`和`图片`，如果有文本就把 `ocr` 识别的这一页 pdf 文本长字符串加入到  `resp` 中，如果有图片就通过可能的旋转和 ocr 等操作，将识别出来的图片中的文本加入到 `resp` 中，最后将 `resp` 返回即可。我们此时拿到的是一个非常长的字符串，里面包含了全部 pdf 的文字信息，这种很长的内容不容易进行向量化，所以要切分一下，这里我们用到了一个现成的函数 `partition_text` ，其实就是根据内置的参数算法，将长文本都切分成了较短的字符串列表。 

```
class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            '''
            img   --image
            angle --rotation angle
            return--rotated img
            '''
            
            h, w = img.shape[:2]
            rotate_center = (w/2, h/2)
            #获取旋转矩阵
            # 参数1为旋转中心点;
            # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
            # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            #计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            #调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img
        
        def pdf2text(filepath):
            import fitz # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np
            ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for i, page in enumerate(doc):
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                b_unit.refresh()
                text = page.get_text("")
                resp += text + "\n"

                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    if xref := img.get("xref"):
                        bbox = img["bbox"]
                        # 检查图片尺寸是否超过设定的阈值
                        if ((bbox[2] - bbox[0]) / (page.rect.width) < PDF_OCR_THRESHOLD[0]
                            or (bbox[3] - bbox[1]) / (page.rect.height) < PDF_OCR_THRESHOLD[1]):
                            continue
                        pix = fitz.Pixmap(doc, xref)
                        samples = pix.samples
                        if int(page.rotation)!=0:  #如果Page有旋转角度，则旋转图片
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                            tmp_img = Image.fromarray(img_array);
                            ori_img = cv2.cvtColor(np.array(tmp_img),cv2.COLOR_RGB2BGR)
                            rot_img = rotate_img(img=ori_img, angle=360-page.rotation)
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)

                        result, _ = ocr(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)
```
# 封装

将得到文本之后还要进行拆分，以方便后续的内容向量化，将上面的结果进行处理后的结果如下图所示，得到的是一个包含了许多 Document 列表，这些 Document 有利于后续向量化入库，每个 Document 中有 `pagecontent` 和 `metadata` ，前者存放部分文本内容，后者存放该内容的元数据，比如文件位置等等。


<p align=center><img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/316f6139207249b88fec4fa4be2a8807~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1745&h=467&s=99601&e=png&b=4b4d4d" alt="2dfada1ee5d34588e0d527b528693d5.png"  /></p>




# 存入向量库
随便找一个可以使用的向量模型，我这里使用的是 `m3e-large` ，另外还有找自己合适的向量数据库，我这里使用的是 `fassi` ，将上面处理好的内容都经过向量化存入 fassi 中，后面结合大模型即可即可进行文档的问答和检索。这里展示了使用我这个文档进行的问答过程。


![image.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/a68bf65350024f16bad03269e2b6c6bf~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=1421&h=885&s=136685&e=png&b=fdfdfd)



# 参考
- https://github.com/chatchat-space/Langchain-Chatchat