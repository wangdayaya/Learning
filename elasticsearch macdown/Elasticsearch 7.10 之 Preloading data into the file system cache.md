## Elasticsearch 7.10 之 Preloading data into the file system cache

NOTE：这是一个专家设置，其详细信息将来可能会更改

默认情况下，Elasticsearch 完全依靠操作系统文件系统缓存来缓存 I/O 操作。可以设置 **index.store.preload** ，以告知操作系统在打开时将热索引文件的内容加载到内存中。此设置接受以逗号分隔的文件扩展名列表：扩展名在列表中的所有文件将在打开时预加载。这对提高索引的搜索性能很有用，尤其是在重新启动主机操作系统时，因为这会导致文件系统缓存被破坏。但是请注意，这可能会减慢索引的打开速度，因为只有将数据加载到物理内存中后，索引才能变得可用。

此设置仅是尽力而为的，可能根本不起作用，具体取决于存储类型和主机操作系统。

**index.store.preload** 是一个静态设置，可以在 **config/elasticsearch.yml** 中进行设置：

	index.store.preload: ["nvd", "dvd"]
或在创建索引时在索引设置中：

	PUT /my-index-000001
	{
	  "settings": {
	    "index.store.preload": ["nvd", "dvd"]
	  }
	}
 
默认值为空数组，这意味着将不会将任何内容快速加载到文件系统缓存中。对于主动搜索的索引，您可能需要将其设置为 ["nvd", "dvd"] ，这将导致规范和 doc 值被急切地加载到物理内存中。这是要查看的两个首要扩展，因为 Elasticsearch 对它们执行随机访问。

可以使用通配符来指示应预加载所有文件：index.store.preload: ["*"] 。但是请注意，将所有文件（尤其是存储的字段和术语向量的文件）加载到内存中通常没有什么用，因此更好的选择可能是将其设置为 **["nvd", "dvd", "tim", "doc", "dim"]** ，它将预先加载规范，doc 值，术语词典，发布列表和要点，它们是搜索和汇总索引中最重要的部分。

请注意，此设置在大于主机主内存大小的索引上可能很危险，因为它会导致大型合并后重新打开文件系统缓存时将其浪费掉，这会使索引和搜索变慢。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/preload-data-to-file-system-cache.html
