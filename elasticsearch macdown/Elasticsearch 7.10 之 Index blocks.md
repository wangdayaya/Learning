## Elasticsearch 7.10 之 Index blocks

索引块限制了特定索引上可用的操作类型。这些块有不同的样式，可以阻止写，读或元数据操作。可以使用动态索引设置来设置/删除这些块，或者可以使用专用的 API 来添加这些块，这还可以确保写入块，一旦成功返回给用户，则索引的所有分片都可以正确地拥有该块，例如添加写块后，所有对索引的动态写入已完成。

### Index block settings


以下动态索引设置决定索引上存在的块：

**index.blocks.read_only**: 设置为 true 以使索引和索引元数据为只读，设置为 false 以允许写入和元数据更改

**index.blocks.read\_only\_allow_delete**: 
与 **index.blocks.read_only** 类似，但也允许删除索引以使释放资源。基于磁盘的分片分配器可以自动添加和删除此块

从索引删除文档以释放资源-而不是删除索引本身-会随着时间的推移增加索引的大小。当 **index.blocks.read\_only\_allow_delete** 设置为 **true** 时，不允许删除文档。但是，删除索引本身会释放只读索引块，并使资源几乎立即可用

IMPORTANT: 当磁盘利用率降至高水位线以下时，Elasticsearch 自动添加和删除只读索引块，该值由 **cluster.routing.allocation.disk.watermark.flood_stage** 控制

**index.blocks.read**: 设置为 **true** 以禁用对索引的读取操作

**index.blocks.write**: 设置为 **true** 以禁用对索引的数据写操作。与 **read_only** 不同，此设置不会影响元数据。例如，您可以使用 **write** 块关闭索引，但不能使用 **read_only** 块关闭索引
 
**index.blocks.metadata**: 设置为 **true** 以禁用索引元数据读取和写入

### Add index block API

将索引块添加到索引。

	PUT /my-index-000001/_block/write

**Request**

	PUT /<index>/_block/<block>

**Path parameters**

\<index>

（可选，字符串）索引名称的逗号分隔列表或通配符表达式，用于限制请求。

要将块添加到所有索引，请使用 **_all** 或 **\*** 。要禁止使用 **_all** 或通配符表达式向索引添加块，请将 **action.destructive\_requires_name** 群集设置更改为 **true** 。您可以在 **elasticsearch.yml** 文件中或使用 **cluster update settings API** 更新此设置。

\<block>

（必需，字符串）要添加到索引的块类型。**\<block>** 的有效值:

*  **metadata** :禁用元数据更改，例如关闭索引
*  **read** :禁用读取操作
*  **read_only** :禁用写操作和元数据更改
*  **write** :禁用写操作。但是仍然允许元数据更改。

**Query parameters**

 allow\_no_indices
 
（可选，布尔值）如果为 **false** ，则如果任何通配符表达式，索引别名或 **_all** 值仅针对缺失或关闭的索引为目标，则请求将返回错误。即使该请求针对其他打开的索引，此行为也适用。例如，如果索引以 **foo** 开头但没有索引以 **bar** 开头，则针对 **foo*** ，**bar*** 的请求将返回错误。默认为 **true** 。

 expand_wildcards
 
（可选，字符串）控制通配符表达式可以扩展到的索引类型。多个值在用逗号分隔时（如 **open,hidden** ）被接受。默认为 **open**。有效值为：

*  all :扩大到打开的和关闭的索引，包括隐藏索引。
*  open :仅扩展到打开的索引。
*  closed :仅扩展到关闭的索引。
*  hidden :通配符的扩展将包括隐藏索引。必须与 **open** ，**closed** 或两者结合使用。
*  none :不接受通配符表达式。


ignore\_unavailable

（可选，布尔值）如果为 **true** ，则响应中不包括丢失或闭合的索引。默认为 **false** 。

master_timeout

（可选，时间单位）指定等待连接到主节点的时间。如果在超时到期之前未收到任何响应，则请求将失败并返回错误。默认为 **30s** 。
 
timeout

（可选，时间单位）指定等待响应的时间。如果在超时到期之前未收到任何响应，则请求将失败并返回错误。默认为 **30s** 。

**Examples**

以下示例显示如何添加索引块：

	PUT /my-index-000001/_block/write
 
API返回以下响应：

	{
	  "acknowledged" : true,
	  "shards_acknowledged" : true,
	  "indices" : [ {
	    "name" : "my-index-000001",
	    "blocked" : true
	  } ]
	}

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-blocks.html
