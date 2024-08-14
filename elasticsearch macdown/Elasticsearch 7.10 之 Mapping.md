## Elasticsearch 7.10 之 Mapping

映射是定义文档及其包含的字段的存储和索引方式的过程。例如，使用映射定义：

* 哪些字符串字段应视为全文本字段
* 哪些字段包含数字，日期或地理位置
* 日期值的格式
* 自定义规则来控制动态添加字段的映射

映射定义具有：

**Metadata fields** ：元数据字段用于自定义如何处理文档的相关元数据。元数据字段的示例包括文档的 \_index ，\_id 和 \_source 字段。

**Fields** ：映射包含与文档相关的字段或属性的列表。每个字段都有其自己的数据类型。


### Settings to prevent mappings explosion

在索引中定义太多字段会导致映射爆炸，这可能会导致内存不足错误和难以恢复的情况。

考虑一种情况，其中每个插入的新文档都引入了新字段，例如动态映射。每个新字段都添加到索引映射中，随着映射的增长，这可能会成为问题。

使用以下设置来限制（手动或动态创建的）字段映射的数量，并防止文档引起映射爆炸：

**index.mapping.total_fields.limit** :索引中的最大字段数。字段和对象的映射以及字段别名都计入此限制。默认值为 1000 。

该限制以防止映射和搜索变得太大。较高的值可能导致性能下降和内存问题，尤其是在负载较高或资源很少的群集中。

如果增加此设置，建议您也增加 **index.query.bool.max_clause_count** 设置，该设置限制查询中布尔型子句的最大数量。

TIP: 如果字段映射包含大量任意键，请考虑使用扁平化的数据类型。

**index.mapping.depth.limit** :字段的最大深度，以内部对象的数量衡量。例如，如果所有字段都在根对象级别定义，则深度为 1 。如果有一个对象映射，则深度为 2 ，依此类推。默认值为 20 。

**index.mapping.nested_fields.limit** :索引中最大的不同嵌套映射数。嵌套类型仅在特殊情况下才需要使用，当需要相互独立地查询对象数组时。为了防止设计不良的映射，此设置限制了每个索引的唯一嵌套类型的数量。默认值为 50 。

**index.mapping.nested_objects.limit** :一个文档可以在所有嵌套类型中包含的嵌套 JSON 对象的最大数量。当文档包含太多嵌套对象时，此限制有助于防止出现内存不足错误。默认值为 10000 。

**index.mapping.field_name_length.limit** :设置字段名称的最大长度。此设置实际上不能解决映射爆炸问题，但是如果您想限制字段长度，该设置可能仍然有用。通常无需设置此设置。除非用户开始添加名称非常长的大量字段，否则默认设置是可以的。默认值为 Long.MAX_VALUE（无限制）。

### Dynamic mapping


字段和映射类型在使用之前不需要定义。通过动态映射，仅通过索引文档即可自动添加新的字段名称。新字段既可以添加到顶级映射类型，也可以添加到内部对象和嵌套字段。

可以将动态映射规则配置为自定义用于新字段的映射。

### Explicit mappings

您对数据的了解超出了 Elasticsearch 的猜测，因此尽管动态映射对于入门非常有用，但有时您仍需要指定自己的显式映射。

当你在 [create an index](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html#create-mapping) 和 [add fields to an existing index](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html#add-field-mapping) 时创建字段映射。

### Create an index with an explicit mapping

您可以使用 create index API 创建带有显式映射的新索引。

	PUT /my-index-000001
	{
	  "mappings": {
	    "properties": {
	      "age":    { "type": "integer" },    # 创建 age ，一个整数字段
	      "email":  { "type": "keyword"  },   # 创建email ，一个关键字字段
	      "name":   { "type": "text"  }   # 创建 name ，一个文本字段    
	    }
	  }
	}
 
### Add a field to an existing mapping

您可以使用 [put mapping API](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-put-mapping.html) 将一个或多个新字段添加到现有索引。

下面的示例添加 employee-id ，这是一个关键字字段，其索引映射参数值为 **false** 。这意味着已存储 employee-id 字段的值，但不会索引或用于搜索。

	PUT /my-index-000001/_mapping
	{
	  "properties": {
	    "employee-id": {
	      "type": "keyword",
	      "index": false
	    }
	  }
	}
 
### Update the mapping of a field

除了支持的映射参数外，您无法更改现有字段的映射或字段类型。更改现有字段可能会使已经建立索引的数据无效。

如果您需要更改数据流的后备索引中字段的映射，请参阅 [Change mappings and settings for a data stream](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-streams-change-mappings-and-settings.html) 。

如果您需要更改其他索引中字段的映射，请使用正确的映射创建一个新索引，然后将数据 [reindex](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-reindex.html) 到该索引中。

重命名字段会使在旧字段名称下已建立索引的数据无效。添加一个 [alias](https://www.elastic.co/guide/en/elasticsearch/reference/current/alias.html) 字段以创建备用字段名称。

### View the mapping of an index

您可以使用 [get mapping](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-get-mapping.html) API 查看现有索引的映射。

	GET /my-index-000001/_mapping
 
API 返回以下响应：

	{
	  "my-index-000001" : {
	    "mappings" : {
	      "properties" : {
	        "age" : {
	          "type" : "integer"
	        },
	        "email" : {
	          "type" : "keyword"
	        },
	        "employee-id" : {
	          "type" : "keyword",
	          "index" : false
	        },
	        "name" : {
	          "type" : "text"
	        }
	      }
	    }
	  }
	}
### View the mapping of specific fields

如果您只想查看一个或多个特定字段的映射，则可以使用 [get field mapping API](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-get-field-mapping.html) 。

如果您不需要索引的完整映射或索引包含大量字段，这将很有用。

以下请求检索 employee-id 字段的映射。

	GET /my-index-000001/_mapping/field/employee-id
 
API 返回以下响应：

	{
	  "my-index-000001" : {
	    "mappings" : {
	      "employee-id" : {
	        "full_name" : "employee-id",
	        "mapping" : {
	          "employee-id" : {
	            "type" : "keyword",
	            "index" : false
	          }
	        }
	      }
	    }
	  }
	}

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping.html
