## Elasticsearch 7.10 之 Retrieve selected fields from a search

默认情况下，搜索响应中的每个命中结果都包含文档 [_source](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/mapping-source-field.html) ，这是在为文档建立索引时提供的整个 JSON 对象。要检索搜索响应中的特定字段，可以使用 fields 参数：

	POST my-index-000001/_search
	{
	  "query": {
	    "match": {
	      "message": "foo"
	    }
	  },
	  "fields": ["user.id", "@timestamp"],
	  "_source": false
	}
 
fields 参数同时查阅文档的 \_source 和索引映射以加载和返回值。因为它利用了映射，所以与直接引用 \_source 相比，字段具有一些优点：它接受 [multi-fields](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/multi-fields.html) 和  [field aliases](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/alias.html) ，并且还以一致的方式设置诸如日期之类的字段值的格式。

文档的 \_source 存储在 Lucene 中的单个字段中。因此，即使仅请求少量字段，也必须加载并解析整个 \_source 对象。为避免此限制，您可以尝试另一种加载字段的方法：

* 使用 [docvalue_fields](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-fields.html#docvalue-fields) 参数获取选定字段的值。当返回相当少的支持 doc 值的字段（例如关键字和日期）时，这是一个不错的选择。
* 使用 [stored_fields](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-request-body.html#request-body-search-stored-fields) 参数可获取特定存储字段（使用 [store](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/mapping-store.html) 映射选项的字段）的值。

如果需要，可以使用 [script_field](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-fields.html#script-fields) 参数通过脚本转换响应中的字段值。但是，脚本无法利用 Elasticsearch 的索引结构或相关优化。有时这可能会导致搜索速度降低。

您可以在以下各节中找到有关每种方法的更多详细信息：

* Fields
* Doc value fields
* Stored fields
* Source filtering
* Script fields

### Fields

WARNING: 此功能处于 beta 版本，可能会更改。该设计和代码不如正式的 GA 功能成熟，并且按原样提供，不提供任何担保。 Beta 功能不受官方 GA 功能的支持 SLA 约束。

fields 参数允许检索搜索响应中的文档字段列表。它同时查阅文档 _source 和索引映射，以符合其映射类型的标准化方式返回每个值。默认情况下，日期字段是根据其映射中的日期格式参数设置格式的。

以下搜索请求使用 fields 参数检索 user.id 字段，以 http.response. 开头的所有字段以及 @timestamp 字段的值：

	POST my-index-000001/_search
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  },
	  "fields": [
	    "user.id",
	    "http.response.*",    # 完整字段名和通配符模式都可以接受
	    {
	      "field": "@timestamp",
	      "format": "epoch_millis"   # 使用对象符号，您可以传递格式参数以对字段的值应用自定义格式。日期字段 date 和 date_nanos 接受日期格式。空间字段接受 geoJSON 作为 GeoJSON（默认值），或者接受 wkt 作为熟知文本。其他字段类型不支持 format 参数
	    }
	  ],
	  "_source": false
	}
 

在每次匹配中， fields 部分中的值作为平面列表返回：

	{
	  "took" : 2,
	  "timed_out" : false,
	  "_shards" : {
	    "total" : 1,
	    "successful" : 1,
	    "skipped" : 0,
	    "failed" : 0
	  },
	  "hits" : {
	    "total" : {
	      "value" : 1,
	      "relation" : "eq"
	    },
	    "max_score" : 1.0,
	    "hits" : [
	      {
	        "_index" : "my-index-000001",
	        "_id" : "0",
	        "_score" : 1.0,
	        "_type" : "_doc",
	        "fields" : {
	          "user.id" : [
	            "kimchy"
	          ],
	          "@timestamp" : [
	            "4098435132000"
	          ],
	          "http.response.bytes": [
	            1070000
	          ],
	          "http.response.status_code": [
	            200
	          ]
	        }
	      }
	    ]
	  }
	}

仅返回叶子字段，fields 不允许提取整个对象。

fields 参数用于处理字段类型，例如 [field aliases](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/alias.html) 和 [constant_keyword](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/keyword.html#constant-keyword-field-type)，其值并不总是出现在 _source 中。还考虑其他映射选项，包括 [ignore_above](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/ignore-above.html) ，[ignore_malformed](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/ignore-malformed.html) 和 [null_value](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/null-value.html)。

NOTE：即使 _source 中只有一个值，fields 响应也总是为每个字段返回一个值数组。这是因为 Elasticsearch 没有专用的数组类型，并且任何字段都可以包含多个值。 fields 参数也不能保证以特定顺序返回数组值。有关更多背景，请参见 [arrays](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/array.html) 映射文档。

### Doc value fields

您可以使用 docvalue_fields 参数返回搜索响应中一个或多个字段的 doc values 。

doc values 存储与 \_source 相同的值，但在磁盘上基于列的结构中进行了优化，以进行排序和汇总。由于每个字段都是单独存储的，因此 Elasticsearch 仅读取请求的字段值，并且可以避免加载整个文档 \_source。

默认情况下，将为支持的字段存储文档值。但是，text 或 text_annotated 字段不支持 doc values 。

以下搜索请求使用 docvalue_fields 参数检索 user.id 字段，以 http.response. 开头的所有字段以及 @timestamp 字段的 doc values ：

	GET my-index-000001/_search
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  },
	  "docvalue_fields": [
	    "user.id",
	    "http.response.*",   # 完整字段名和通配符模式都可以接受
	    {
	      "field": "date",
	      "format": "epoch_millis"   # 使用对象符号，您可以传递格式参数，以将自定义格式应用于该字段的 doc values。日期字段支持日期格式。数值字段支持 DecimalFormat 模式。其他字段数据类型不支持 format 参数
	    }
	  ]
	}
 

TIP: 您不能使用 docvalue_fields 参数检索嵌套对象的 doc values 。如果指定嵌套对象，则搜索将为该字段返回一个空数组（[]）。要访问嵌套字段，请使用 inner_hits 参数的 docvalue_fields 属性。

### Stored fields

也可以使用 [store](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/mapping-store.html) 映射选项来存储单个字段的值。您可以使用 stored_fields 参数将这些存储的值包括在搜索响应中。

WARNING: stored_fields 参数用于显式标记为存储在映射中的字段，默认情况下处于关闭状态，通常不建议这样做。而是使用源过滤来选择要返回的原始源文档的子集。

允许有选择地为搜索命中表示的每个文档加载特定的存储字段。

	GET /_search
	{
	  "stored_fields" : ["user", "postDate"],
	  "query" : {
	    "term" : { "user" : "kimchy" }
	  }
	}
 
\* 可用于从文档加载所有存储的字段。

空数组将导致只返回每个匹配的 \_id 和 \_type ，例如：

	GET /_search
	{
	  "stored_fields" : [],
	  "query" : {
	    "term" : { "user" : "kimchy" }
	  }
	}
 
如果未存储请求的字段（将 store 映射设置为 false ），则将忽略它们。

从文档本身获取的存储字段值始终以数组形式返回。相反，像 _routing 这样的元数据字段从不作为数组返回。

此外，只能通过 stored_fields 选项返回叶子字段。如果指定了对象字段，它将被忽略。

NOTE：就其本身而言，stored_fields 不能用于加载嵌套对象中的字段，如果字段在其路径中包含嵌套对象，则不会为该存储字段返回任何数据。要访问嵌套字段，必须在 inner_hits 块内使用 stored_fields 。

### Disable stored fields
要完全禁用存储的字段（和元数据字段），请使用 \_none\_ ：

	GET /_search
	{
	  "stored_fields": "_none_",
	  "query" : {
	    "term" : { "user" : "kimchy" }
	  }
	}
 
NOTE: 如果使用 \_none_ ，则不能激活 _source 和 version 参数。

### Source filtering

您可以使用 _source 参数选择返回源的哪些字段,这称为源过滤。

以下搜索 API 请求将 _source 请求主体参数设置为 false 。该文档源不包含在响应中。

	GET /_search
	{
	  "_source": false,
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
要仅返回源字段的子集，请在 _source 参数中指定通配符（*）模式。以下搜索 API 请求仅返回 obj 字段及其属性的源。

	GET /_search
	{
	  "_source": "obj.*",
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
您还可以在 _source 字段中指定通配符模式的数组。以下搜索 API 请求仅返回 obj1 和 obj2 字段及其属性的源。

	GET /_search
	{
	  "_source": [ "obj1.*", "obj2.*" ],
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
 
为了更好地控制，您可以在 _source 参数中指定一个包含 includes 和 excludes 模式的数组的对象。

如果指定了 includes 属性，则仅返回与其模式之一匹配的源字段。您可以使用 excludes 属性从此子集中排除字段。

如果未指定 include 属性，则返回整个文档源，不包括与 excludes 属性中与模式匹配的任何字段。

以下搜索 API 请求仅返回 obj1 和 obj2 字段及其属性的源，不包括任何子 description 字段。

	GET /_search
	{
	  "_source": {
	    "includes": [ "obj1.*", "obj2.*" ],
	    "excludes": [ "*.description" ]
	  },
	  "query": {
	    "term": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
### Script fields

您可以使用 **script_fields** 参数为每个匹配检索脚本评估（基于不同的字段）。例如：

	GET /_search
	{
	  "query": {
	    "match_all": {}
	  },
	  "script_fields": {
	    "test1": {
	      "script": {
	        "lang": "painless",
	        "source": "doc['price'].value * 2"
	      }
	    },
	    "test2": {
	      "script": {
	        "lang": "painless",
	        "source": "doc['price'].value * params.factor",
	        "params": {
	          "factor": 2.0
	        }
	      }
	    }
	  }
	}
 
脚本字段可以在未存储的字段上工作（在上述情况下为价格），并允许返回要返回的自定义值（脚本的评估值）。

脚本字段还可以使用 **params['\_source']** 访问实际的 **\_source** 文档并提取要从中返回的特定元素。这是一个例子：

	GET /_search
	    {
	        "query" : {
	            "match_all": {}
	        },
	        "script_fields" : {
	            "test1" : {
	                "script" : "params['_source']['message']"
	            }
	        }
	    }
 
请注意此处的 _source 关键字，以浏览类似 json 的模型。

了解 **doc['my_field'].value** 和 **params['_source']['my_field']** 之间的区别很重要。第一个使用 doc 关键字，将导致将该字段的术语加载到内存中（缓存），这将导致执行速度更快，但会占用更多内存。另外，**doc[...]** 表示法仅允许使用简单值字段（您不能从中返回 json 对象），并且仅对未分析或基于单个术语的字段有意义。但是，仍然建议使用 doc（即使有可能）从文档访问值的方式，因为 **\_source** 每次使用时都必须加载和解析。使用 **\_source** 非常慢。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-fields.html
