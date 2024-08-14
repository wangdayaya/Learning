## Elasticsearch 7.10 之 Index Sorting

在 Elasticsearch 中创建新索引时，可以配置如何对每个分片内的 segments 进行排序。默认情况下，Lucene 不进行任何排序。  **index.sort.\*** 设置定义应使用哪些字段对每个 Segment 中的文档进行排序。

WAENING: 嵌套字段与索引排序不兼容，因为它们依赖于以下假设：嵌套文档存储在连续的 doc id 中，可以通过索引排序将其破坏。如果对包含嵌套字段的索引激活索引排序，将引发错误。

例如，以下示例显示如何在单个字段上定义排序：

	PUT my-index-000001
	{
	  "settings": {
	    "index": {
	      "sort.field": "date",    # 该索引按日期字段排序
	      "sort.order": "desc"     # 降序排列
	    }
	  },
	  "mappings": {
	    "properties": {
	      "date": {
	        "type": "date"
	      }
	    }
	  }
	}
 

也可以按多个字段对索引进行排序：

	PUT my-index-000001
	{
	  "settings": {
	    "index": {
	      "sort.field": [ "username", "date" ],    # 该索引首先按 username 排序，然后按 date 排序
	      "sort.order": [ "asc", "desc" ]    # username 字段按升序排列，date 字段按降序排列   
	    }
	  },
	  "mappings": {
	    "properties": {
	      "username": {
	        "type": "keyword",
	        "doc_values": true
	      },
	      "date": {
	        "type": "date"
	      }
	    }
	  }
	}
 

索引排序支持以下设置：

**index.sort.field** :用于对索引进行排序的字段列表。此处仅允许使用带有 **doc_values** 的 **boolean** , **numeric** , **date** 和 **keyword** 字段。

**index.sort.order** :每个字段要使用的排序顺序。 order 选项可以具有以下值：

* asc：升序排列
* desc：降序排列

**index.sort.mode** : Elasticsearch 支持按多值字段排序。模式选项控制选择哪个值对文档进行排序。模式选项可以具有以下值：

* min：选择最低值
* max：选择最大值

**index.sort.missing** : missing 参数指定应该如何处理缺少该字段的文档。缺少的值可以具有以下值：

* _last：没有该字段值的文档排在最后
* _first：没有该字段值的文档将首先排序

WARNING: 索引排序只能在创建索引时定义一次。不允许在现有索引上添加或更新排序。由于必须在刷新和合并时对文档进行排序，因此索引排序在索引吞吐量方面也要付出代价。在激活此功能之前，应该测试对应用程序的影响。

### Early termination of search request

默认情况下，在 Elasticsearch 中，搜索请求必须访问与查询匹配的每个文档，以检索按指定排序的最重要文档。尽管当索引排序和搜索排序相同时，可以限制每个段应访问的文档数，以全局检索 N 个排名最高的文档。例如，假设我们有一个索引，其中包含按时间戳记字段排序的事件：

	PUT events
	{
	  "settings": {
	    "index": {
	      "sort.field": "timestamp",
	      "sort.order": "desc"    # 该索引按时间戳按降序排序（最新的优先）
	    }
	  },
	  "mappings": {
	    "properties": {
	      "timestamp": {
	        "type": "date"
	      }
	    }
	  }
	}
 



您可以使用以下内容搜索最近的 10 个事件：

	GET /events/_search
	{
	  "size": 10,
	  "sort": [
	    { "timestamp": "desc" }
	  ]
	}
	
Elasticsearch 将检测那些已在索引中排序的段中顶级的文档，并且只会比较每个段的前 N 个文档。收集与查询匹配的其余文档以计算结果总数并建立汇总。

如果您只查找最近的 10 个事件，而对与查询匹配的文档总数不感兴趣，则可以将 **track\_total_hits** 设置为 false ：

	GET /events/_search
	{
	  "size": 10,
	  "sort": [    # 索引排序将用于对排名靠前的文档进行排序，并且每个段将在前 10 个匹配项之后提前终止集合
	      { "timestamp": "desc" }
	  ],
	  "track_total_hits": false
	}
 


这次，Elasticsearch 将不会尝试计算文档数，并且一旦每个段收集了 N 个文档，便可以终止查询。
	
	{
	  "_shards": ...
	   "hits" : {     # 由于提前终止，与查询匹配的总数是未知的
	      "max_score" : null,
	      "hits" : []
	  },
	  "took": 20,
	  "timed_out": false
	}

NOTE: 聚合将收集与查询匹配的所有文档，而与 track\_total_hits 的值无关


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-index-sorting.html
