## Elasticsearch 7.10 之 Search your data

搜索查询是对 Elasticsearch 数据流或索引中的数据信息的请求。

您可以将查询视为一个问题，以 Elasticsearch 理解的方式编写。根据您的数据，您可以使用查询来获取问题的答案，例如：

* 服务器上的哪些进程需要超过 500 毫秒的响应时间？
* 过去一周内，我网络上的哪些用户运行了 regsvr32.exe ？
* 我网站上的哪些页面包含特定的单词或短语？

搜索包含一个或多个查询，这些查询被组合并发送到 Elasticsearch 。与搜索查询匹配的文档会在响应的匹配数或搜索结果中返回。

搜索还可能包含用于更好地处理其查询的其他信息。例如，搜索可能仅限于特定索引，或者仅返回特定数量的结果。

### Run a search

您可以使用 [search API](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html) 搜索和聚合存储在 Elasticsearch 数据流或索引中的数据。 API 的查询请求正文参数接受以 [Query DSL](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html) 编写的查询。

以下请求使用 **match** 查询搜索 **my-index-000001** 。此查询将 user.id 值设置为 kimchy 的文档进行匹配。

	GET /my-index-000001/_search
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
API 响应返回 hits.hits 属性中与查询匹配的前10个文档。

	{
	  "took": 5,
	  "timed_out": false,
	  "_shards": {
	    "total": 1,
	    "successful": 1,
	    "skipped": 0,
	    "failed": 0
	  },
	  "hits": {
	    "total": {
	      "value": 1,
	      "relation": "eq"
	    },
	    "max_score": 1.3862942,
	    "hits": [
	      {
	        "_index": "my-index-000001",
	        "_type": "_doc",
	        "_id": "kxWFcnMByiguvud1Z8vC",
	        "_score": 1.3862942,
	        "_source": {
	          "@timestamp": "2099-11-15T14:12:12",
	          "http": {
	            "request": {
	              "method": "get"
	            },
	            "response": {
	              "bytes": 1070000,
	              "status_code": 200
	            },
	            "version": "1.1"
	          },
	          "message": "GET /search HTTP/1.1 200 1070000",
	          "source": {
	            "ip": "127.0.0.1"
	          },
	          "user": {
	            "id": "kimchy"
	          }
	        }
	      }
	    ]
	  }
	}

### Common search options

您可以使用以下选项来自定义搜索。

##### Query DSL

[Query DSL](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html) 支持多种查询类型，您可以将它们混合然后去匹配以获得所需的结果。查询类型包括：

* [Boolean](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html) 和其他 [compound](https://www.elastic.co/guide/en/elasticsearch/reference/current/compound-queries.html) 查询，可让您组合查询并根据多个条件匹配结果
* [Term-level](https://www.elastic.co/guide/en/elasticsearch/reference/current/term-level-queries.html) 查询，用于过滤和查找完全匹配
* [Full text](https://www.elastic.co/guide/en/elasticsearch/reference/current/full-text-queries.html) 查询，通常在搜索引擎中使用
* [Geo](https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-queries.html) 和 [spatial](https://www.elastic.co/guide/en/elasticsearch/reference/current/shape-queries.html) 查询

##### Aggregations
您可以使用 [search aggregations](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations.html) 来获取搜索结果的统计信息和其他分析。汇总可帮助您回答以下问题：

* 我的服务器平均响应时间是多少？
* 我的网络上的用户访问的最重要的 IP 地址是什么？
* 客户的总交易收入是多少？

##### Search multiple data streams and indices
您可以使用逗号分隔的值和类似 grep 的索引模式来搜索同一请求中的多个数据流和索引。您甚至可以提高特定索引的搜索结果。请参阅 [Search multiple data streams and indices](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-multiple-indices.html) 。

##### Paginate search results
默认情况下，搜索仅返回匹配的前 10 个匹配项。要检索更多或更少的文档，请参阅 [Paginate search results](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html)。

##### Retrieve selected fields
搜索响应的 hit.hits 属性包含每个匹配项的完整文档 _source 。要仅检索 _source 或其他字段的子集，请参阅 [Retrieve selected fields](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-fields.html)。

##### Sort search results
默认情况下，搜索结果按 _score 排序，这是一个相关分数，用于衡量每个文档与查询的匹配程度。要自定义这些分数的计算，请使用 [script_score](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html) 查询。要按其他字段值对搜索结果进行排序，请参阅 [Sort search results](https://www.elastic.co/guide/en/elasticsearch/reference/current/sort-search-results.html)。

##### Run an async search
Elasticsearch 搜索旨在快速运行在大量数据上，通常以毫秒为单位返回结果。因此，默认情况下搜索是同步的。搜索请求在返回响应之前会等待完整的结果。

但是，对于跨 [frozen indices](https://www.elastic.co/guide/en/elasticsearch/reference/current/frozen-indices.html) 或  [multiple clusters](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-cross-cluster-search.html) 的搜索，完整的结果可能需要更长的时间。

为避免长时间等待，您可以运行异步搜索。[async search ](https://www.elastic.co/guide/en/elasticsearch/reference/current/async-search-intro.html)使您可以立即检索长期运行的部分结果，以后再获取完整结果。

### Search timeout
默认情况下，搜索请求不会超时。该请求等待完整结果，然后返回响应。

虽然异步搜索是为长时间运行的搜索而设计的，但是您也可以使用 timeout 参数指定要等待搜索完成的持续时间。如果在此期间结束之前未收到任何响应，则请求将失败并返回错误。

	GET /my-index-000001/_search
	{
	  "timeout": "2s",
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
要为所有搜索请求设置集群范围的默认超时，请使用 [cluster settings API](https://www.elastic.co/guide/en/elasticsearch/reference/current/cluster-update-settings.html) 配置 **search.default_search_timeout** 。如果请求中未传递超时参数，则使用此全局超时持续时间。如果全局搜索超时在搜索请求完成之前到期，则使用 [task cancellation](https://www.elastic.co/guide/en/elasticsearch/reference/current/tasks.html#task-cancellation) 来取消该请求。 **search.default\_search_timeout** 设置默认为 -1（无超时）。

### Search cancellation

您可以使用 [task management API](https://www.elastic.co/guide/en/elasticsearch/reference/current/tasks.html#task-cancellation) 取消搜索请求。当客户端的 HTTP 连接关闭时，Elasticsearch 还会自动取消搜索请求。我们建议您将客户端设置为在搜索请求中止或超时时关闭 HTTP 连接。

### Track total hits

通常，如果不访问所有匹配项，就无法准确计算总匹配数，这对于匹配大量文档的查询而言代价很高。 track_total_hits 参数使您可以控制如何跟踪总点击数。考虑到通常只需命中次数的下限即可，例如 “至少有 10000 次命中”，因此默认设置为 10000 。这意味着请求将准确计算总匹配数，最高可达 10000 个匹配数。如果您在一定的阈值后不需要准确的点击数，那么这是加快搜索速度的一个不错的权衡。

设置为 true 时，搜索响应将始终跟踪准确匹配查询的匹配数（例如，当track_total_hits 设置为 true 时，total.relation 始终等于“ eq”）。否则搜索响应中 “total” 对象中返回的 “relation” 决定了应如何解释 “total.value”。值 “gte” 表示 “total.value” 是匹配查询的总匹配的下限，而值 “eq” 表示 “total.value” 是准确的计数。
	
	GET my-index-000001/_search
	{
	  "track_total_hits": true,
	  "query": {
	    "match" : {
	      "user.id" : "elkbee"
	    }
	  }
	}
 
...返回：

	{
	  "_shards": ...
	  "timed_out": false,
	  "took": 100,
	  "hits": {
	    "max_score": 1.0,
	    "total" : {
	      "value": 2048,     # 与查询匹配的总点击数
	      "relation": "eq"    # 该计数是准确的（例如，“ eq”表示相等）
	    },
	    "hits": ...
	  }
	}



也可以将 **track\_total_hits** 设置为整数。例如，以下查询将准确跟踪与该查询匹配的总命中数（最多 100 个文档）：

	GET my-index-000001/_search
	{
	  "track_total_hits": 100,
	  "query": {
	    "match": {
	      "user.id": "elkbee"
	    }
	  }
	}
 
响应中的 hits.total.relation 将指示 hits.total.value 中返回的值是准确的（“eq”）或者总数的下限（“gte”）。

例如以下响应：

	{
	  "_shards": ...
	  "timed_out": false,
	  "took": 30,
	  "hits": {
	    "max_score": 1.0,
	    "total": { 
	      "value": 42,   # 42 个符合条件的文件      
	      "relation": "eq"     # 计数是准确的（“eq”）
	    },
	    "hits": ...
	  }
	}

...表示返回的总命中数是准确的。

如果与查询匹配的总命中数大于 **track\_total_hits** 中设置的值，则响应中的总命中数将指示返回的值是一个下限：

	{
	  "_shards": ...
	  "hits": {
	    "max_score": 1.0,
	    "total": {
	      "value": 100,     # 至少有100个符合查询条件的文档    
	      "relation": "gte"    # 这是一个下限（“gte”） 
	    },
	    "hits": ...
	  }
	}


如果您根本不需要跟踪总命中数，则可以通过将此选项设置为 false 来缩短查询时间：

	GET my-index-000001/_search
	{
	  "track_total_hits": false,
	  "query": {
	    "match": {
	      "user.id": "elkbee"
	    }
	  }
	}
 
...返回：

	{
	  "_shards": ...
	  "timed_out": false,
	  "took": 10,
	  "hits": {       # 总命中数未知      
	    "max_score": 1.0,
	    "hits": ...
	  }
	}



最后，您可以通过在请求中将 “track_total_hits” 设置为 true 来强制进行精确计数。

### Quickly check for matching docs

如果仅想知道是否有匹配特定查询的文档，可以将大小设置为 0 表示我们对搜索结果不感兴趣。您也可以将 terminate\_after 设置为 1 ，以指示只要找到第一个匹配的文档（每个分片）就可以终止查询执行。

	GET /_search?q=user.id:elkbee&size=0&terminate_after=1
 
NOTE：当在分片上收集到足够的匹配量时，terminate_after 总是在 post\_filter 之后应用，并停止查询以及聚合执行。尽管汇总的文档数可能无法反映响应中的 hits.total ，因为汇总是在后过滤之前应用的。

响应将不包含任何匹配，因为大小设置为 0 。hits.total 等于 0，表示没有匹配的文档，或者大于 0 ，表示匹配查询的文档至少与之相同。当它提前终止时。同样，如果查询提早终止，则响应中的 Terminate_early 标志将设置为 true 。
	
	{
	  "took": 3,
	  "timed_out": false,
	  "terminated_early": true,
	  "_shards": {
	    "total": 1,
	    "successful": 1,
	    "skipped" : 0,
	    "failed": 0
	  },
	  "hits": {
	    "total" : {
	        "value": 1,
	        "relation": "eq"
	    },
	    "max_score": null,
	    "hits": []
	  }
	}
	
响应中所花费的时间包含此请求处理所需的毫秒数，从节点接收到查询后开始，直到完成所有与搜索相关的工作并且将上述 JSON 返回给客户端之前的时间都在内。这意味着它包括在线程池中等待，在整个集群中执行分布式搜索以及收集所有结果所花费的时间。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/search-your-data.html
