## Elasticsearch 7.10 之 Paginate search results

默认情况下，搜索会返回前 10 个匹配的匹配项。要浏览更大的一组结果，可以使用 [search API](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html) 的 from 和 size 参数。 from 参数定义要跳过的匹配数，默认为 0 。size 参数是要返回的最大匹配数。这两个参数共同定义了结果页面。

	GET /_search
	{
	  "from": 5,
	  "size": 20,
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
避免过度使用 from 和 size 来分页或一次请求太多结果。搜索请求通常跨越多个分片。每个分片必须将其请求的命中以及任何先前页面的命中加载到内存中。对于较深的页面或大量结果，这些操作会显着增加内存和 CPU 使用率，从而导致性能下降或节点故障。

默认情况下，您不能使用 from 和 size 翻页超过 10000 个匹配项。此限制是 [index.max\_result_window](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules.html#index-max-result-window) 索引设置所设置的保护措施。如果您需要分页浏览超过 10,000 个匹配项，请改用 [search_after](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html#search-after) 参数。

WARNING: Elasticsearch 使用 Lucene 的内部文档 ID 作为平局。这些内部文档 ID 在相同数据的副本之间可能完全不同。当分页搜索命中时，您可能偶尔会看到具有相同排序值的文档的顺序不一致。

### Search after

您可以使用 search_after 参数，使用前一页中的一组 [sort values](https://www.elastic.co/guide/en/elasticsearch/reference/current/sort-search-results.html) 来检索匹配的下一页。

使用 search_after 要求具有相同 query 和 sort 值的多个搜索请求。如果在这些请求之间发生了 refresh ，则结果的顺序可能会更改，从而导致页面之间的结果不一致。为防止这种情况，您可以创建一个时间点（PIT）以在搜索过程中保留当前索引状态。

	POST /my-index-000001/_pit?keep_alive=1m
 
API 返回一个 PIT ID。

	{
	  "id": "46ToAwMDaWR4BXV1aWQxAgZub2RlXzEAAAAAAAAAAAEBYQNpZHkFdXVpZDIrBm5vZGVfMwAAAAAAAAAAKgFjA2lkeQV1dWlkMioGbm9kZV8yAAAAAAAAAAAMAWICBXV1aWQyAAAFdXVpZDEAAQltYXRjaF9hbGw_gAAAAA=="
	}
	
要获得结果的第一页，请提交带有 sort 参数的搜索请求。如果使用 PIT，请在 pit.id 参数中指定 PIT ID ，并从请求路径中省略目标数据流或索引。

WARNING：我们建议您在排序中加入 tiebreaker 字段。tiebreaker 字段应为每个文档包含了唯一值。如果您不包含 tiebreaker 字段，则分页结果可能会丢失或重复匹配。

	GET /_search
	{
	  "size": 10000,
	  "query": {
	    "match" : {
	      "user.id" : "elkbee"
	    }
	  },
	  "pit": {
		    "id":  "46ToAwMDaWR4BXV1aWQxAgZub2RlXzEAAAAAAAAAAAEBYQNpZHkFdXVpZDIrBm5vZGVfMwAAAAAAAAAAKgFjA2lkeQV1dWlkMioGbm9kZV8yAAAAAAAAAAAMAWICBXV1aWQyAAAFdXVpZDEAAQltYXRjaF9hbGw_gAAAAA==",       # 搜索的 PIT ID
		    "keep_alive": "1m"
	  },
	  "sort": [    # 对搜索结果进行排序
	    {"@timestamp": "asc"},
	    {"tie_breaker_id": "asc"}
	  ]
	}


搜索响应包括每个匹配项的排序值数组。如果您使用的是 PIT ，则响应的 pit_id 参数包含了已经更新的 PIT ID 。

	{
	  "pit_id" : "46ToAwEPbXktaW5kZXgtMDAwMDAxFnVzaTVuenpUVGQ2TFNheUxVUG5LVVEAFldicVdzOFFtVHZTZDFoWWowTGkwS0EAAAAAAAAAAAQURzZzcUszUUJ5U1NMX3Jyak5ET0wBFnVzaTVuenpUVGQ2TFNheUxVUG5LVVEAAA==",    # 该时间点更新了的 ID
	  "took" : 17,
	  "timed_out" : false,
	  "_shards" : ...,
	  "hits" : {
	    "total" : ...,
	    "max_score" : null,
	    "hits" : [
	      ...
	      {
	        "_index" : "my-index-000001",
	        "_id" : "FaslK3QBySSL_rrj9zM5",
	        "_score" : null,
	        "_source" : ...,
	        "sort" : [       # 返回的最后一个匹配的值                         
	          4098435132000,
	          "FaslK3QBySSL_rrj9zM5"
	        ]
	      }
	    ]
	  }
	}


要获取下一页结果，请使用上次匹配的排序值作为 search_after 参数重新运行上一个搜索。如果使用 PIT ，请在 pit.id 参数中使用最新的 PIT ID 。搜索的 query 和 sort 参数必须保持不变。如果提供，则 from 参数必须为 0（默认值）或 -1 。

	GET /_search
	{
	  "size": 10000,
	  "query": {
	    "match" : {
	      "user.id" : "elkbee"
	    }
	  },
	  "pit": {
		    "id":  "46ToAwEPbXktaW5kZXgtMDAwMDAxFnVzaTVuenpUVGQ2TFNheUxVUG5LVVEAFldicVdzOFFtVHZTZDFoWWowTGkwS0EAAAAAAAAAAAQURzZzcUszUUJ5U1NMX3Jyak5ET0wBFnVzaTVuenpUVGQ2TFNheUxVUG5LVVEAAA==",    # 先前搜索返回的 PIT ID
		    "keep_alive": "1m"
	  },
	  "sort": [
	    {"@timestamp": "asc"},
	    {"tie_breaker_id": "asc"}
	  ],
	  "search_after": [        # 对上一次搜索的最后一个匹配结果之后的值进行排序                        
	    4098435132000,
	    "FaslK3QBySSL_rrj9zM5"
	  ]
	} 



您可以重复此过程以获取其他页面的结果。如果使用 PIT ，则可以使用每个搜索请求的 keep_alive 参数来延长 PIT 的保留期限。

完成后，您应该删除您的 PIT

	DELETE /_pit
	{
	    "id" : "46ToAwEPbXktaW5kZXgtMDAwMDAxFnVzaTVuenpUVGQ2TFNheUxVUG5LVVEAFldicVdzOFFtVHZTZDFoWWowTGkwS0EAAAAAAAAAAAQURzZzcUszUUJ5U1NMX3Jyak5ET0wBFnVzaTVuenpUVGQ2TFNheUxVUG5LVVEAAA=="
	}


### Scroll search results
IMPORTANT: 我们不再建议使用 scroll API 进行深度分页。如果在翻阅 10000 多个匹配项时需要保留索引状态，请使用带有时间点（PIT）的 search_after 参数。

当搜索请求返回单个“页面”结果时，scroll API 可用于从单个搜索请求中检索大量结果（甚至所有结果），其方式与在传统数据库上使用光标的方式几乎相同。

scroll 并非用于实时用户请求，而是用于处理大量数据，例如为了将一个数据流或索引的内容重新索引为具有不同配置的新数据流或索引。


NOTE：从 scroll 请求返回的结果，反映了发出初始搜索请求时，数据流或索引的状态，如时间快照。对文档的后续更改（索引，更新或删除）将仅影响以后的搜索请求。

为了使用 scroll ，初始搜索请求应在查询字符串中指定 scroll 参数，该参数告诉 Elasticsearch 它应将“搜索上下文”保持活动状态的时间（请 [Keeping the search context alive](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html#scroll-search-context) ），例如 ?scroll=1m 。

	POST /my-index-000001/_search?scroll=1m
	{
	  "size": 100,
	  "query": {
	    "match": {
	      "message": "foo"
	    }
	  }
	}
 
上述请求的结果包括一个 _scroll_id ，应将其传递给 scroll API 以便检索下一批结果。
	
	POST /_search/scroll     # 可以使用 GET 或 POST ，并且 URL 不应包含索引名称，而是在原始搜索请求中指定                                                           
	{
	  "scroll" : "1m",     # scroll 参数告诉 Elasticsearch 将搜索的上下文再打开 1 分钟                                                            
	  "scroll_id" : "DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAAD4WYm9laVYtZndUQlNsdDcwakFMNjU1QQ=="      # scroll_id 参数
	}



size 参数允许您配置每批结果返回的最大匹配数。每次对 scroll API 的调用都会返回下一批结果，直到没有其他要返回的结果为止，即 hits 数组为空。

IMPORTANT：初始搜索请求和每个后续 scroll 请求均返回一个 \_scroll\_id 。尽管 \_scroll\_id 在两次请求之间可能会发生变化，但并非总是会发生变化，在任何情况下，都应仅使用最近收到的 _scroll\_id 。

NOTE：如果请求指定聚合，则仅初始搜索响应将包含聚合结果。

NOTE：scroll 请求具有优化功能，可以使排序顺序为 _doc 时更快。如果要遍历所有文档而不考虑顺序，这是最有效的选择：

	GET /_search?scroll=1m
	{
	  "sort": [
	    "_doc"
	  ]
	}
 
##### Keeping the search context alive
scroll 返回在初始搜索请求时与搜索匹配到的所有文档。它忽略了对这些文档的任何后续更改。 scroll_id 标识一个搜索上下文，该上下文跟踪 Elasticsearch 返回正确文档所需的一切。搜索上下文由初始请求创建，并由后续请求保持活动状态。

scroll 参数（传递给搜索请求和每个 scroll 请求）告诉 Elasticsearch 它应该保持搜索上下文存活多长时间。它的值（例如 1m ，请参阅 [Time units](https://www.elastic.co/guide/en/elasticsearch/reference/current/common-options.html#time-units) ）不需要足够长的时间来处理所有数据，仅需要足够长的时间来处理前一批结果。每个 scroll 请求（带有 scroll 参数）都设置一个新的到期时间。如果没有在 scroll 参数中传递 scroll 请求，则搜索上下文将作为该 scroll 请求的一部分被释放。

通常，后台合并过程通过将较小的段合并在一起以创建新的较大的段来优化索引。一旦不再需要较小的段，则将其删除。在 scroll 过程中，此过程继续进行，但是开放的搜索上下文可防止删除旧的段，因为它们仍在使用中。

TIP：使较旧的段保持活动状态意味着需要更多的磁盘空间和文件句柄。确保已将节点配置为具有足够的空闲文件句柄。请参阅 [File Descriptors](https://www.elastic.co/guide/en/elasticsearch/reference/current/file-descriptors.html)。

此外，如果一个段包含已删除或更新的文档，则搜索上下文必须跟踪该段中的每个文档在初始搜索请求时是否处于活动状态。如果索引上有许多打开的 scrolls ，这些索引会不断进行删除或更新，请确保节点具有足够的堆空间。


NOTE: 为了防止打开太多 scrolls 引起的问题，不允许用户打开超过特定限制的滚动条。默认情况下，最大打开 scrolls 为 500 。可以使用 search.max\_open\_scroll\_context 群集设置来更新此限制。

您可以使用 [nodes stats API](https://www.elastic.co/guide/en/elasticsearch/reference/current/cluster-nodes-stats.html) 检查打开了多少搜索上下文：

	GET /_nodes/stats/indices/search
 
##### Clear scroll
当 scroll 超时时，搜索上下文将自动删除。但是，如上一节所述，保持 scroll 打开是有代价的，因此一旦不再使用它，则用 clear-scroll API 以明确清除滚动​​：

	DELETE /_search/scroll
	{
	  "scroll_id" : "DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAAD4WYm9laVYtZndUQlNsdDcwakFMNjU1QQ=="
	}
 
多个 scroll ID 可以作为数组传递：

	DELETE /_search/scroll
	{
	  "scroll_id" : [
	    "DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAAD4WYm9laVYtZndUQlNsdDcwakFMNjU1QQ==",
	    "DnF1ZXJ5VGhlbkZldGNoBQAAAAAAAAABFmtSWWRRWUJrU2o2ZExpSGJCVmQxYUEAAAAAAAAAAxZrUllkUVlCa1NqNmRMaUhiQlZkMWFBAAAAAAAAAAIWa1JZZFFZQmtTajZkTGlIYkJWZDFhQQAAAAAAAAAFFmtSWWRRWUJrU2o2ZExpSGJCVmQxYUEAAAAAAAAABBZrUllkUVlCa1NqNmRMaUhiQlZkMWFB"
	  ]
	}
 
可以使用 _all 参数清除所有搜索上下文：

	DELETE /_search/scroll/_all
 
scroll_id 也可以作为查询字符串参数或在请求正文中传递。多个 scroll ID 可以作为逗号分隔的值传递：

	DELETE /_search/scroll/DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAAD4WYm9laVYtZndUQlNsdDcwakFMNjU1QQ==,DnF1ZXJ5VGhlbkZldGNoBQAAAAAAAAABFmtSWWRRWUJrU2o2ZExpSGJCVmQxYUEAAAAAAAAAAxZrUllkUVlCa1NqNmRMaUhiQlZkMWFBAAAAAAAAAAIWa1JZZFFZQmtTajZkTGlIYkJWZDFhQQAAAAAAAAAFFmtSWWRRWUJrU2o2ZExpSGJCVmQxYUEAAAAAAAAABBZrUllkUVlCa1NqNmRMaUhiQlZkMWFB
 
##### Sliced scroll
对于返回大量文档的 scroll 查询，可以将 scroll 分为多个切片，这些切片可以独立使用：

	GET /my-index-000001/_search?scroll=1m
	{
	  "slice": {
	    "id": 0,      # 切片的 ID                
	    "max": 2      # 最大切片数                
	  },
	  "query": {
	    "match": {
	      "message": "foo"
	    }
	  }
	}
	GET /my-index-000001/_search?scroll=1m
	{
	  "slice": {
	    "id": 1,
	    "max": 2
	  },
	  "query": {
	    "match": {
	      "message": "foo"
	    }
	  }
	}


第一个请求的结果返回了属于第一个片（id：0）的文档，第二个请求的结果返回了属于第二个片的文档。由于切片的最大数量设置为 2 ，所以两个请求的结果的并集等效于不切片的滚动查询的结果。默认情况下，首先在分片上进行拆分，然后使用 _id 字段在每个分片上进行本地拆分，其公式如下：slice(doc) = floorMod(hashCode(doc.\_id), max) 例如，如果分片数为 2 和用户请求 4 个切片，则将切片 0 和 2 分配给第一个分片，并将切片 1 和 3 分配给第二个分片。

每个 scroll 都是独立的，并且可以像任何 scroll 请求一样并行处理。

NOTE: 如果切片的数量大于分片的数量，则切片过滤器在第一次调用时非常慢，它的复杂度为 O（N），内存成本等于每个切片 N 位，其中 N 是在分片中文档总数。几次调用后，应将筛选器缓存起来，随后的调用应更快，但应限制并行执行的切片查询的数量，以避免内存爆炸。

为了完全避免这种花费，可以使用另一个字段的 doc_values 进行切片，但是用户必须确保该字段具有以下属性：


* 该字段是数字
* doc_values 在该字段上启用
* 每个文档应包含一个值。如果文档的指定字段具有多个值，则使用第一个值
* 创建文档时，每个文档的值应设置一次，并且永远不要更新。这样可确保每个切片得到确定的结果
* 该字段的基数应该很高。这样可以确保每个切片获得的文档数量大致相同
	
		
		GET /my-index-000001/_search?scroll=1m
		{
		  "slice": {
		    "field": "@timestamp",
		    "id": 0,
		    "max": 10
		  },
		  "query": {
		    "match": {
		      "message": "foo"
		    }
		  }
		}
 
对于仅追加基于时间的索引，可以安全地使用 timestamp 字段。

默认情况下，每次 scroll 允许的最大切片数限制为 1024 。您可以更新 index.max\_slices\_per\_scroll 索引设置以绕过此限制。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html
