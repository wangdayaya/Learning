## Elasticsearch 7.10 之 Collapse search results

您可以使用 **collapse** 参数根据字段值折叠搜索结果。折叠是通过每个折叠键仅选择排序最靠前的文档来完成的。

例如，以下搜索按 user.id 折叠结果，并按 http.response.bytes 对其进行排序。

	GET /my-index-000001/_search
	{
	  "query": {
	    "match": {
	      "message": "GET /search"
	    }
	  },
	  "collapse": {
	    "field": "user.id"    # 使用 "user.id" 字段折叠结果集            
	  },
	  "sort": [ "http.response.bytes" ],    # 按 http.response.bytes 对结果进行排序
	  "from": 10     # 定义第一个折叠结果的偏移量                     
	}


WARNING: 响应中的总命中数表示没有折叠的匹配文档数。不同组的总数是未知的。

用于折叠的字段必须是单值 **keyword** 或激活了 **doc_values** 的 **numeric** 字段

NOTE: 折叠只会应用于热门匹配，并且不会影响聚合。

### Expand collapse results

也可以使用 inner_hits 选项扩展每个折叠的热门匹配。

	GET /my-index-000001/_search
	{
	  "query": {
	    "match": {
	      "message": "GET /search"
	    }
	  },
	  "collapse": {
	    "field": "user.id",    # 使用 "user.id" 字段折叠结果集                   
	    "inner_hits": {
	      "name": "most_recent",     # 响应中内部匹配部分使用的名称             
	      "size": 5,     # 每个合拢键要检索的 inner_hits 数                         
	      "sort": [ { "@timestamp": "asc" } ]    # 如何在每个组中对文档进行排序 
	    },
	    "max_concurrent_group_searches": 4    # 每个组允许检索并发 inner_hits 的并发请求数    
	  },
	  "sort": [ "http.response.bytes" ]
	}


有关支持的选项的完整列表和响应的格式，请参见 [inner hits](https://www.elastic.co/guide/en/elasticsearch/reference/current/inner-hits.html) 。

还可以为每个折叠的匹配请求多个 inner_hits 。当您想要获得折叠后的匹配的多种表示形式时，这很有用。

	GET /my-index-000001/_search
	{
	  "query": {
	    "match": {
	      "message": "GET /search"
	    }
	  },
	  "collapse": {
	    "field": "user.id",      # 使用 "user.id" 字段折叠结果集                
	      "inner_hits": [
	      {
	        "name": "largest_responses",     # 返回用户的三个最大的 HTTP 响应    
	        "size": 3,
	        "sort": [ "http.response.bytes" ]
	      },
	      {
	        "name": "most_recent",     # 返回用户的三个最新 HTTP 响应          
	        "size": 3,
	        "sort": [ { "@timestamp": "asc" } ]
	      }
	    ]
	  },
	  "sort": [ "http.response.bytes" ]
	}
 

通过为响应中返回的每个折叠的匹配项的每个 inner_hit 请求发送一个附加查询，来完成组的扩展。如果您有太多的组和/或 inner_hit 请求，这会大大降低速度。

max_concurrent_group_searches 请求参数可用于控制此阶段允许的最大并发搜索数。默认值基于数据节点的数量和默认搜索线程池大小。

WARNING: collapse 不能与 [scroll](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html#scroll-search-results), [rescore](https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html#rescore) or [search after](https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html#search-after) 结合使用。

### Second level of collapsing

还支持第二级折叠，并将其应用于 inner_hits 。

例如，以下搜索按 geo.country_name 折叠结果。在每个 geo.country_name 内，user.id 折叠内部匹配。

	GET /my-index-000001/_search
	{
	  "query": {
	    "match": {
	      "message": "GET /search"
	    }
	  },
	  "collapse": {
	    "field": "geo.country_name",
	    "inner_hits": {
	      "name": "by_location",
	      "collapse": { "field": "user.id" },
	      "size": 3
	    }
	  }
	}
响应：

	{
	  ...
	  "hits": [
	    {
	      "_index": "my-index-000001",
	      "_type": "_doc",
	      "_id": "9",
	      "_score": ...,
	      "_source": {...},
	      "fields": { "geo": { "country_name": [ "UK" ] }},
	      "inner_hits": {
	        "by_location": {
	          "hits": {
	            ...,
	            "hits": [
	              {
	                ...
	                "fields": { "user": "id": { [ "user124" ] }}
	              },
	              {
	                ...
	                "fields": { "user": "id": { [ "user589" ] }}
	              },
	              {
	                ...
	                "fields": { "user": "id": { [ "user001" ] }}
	              }
	            ]
	          }
	        }
	      }
	    },
	    {
	      "_index": "my-index-000001",
	      "_type": "_doc",
	      "_id": "1",
	      "_score": ..,
	      "_source": {...
	      },
	      "fields": { "geo": { "country_name": [ "Canada" ] }},
	      "inner_hits": {
	        "by_location": {
	          "hits": {
	            ...,
	            "hits": [
	              {
	                ...
	                "fields": { "user": "id": { [ "user444" ] }}
	              },
	              {
	                ...
	                "fields": { "user": "id": { [ "user1111" ] }
	              },
	              {
	                ...
	                  "fields": { "user": "id": { [ "user999" ] }}
	              }
	            ]
	          }
	        }
	      }
	    },
	    ...
	  ]
	}

NOTE: 折叠合并的第二级不允许 inner_hits 。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/collapse-search-results.html
