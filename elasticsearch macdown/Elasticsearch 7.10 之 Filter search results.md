## Elasticsearch 7.10 之 Filter search results

您可以使用两种方法来过滤搜索结果：

* 将布尔查询与 filter 子句一起使用。搜索请求将布尔过滤器应用于搜索结果和汇总。
* 使用搜索 API 的 post_filter 参数。搜索请求仅将 [post filters](https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html#post-filter) 应用于搜索命中，而不应用于汇总。您可以使用 post filter 根据更广泛的结果集计算聚合，然后进一步缩小结果范围。

您还可以在 post filter 之后 [rescore](https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html#rescore) ，以提高相关性和重新排序结果。

### Post filter

使用 post_filter 参数过滤搜索结果时，将在计算聚合后过滤搜索结果。post filter 对聚合结果没有影响。

例如，您正在销售具有以下属性的衬衫：

	PUT /shirts
	{
	  "mappings": {
	    "properties": {
	      "brand": { "type": "keyword"},
	      "color": { "type": "keyword"},
	      "model": { "type": "keyword"}
	    }
	  }
	}
	
	PUT /shirts/_doc/1?refresh
	{
	  "brand": "gucci",
	  "color": "red",
	  "model": "slim"
	}
 
假设用户指定了两个过滤器：

color:red 和 brand:gucci 。您只想在搜索结果中向他们显示 Gucci 制造的红色衬衫。通常，您可以使用布尔查询来执行此操作：

	GET /shirts/_search
	{
	  "query": {
	    "bool": {
	      "filter": [
	        { "term": { "color": "red"   }},
	        { "term": { "brand": "gucci" }}
	      ]
	    }
	  }
	}
 
但是，您还希望使用分面导航来显示用户可以单击的其他选项的列表。也许您有一个 model 字段，该字段允许用户将搜索结果限制为红色 Gucci t-shirts 或 dress-shirts 。

这可以通过 [terms aggregation](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-terms-aggregation.html) 来完成：

	GET /shirts/_search
	{
	  "query": {
	    "bool": {
	      "filter": [
	        { "term": { "color": "red"   }},
	        { "term": { "brand": "gucci" }}
	      ]
	    }
	  },
	  "aggs": {
	    "models": {
	      "terms": { "field": "model" }   # 返回 Gucci 最受欢迎的红色衬衫型号
	    }
	  }
	}

但是也许您还想告诉用户有多少种其他颜色的 Gucci 衬衫可供选择。如果仅在 color 字段上添加 terms aggregation ，则只会返回 red ，因为查询仅返回 Gucci 的红色衬衫。

相反，您希望在聚合过程中包括所有颜色的衬衫，然后仅将 color 过滤器应用于搜索结果。这是 post_filter 的目的：

	GET /shirts/_search
	{
	  "query": {
	    "bool": {
	      "filter": {
	        "term": { "brand": "gucci" }    # 现在，主查询将查找所有 Gucci 衬衫，无论颜色如何
	      }
	    }
	  },
	  "aggs": {
	    "colors": {
	      "terms": { "field": "color" }    # color agg 返回 Gucci 衬衫流行的颜色
	    },
	    "color_red": {
	      "filter": {
	        "term": { "color": "red" }    # color_red agg 将模型子集合限制为红色 Gucci 衬衫
	      },
	      "aggs": {
	        "models": {
	          "terms": { "field": "model" } 
	        }
	      }
	    }
	  },
	  "post_filter": {    # 最后 post_filter 会从搜索结果中删除红色以外的颜色
	    "term": { "color": "red" }
	  }
	}
 

### Rescore filtered search results

通过使用次要（通常更昂贵）算法，而不是对索引中的所有文档应用昂贵的算法，通过仅对 query 和 post_filter 阶段返回的最顶部（例如100-500）文档进行重新排序，记录可以帮助提高准确性。

在每个分片返回结果之前，将对每个分片执行一个 rescore 请求，该结果将由处理整个搜索请求的节点进行排序。

当前 rescore API 仅具有一种实现：查询 rescorer，该查询使用查询来调整得分。将来可能会提供替代的记录器，例如 pair-wise rescorer 。

NOTE: 如果为 rescore 查询提供了显式排序（降序为 _score 除外），则将引发错误。

NOTE: 在向用户展示分页时，您不应在逐步浏览每个页面时更改 window_size（通过传递与值不同的值），因为这可能会更改顶部匹配，从而导致结果在用户逐步浏览页面时引起混乱。

##### Query rescorer

query rescorer 仅对 query 和 post_filter 阶段返回的 Top-K 结果执行第二个查询。每个分片将要检查的文档数量可以由 window\_size 参数控制，该参数默认为 10。

默认情况下，原始查询和重新评分查询的评分会线性组合，以生成每个文档的最终 _score 。原始查询和 rescore 查询的相对重要性可以分别通过 query\_weight 和 rescore\_query\_weight 进行控制。两者都默认为 1 。

例如：

	POST /_search
	{
	   "query" : {
	      "match" : {
	         "message" : {
	            "operator" : "or",
	            "query" : "the quick brown"
	         }
	      }
	   },
	   "rescore" : {
	      "window_size" : 50,
	      "query" : {
	         "rescore_query" : {
	            "match_phrase" : {
	               "message" : {
	                  "query" : "the quick brown",
	                  "slop" : 2
	               }
	            }
	         },
	         "query_weight" : 0.7,
	         "rescore_query_weight" : 1.2
	      }
	   }
	}
 
分数的组合方式可以通过 **score_mode** 控制：


分数模式|说明
---- | ---- 
total | 添加原始分数和重新评分查询分数，是默认值
multiply | 将原始分数乘以重新评分查询分数，对于函数查询重计很有用
avg | 平均原始分数和重新评分查询分数
max | 取得原始分数和重新评分查询分数的最大值
min | 取原始分数和重新评分查询分数的最小值

##### Multiple rescores

也可以依次执行多个重新评分：
	
	POST /_search
	{
	   "query" : {
	      "match" : {
	         "message" : {
	            "operator" : "or",
	            "query" : "the quick brown"
	         }
	      }
	   },
	   "rescore" : [ {
	      "window_size" : 100,
	      "query" : {
	         "rescore_query" : {
	            "match_phrase" : {
	               "message" : {
	                  "query" : "the quick brown",
	                  "slop" : 2
	               }
	            }
	         },
	         "query_weight" : 0.7,
	         "rescore_query_weight" : 1.2
	      }
	   }, {
	      "window_size" : 10,
	      "query" : {
	         "score_mode": "multiply",
	         "rescore_query" : {
	            "function_score" : {
	               "script_score": {
	                  "script": {
	                    "source": "Math.log10(doc.count.value + 2)"
	                  }
	               }
	            }
	         }
	      }
	   } ]
	}
 
第一个获取查询的结果，然后第二个获取第一个的查询结果，依此类推。第二个重新评分将看到第一个重新评分完成的排序，因此可以在第一个重新评分上使用大窗口来查询将文档拉入较小的窗口中以进行第二次重新评分。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html
