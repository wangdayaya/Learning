## Elasticsearch 7.10 之 Aggregations

汇总会将您的数据汇总为指标、统计信息或其他分析结果。 汇总可帮助您回答以下问题：

* 我的网站平均加载时间是多少？
* 根据交易量，谁是我最有价值的客户？
* 什么会被视为我网络上的大文件？
* 每个产品类别中有多少个产品？

Elasticsearch 将聚合分为三类：

* Metric aggregations：从字段值计算指标（例如总和或平均值）的指标聚合。
* Bucket aggregations：桶聚合，根据字段值、范围或其他条件将文档分组为桶（也称为箱）。
* Pipeline aggregations：管道聚合从其他聚合（而不是文档或字段）获取输入。

### Run an aggregation

通过指定 search API 的 **aggs** 参数，您可以在搜索过程中运行聚合。以下搜索在 **my-field** 上运行[ terms aggregation ](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-aggregations-bucket-terms-aggregation.html)：

	GET /my-index-000001/_search
	{
	  "aggs": {
	    "my-agg-name": {
	      "terms": {
	        "field": "my-field"
	      }
	    }
	  }
	}
 
汇总结果位于响应的汇总对象中：
	
	{
	  "took": 78,
	  "timed_out": false,
	  "_shards": {
	    "total": 1,
	    "successful": 1,
	    "skipped": 0,
	    "failed": 0
	  },
	  "hits": {
	    "total": {
	      "value": 5,
	      "relation": "eq"
	    },
	    "max_score": 1.0,
	    "hits": [...]
	  },
	  "aggregations": {
	    "my-agg-name": {      # my-agg-name 聚合的结果                     
	      "doc_count_error_upper_bound": 0,
	      "sum_other_doc_count": 0,
	      "buckets": []
	    }
	  }
	}



### Change an aggregation’s scope
使用 query 参数限制运行聚合的文档：
	
	GET /my-index-000001/_search
	{
	  "query": {
	    "range": {
	      "@timestamp": {
	        "gte": "now-1d/d",
	        "lt": "now/d"
	      }
	    }
	  },
	  "aggs": {
	    "my-agg-name": {
	      "terms": {
	        "field": "my-field"
	      }
	    }
	  }
	}
 
### Return only aggregation results

默认情况下，包含聚合的搜索会同时返回搜索结果和聚合结果。要仅返回聚合结果，请将 **size** 设置为 0 ：

	GET /my-index-000001/_search
	{
	  "size": 0,
	  "aggs": {
	    "my-agg-name": {
	      "terms": {
	        "field": "my-field"
	      }
	    }
	  }
	}
 
### Run multiple aggregations

您可以在同一请求中指定多个聚合：

	GET /my-index-000001/_search
	{
	  "aggs": {
	    "my-first-agg-name": {
	      "terms": {
	        "field": "my-field"
	      }
	    },
	    "my-second-agg-name": {
	      "avg": {
	        "field": "my-other-field"
	      }
	    }
	  }
	}

### Run sub-aggregations

桶聚合支持桶或度量标准子聚合。例如，一项带有 avg 子聚合的术语聚合可为每个文档桶计算一个平均值。嵌套子聚合没有级别或深度限制。

	GET /my-index-000001/_search
	{
	  "aggs": {
	    "my-agg-name": {
	      "terms": {
	        "field": "my-field"
	      },
	      "aggs": {
	        "my-sub-agg-name": {
	          "avg": {
	            "field": "my-other-field"
	          }
	        }
	      }
	    }
	  }
	}
 
响应将子聚合结果嵌套在其父聚合下：

	{
	  ...
	  "aggregations": {
	    "my-agg-name": {      # 父级聚合的结果 my-agg-name                     
	      "doc_count_error_upper_bound": 0,
	      "sum_other_doc_count": 0,
	      "buckets": [
	        {
	          "key": "foo",
	          "doc_count": 5,
	          "my-sub-agg-name": {    # my-agg-name 的子聚合结果 my-sub-agg-name             
	            "value": 75.0
	          }
	        }
	      ]
	    }
	  }
	}



### Add custom metadata

使用 **meta** 对象将自定义元数据与聚合关联：

	GET /my-index-000001/_search
	{
	  "aggs": {
	    "my-agg-name": {
	      "terms": {
	        "field": "my-field"
	      },
	      "meta": {
	        "my-metadata-field": "foo"
	      }
	    }
	  }
	}
 
响应将原位返回元对象：

	{
	  ...
	  "aggregations": {
	    "my-agg-name": {
	      "meta": {
	        "my-metadata-field": "foo"
	      },
	      "doc_count_error_upper_bound": 0,
	      "sum_other_doc_count": 0,
	      "buckets": []
	    }
	  }
	}

### Return the aggregation type
默认情况下，聚合结果包括聚合名称，但不包括其类型。要返回聚合类型，请使用 **typed\_keys** 查询参数。

	GET /my-index-000001/_search?typed_keys
	{
	  "aggs": {
	    "my-agg-name": {
	      "histogram": {
	        "field": "my-field",
	        "interval": 1000
	      }
	    }
	  }
	}
 
响应返回聚合类型作为聚合名称的前缀。

某些聚合返回的聚合类型与请求中的类型不同。例如，术语，重要术语和百分位数聚合返回的聚合类型取决于聚合字段的数据类型。

	{
	  ...
	  "aggregations": {
	    "histogram#my-agg-name": {                 
	      "buckets": []
	    }
	  }
	}

聚合类型的直方图，后跟＃分隔符，以及聚合名称 my-agg-name。

### Use scripts in an aggregation

一些聚合支持脚本。您可以使用 script 来提取或生成聚合值：

	GET /my-index-000001/_search
	{
	  "aggs": {
	    "my-agg-name": {
	      "histogram": {
	        "interval": 1000,
	        "script": {
	          "source": "doc['my-field'].value.length()"
	        }
	      }
	    }
	  }
	}
 
如果您还指定一个字段，则脚本将修改聚合中使用的字段值。以下聚合使用脚本来修改我的字段值：

	GET /my-index-000001/_search
	{
	  "aggs": {
	    "my-agg-name": {
	      "histogram": {
	        "field": "my-field",
	        "interval": 1000,
	        "script": "_value / 1000"
	      }
	    }
	  }
	}

 
某些聚合仅适用于特定的数据类型。使用 **value\_type** 参数为脚本生成的值或未映射的字段指定数据类型。 **value\_type** 接受以下值：

* boolean
* date
* double, used for all floating-point numbers
* long, used for all integers
* ip
* string

		GET /my-index-000001/_search
		{
		  "aggs": {
		    "my-agg-name": {
		      "histogram": {
		        "field": "my-field",
		        "interval": 1000,
		        "script": "_value / 1000",
		        "value_type": "long"
		      }
		    }
		  }
		}
 
### Aggregation caches
为了获得更快的响应，Elasticsearch 将频繁运行的聚合结果缓存在分片请求缓存中。要获取缓存的结果，请对每个搜索使用相同的首选项字符串。如果您不需要搜索匹配，请将大小设置为 0 ，以避免填满缓存。

Elasticsearch 将具有相同首选项字符串的搜索路由到相同的分片。如果分片的数据在两次搜索之间没有变化，则分片将返回缓存的聚合结果。

### Limits for long values
运行聚合时，Elasticsearch 使用 double 值来保存和表示数字数据。因此大于 2^53 的长整数上的聚合是近似的。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-aggregations.html
