## Elasticsearch 7.10 之 Suggesters

通过使用 suggester ，根据提供的文本来建议外观相似的字词。建议功能的某些部分仍在开发中。

	POST my-index-000001/_search
	{
	  "query" : {
	    "match": {
	      "message": "tring out Elasticsearch"
	    }
	  },
	  "suggest" : {
	    "my-suggestion" : {
	      "text" : "tring out Elasticsearch",
	      "term" : {
	        "field" : "message"
	      }
	    }
	  }
	}
 
### Request

提示功能通过使用 suggester 来根据提供的文本提示外观相似的术语。请求部分在 \_search 请求中与查询部分一起定义。如果查询部分被忽略，则仅返回建议。

NOET: \_suggest 已被弃用，转而建议通过 _search 端点使用建议功能。在 5.0 中，_search 端点已针对仅建议搜索请求进行了优化。

### Examples

每个请求可以指定几个建议。每个建议都以任意名称标识。在下面的示例中，请求了两个建议。 my-suggest-1 和 my-suggest-2 建议都使用term suggester ，但 text 字段不同。

	POST _search
	{
	  "suggest": {
	    "my-suggest-1" : {
	      "text" : "tring out Elasticsearch",
	      "term" : {
	        "field" : "message"
	      }
	    },
	    "my-suggest-2" : {
	      "text" : "kmichy",
	      "term" : {
	        "field" : "user.id"
	      }
	    }
	  }
	}
 
以下响应示例包括对 my-suggest-1 和 my-suggest-2 的建议响应。每个建议部分都包含条目。每个条目实际上是来自建议文本的一个分词，并且包含建议条目文本、建议文本中的原始起始偏移量、长度以及（如果找到）任意数量的选项。

	{
	  "_shards": ...
	  "hits": ...
	  "took": 2,
	  "timed_out": false,
	  "suggest": {
	    "my-suggest-1": [ {
	      "text": "tring",
	      "offset": 0,
	      "length": 5,
	      "options": [ {"text": "trying", "score": 0.8, "freq": 1 } ]
	    }, {
	      "text": "out",
	      "offset": 6,
	      "length": 3,
	      "options": []
	    }, {
	      "text": "elasticsearch",
	      "offset": 10,
	      "length": 13,
	      "options": []
	    } ],
	    "my-suggest-2": ...
	  }
	}
	
每个选项数组都包含一个选项对象，该选项对象包含被推荐的文本、其文档频率和与建议输入文本相比的得分。分数的含义取决于所使用的 suggester 。term suggester 的得分是基于编辑距离的。

##### Global suggest text
为了避免重复推荐的文本，可以定义全局文本。在下面的示例中，建议文本是全局的，并且适用于 my-suggest-1 和 my-suggest-2 建议。

	POST _search
	{
	  "suggest": {
	    "text" : "tring out Elasticsearch",
	    "my-suggest-1" : {
	      "term" : {
	        "field" : "message"
	      }
	    },
	    "my-suggest-2" : {
	       "term" : {
	        "field" : "user"
	       }
	    }
	  }
	}
 
在上面的示例中，推荐文本也可以被指定为建议特定选项。在建议级别指定的建议文本将覆盖全局的建议文本。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-suggesters.html#phrase-suggester
