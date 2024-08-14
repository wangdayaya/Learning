## Elasticsearch 7.10 之 Similarity module

相似度（评分/排名模型）定义了匹配文档的评分方式。每个字段具有相似性，这意味着可以通过映射为每个字段定义不同的相似性。

配置自定义相似性被认为是专家功能，并且内置相似性很可能足以满足相似性中的描述。

### Configuring a similarity

大多数现有或自定义相似性都有配置选项，可以通过如下所示的索引设置进行配置。创建索引或更新索引设置时可以提供索引选项。

	PUT /index
	{
	  "settings": {
	    "index": {
	      "similarity": {
	        "my_similarity": {
	          "type": "DFR",
	          "basic_model": "g",
	          "after_effect": "l",
	          "normalization": "h2",
	          "normalization.h2.c": "3.0"
	        }
	      }
	    }
	  }
	}
 
在这里，我们配置 DFR 相似度，以便在映射中将其称为**my_similarity** ，如以下示例所示：

	PUT /index/_mapping
	{
	  "properties" : {
	    "title" : { "type" : "text", "similarity" : "my_similarity" }
	  }
	}
 
### Available similarities

**BM25 similarity (default)**

基于 TF/IDF 的相似性具有内置的 tf 规范化功能，应该适用于短字段（例如名称）。有关更多详细信息，请参见 [Okapi_BM25](https://en.wikipedia.org/wiki/Okapi_BM25) 。类型名称：**BM25** 。这种相似性具有以下选项：

* **k1** ：控制非线性项频率归一化（饱和）。默认值为 **1.2** 
* **b** ：控制文档长度将 **tf** 值归一化的程度。默认值为 **0.75**
* **discount_overlaps** ：确定在计算范数时是否忽略重叠标记（位置增量为 0 的标记）。默认情况下为 **true** ，这意味着重叠令牌在计算规范时不计算在内


**DFR similarity**

实现与随机性框架的差异的相似性。类型名称：**DFR** 。这种相似性具有以下选项：

* **basic_model** ：可能的值：**g** ，**if** ，**in** 和 **ine**
* **after_effect** ：可能的值：**b** 和 **l**
* **normalization** ：可能的值：**no** ，**h1** ，**h2** ，**h3** 和 **z**

除第一个选项外，所有选项都需要标准化值。



**DFI similarity**

实现独立性模型差异的相似性。类型名称：**DFI** ，这种相似性具有以下选项：
 
* **independence_measure** ：可能的值是 **standardized** ，**saturated** ，**chisquared**

使用这种相似性时，强烈建议不要删除停用词以取得良好的相关性。另请注意，频率低于预期频率的字词的得分将等于 0 。


**IB similarity**

基于信息的模型。该算法基于以下概念：任何符号分发序列中的信息内容主要取决于其基本元素的重复使用。对于书面文本而言，这一挑战将对应于比较不同作者的写作风格。类型名称：**IB** 。这种相似性具有以下选项：

* **distribution** ：可能的值：ll 和 spl。
* **lambda** ：可能的值：df 和 ttf。
* **normalization** ：与 DFR 相似度相同。



**LM Dirichlet similarity**

LM Dirichlet 相似度。类型名称：**LMDirichlet** 。这种相似性具有以下选项：

* **mu** ：默认为 2000

本文中的计分公式为出现次数少于语言模型预测的次数的词汇分配了负分数，这对 Lucene 是非法的，因此此类词汇的分数为 0 。


**LM Jelinek Mercer similarity**

LM Jelinek Mercer 相似度。该算法尝试捕获文本中的重要模式，同时保留噪声。类型名称：**LMJelinekMercer** 。这种相似性具有以下选项：

* **lambda** ：最佳值取决于集合和查询。标题查询的最佳值约为 0.1 ，长查询的最佳值为 0.7 。默认值为 0.1 。当值接近 0 时，匹配更多查询词的文档将比匹配较少词的文档排名更高。


**Scripted similarity**

一种相似性，使您可以使用脚本来指定应如何计算分数。类型名称：**scripted** 。例如，以下示例显示了如何重新实现 TF-IDF ：

	PUT /index
	{
	  "settings": {
	    "number_of_shards": 1,
	    "similarity": {
	      "scripted_tfidf": {
	        "type": "scripted",
	        "script": {
	          "source": "double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; double norm = 1/Math.sqrt(doc.length); return query.boost * tf * idf * norm;"
	        }
	      }
	    }
	  },
	  "mappings": {
	    "properties": {
	      "field": {
	        "type": "text",
	        "similarity": "scripted_tfidf"
	      }
	    }
	  }
	}
	
	PUT /index/_doc/1
	{
	  "field": "foo bar foo"
	}
	
	PUT /index/_doc/2
	{
	  "field": "bar baz"
	}
	
	POST /index/_refresh
	
	GET /index/_search?explain=true
	{
	  "query": {
	    "query_string": {
	      "query": "foo^1.7",
	      "default_field": "field"
	    }
	  }
	}
 
产生：

	{
	  "took": 12,
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
	    "max_score": 1.9508477,
	    "hits": [
	      {
	        "_shard": "[index][0]",
	        "_node": "OzrdjxNtQGaqs4DmioFw9A",
	        "_index": "index",
	        "_type": "_doc",
	        "_id": "1",
	        "_score": 1.9508477,
	        "_source": {
	          "field": "foo bar foo"
	        },
	        "_explanation": {
	          "value": 1.9508477,
	          "description": "weight(field:foo in 0) [PerFieldSimilarity], result of:",
	          "details": [
	            {
	              "value": 1.9508477,
	              "description": "score from ScriptedSimilarity(weightScript=[null], script=[Script{type=inline, lang='painless', idOrCode='double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; double norm = 1/Math.sqrt(doc.length); return query.boost * tf * idf * norm;', options={}, params={}}]) computed from:",
	              "details": [
	                {
	                  "value": 1.0,
	                  "description": "weight",
	                  "details": []
	                },
	                {
	                  "value": 1.7,
	                  "description": "query.boost",
	                  "details": []
	                },
	                {
	                  "value": 2,
	                  "description": "field.docCount",
	                  "details": []
	                },
	                {
	                  "value": 4,
	                  "description": "field.sumDocFreq",
	                  "details": []
	                },
	                {
	                  "value": 5,
	                  "description": "field.sumTotalTermFreq",
	                  "details": []
	                },
	                {
	                  "value": 1,
	                  "description": "term.docFreq",
	                  "details": []
	                },
	                {
	                  "value": 2,
	                  "description": "term.totalTermFreq",
	                  "details": []
	                },
	                {
	                  "value": 2.0,
	                  "description": "doc.freq",
	                  "details": []
	                },
	                {
	                  "value": 3,
	                  "description": "doc.length",
	                  "details": []
	                }
	              ]
	            }
	          ]
	        }
	      }
	    ]
	  }
	}
	
WARNING：尽管脚本相似性提供了很大的灵活性，但它们需要满足一组规则。否则可能会导致 Elasticsearch 默默返回错误的热门匹配，或者在搜索时因内部错误而失败：

* 返回的分数必须为正。
* 所有其他变量保持相等，当 doc.freq 增加时，分数不得降低。
* 所有其他变量保持相等，当 doc.length 增加时分数不得增加。

您可能已经注意到，上述脚本的很大一部分取决于每个文档都相同的统计信息。通过提供 **weight_script** 可以使上面的代码稍微更有效，它可以计算分数的文档无关部分，并且可以在 **weight** 变量下使用。如果未提供 **weight_script** ，则 **weight** 等于 **1** 。**weight_script** 可以访问与脚本相同的变量，但 **doc** 除外，因为它应该计算与分数无关的文档。

下面的配置将给出相同的 tf-idf 分数，但效率更高：

	PUT /index
	{
	  "settings": {
	    "number_of_shards": 1,
	    "similarity": {
	      "scripted_tfidf": {
	        "type": "scripted",
	        "weight_script": {
	          "source": "double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; return query.boost * idf;"
	        },
	        "script": {
	          "source": "double tf = Math.sqrt(doc.freq); double norm = 1/Math.sqrt(doc.length); return weight * tf * norm;"
	        }
	      }
	    }
	  },
	  "mappings": {
	    "properties": {
	      "field": {
	        "type": "text",
	        "similarity": "scripted_tfidf"
	      }
	    }
	  }
	}
 


**Default Similarity**

默认情况下，Elasticsearch 将使用任何配置为默认的相似性。

创建索引时，可以更改索引中所有字段的默认相似性：

	PUT /index
	{
	  "settings": {
	    "index": {
	      "similarity": {
	        "default": {
	          "type": "boolean"
	        }
	      }
	    }
	  }
	}
 
如果要在创建索引后更改默认相似性，则必须关闭索引，发送以下请求，然后再次打开它：

	POST /index/_close
	
	PUT /index/_settings
	{
	  "index": {
	    "similarity": {
	      "default": {
	        "type": "boolean"
	      }
	    }
	  }
	}
	
	POST /index/_open

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html
`