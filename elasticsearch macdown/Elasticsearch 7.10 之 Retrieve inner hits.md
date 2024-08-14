## Elasticsearch 7.10 之 Retrieve inner hits

[parent-join](https://www.elastic.co/guide/en/elasticsearch/reference/current/parent-join.html) 和 [nested](https://www.elastic.co/guide/en/elasticsearch/reference/current/nested.html) 功能允许返回在不同范围的匹配文档。在父/子情况下，根据子文档中的匹配返回父文档，或者根据父文档中的匹配返回子文档。在嵌套的情况下，将根据嵌套内部对象中的匹配返回文档。

在这两种情况下，不同范围中的实际匹配项导致将要返回的文档被隐藏。在许多情况下，了解哪些内部嵌套对象（对于嵌套的情况）或子/父文档（对于父/子的情况）导致某些信息返回非常有用。内部命中功能可用于此目的。此功能在搜索响应中为每个搜索命中返回附加的嵌套命中，这些嵌套命中导致搜索命中匹配在不同范围内。

可以通过在嵌套的 has_child 或 has_parent 查询和过滤器上定义 inner_hits 定义来使用内部匹配。结构如下：

	"<query>" : {
	  "inner_hits" : {
	    <inner_hits_options>
	  }
	}
如果在支持它的查询上定义了 inner_hits ，则每个搜索命中将包含一个具有以下结构的 inner_hits json 对象：

	"hits": [
	  {
	    "_index": ...,
	    "_type": ...,
	    "_id": ...,
	    "inner_hits": {
	      "<inner_hits_name>": {
	        "hits": {
	          "total": ...,
	          "hits": [
	            {
	              "_type": ...,
	              "_id": ...,
	               ...
	            },
	            ...
	          ]
	        }
	      }
	    },
	    ...
	  },
	  ...
	]
### Options
内部点击数支持以下选项：


|   参数  | 说明  |
|  ----  | ----  |
|from | 从返回的常规搜索结果中每个 inner_hits 的第一个匹配获取位置开始的偏移量|
|size | 每个 inner_hits 返回的最大匹配数。默认情况下，返回前三个匹配项|
|sort | 应如何根据 inner_hits 对内部匹配进行排序。默认情况下，命中按得分排序|
|name | 响应中用于特定内部匹配定义的名称。在单个搜索请求中定义了多个内部匹配项时很有用。默认值取决于定义内部匹配的查询。对于 has_child 查询和过滤器，这是子类型； has_parent 查询和过滤器，这是父类型；嵌套查询和过滤器，这是嵌套路径。|

内部命中数还支持以下每个文档功能：

* Highlighting
* Explain
* Search fields
* Source filtering
* Script fields
* Doc value fields
* Include versions
* Include Sequence Numbers and Primary Terms


### Nested inner hits
嵌套的 inner_hits 可用于包含嵌套的内部对象作为搜索命中的内部命中。

	PUT test
	{
	  "mappings": {
	    "properties": {
	      "comments": {
	        "type": "nested"
	      }
	    }
	  }
	}
	
	PUT test/_doc/1?refresh
	{
	  "title": "Test title",
	  "comments": [
	    {
	      "author": "kimchy",
	      "number": 1
	    },
	    {
	      "author": "nik9000",
	      "number": 2
	    }
	  ]
	}
	
	POST test/_search
	{
	  "query": {
	    "nested": {
	      "path": "comments",
	      "query": {
	        "match": { "comments.number": 2 }
	      },
	      "inner_hits": {}    # 嵌套查询中的内部匹配定义。 无需定义其他选项
	    }
	  }
	}

可以从上述搜索请求中生成响应片段的示例：

	{
	  ...,
	  "hits": {
	    "total": {
	      "value": 1,
	      "relation": "eq"
	    },
	    "max_score": 1.0,
	    "hits": [
	      {
	        "_index": "test",
	        "_type": "_doc",
	        "_id": "1",
	        "_score": 1.0,
	        "_source": ...,
	        "inner_hits": {
	          "comments": {    # 搜索请求中内部匹配定义中使用的名称。 可以通过名称选项使用自定义键
	            "hits": {
	              "total": {
	                "value": 1,
	                "relation": "eq"
	              },
	              "max_score": 1.0,
	              "hits": [
	                {
	                  "_index": "test",
	                  "_type": "_doc",
	                  "_id": "1",
	                  "_nested": {
	                    "field": "comments",
	                    "offset": 1
	                  },
	                  "_score": 1.0,
	                  "_source": {
	                    "author": "nik9000",
	                    "number": 2
	                  }
	                }
	              ]
	            }
	          }
	        }
	      }
	    ]
	  }
	}


\_nested 元数据在上面的示例中至关重要，因为它定义了内部命中来自哪个内部嵌套对象。该字段定义嵌套匹配所来自的对象数组字段以及相对于其在 \_source 中的位置的偏移量。由于排序和计分，inner_hits 中命中对象的实际位置通常与嵌套内部对象的位置不同。

默认情况下，也为 inner_hits 中的命中对象返回 \_source ，但是可以更改。通过 \_source 过滤功能，可以返回或禁用源的一部分。如果在嵌套级别定义了存储的字段，则也可以通过字段功能返回这些字段。

一个重要的默认值是，inner_hits 内部的命中返回的 \_source 是相对于 _nested 元数据。因此在上面的示例中，每个嵌套的匹配仅返回注释部分，而不返回包含注释的顶级文档的整个源。

##### Nested inner hits and _source

嵌套文档没有 \_source 字段，因为整个文档源都与根文档一起存储在 \_source 字段下。为了仅包含嵌套文档的源，将分析根文档的源，并且在内部匹配中仅包含嵌套文档的相关位作为源。对每个匹配的嵌套文档执行此操作会影响执行整个搜索请求所需的时间，尤其是在将大小和内部匹配的大小设置为高于默认值时。为避免嵌套内部匹配的源代码提取相对昂贵，可以禁用源代码，而只能依靠文档值字段。像这样：
	
	PUT test
	{
	  "mappings": {
	    "properties": {
	      "comments": {
	        "type": "nested"
	      }
	    }
	  }
	}
	
	PUT test/_doc/1?refresh
	{
	  "title": "Test title",
	  "comments": [
	    {
	      "author": "kimchy",
	      "text": "comment text"
	    },
	    {
	      "author": "nik9000",
	      "text": "words words words"
	    }
	  ]
	}
	
	POST test/_search
	{
	  "query": {
	    "nested": {
	      "path": "comments",
	      "query": {
	        "match": { "comments.text": "words" }
	      },
	      "inner_hits": {
	        "_source": false,
	        "docvalue_fields": [
	          "comments.text.keyword"
	        ]
	      }
	    }
	  }
	}

### Hierarchical levels of nested object fields and inner hits

如果映射具有多层嵌套对象字段的层次，则可以通过点标记路径访问每个层次。 例如，如果有一个 comments 嵌套字段，其中包含一个 votes 嵌套字段，并且投票应直接与根匹配一起返回，则可以定义以下路径：

	PUT test
	{
	  "mappings": {
	    "properties": {
	      "comments": {
	        "type": "nested",
	        "properties": {
	          "votes": {
	            "type": "nested"
	          }
	        }
	      }
	    }
	  }
	}
	
	PUT test/_doc/1?refresh
	{
	  "title": "Test title",
	  "comments": [
	    {
	      "author": "kimchy",
	      "text": "comment text",
	      "votes": []
	    },
	    {
	      "author": "nik9000",
	      "text": "words words words",
	      "votes": [
	        {"value": 1 , "voter": "kimchy"},
	        {"value": -1, "voter": "other"}
	      ]
	    }
	  ]
	}
	
	POST test/_search
	{
	  "query": {
	    "nested": {
	      "path": "comments.votes",
	        "query": {
	          "match": {
	            "comments.votes.voter": "kimchy"
	          }
	        },
	        "inner_hits" : {}
	    }
	  }
	}

看起来像：

	{
	  ...,
	  "hits": {
	    "total" : {
	        "value": 1,
	        "relation": "eq"
	    },
	    "max_score": 0.6931471,
	    "hits": [
	      {
	        "_index": "test",
	        "_type": "_doc",
	        "_id": "1",
	        "_score": 0.6931471,
	        "_source": ...,
	        "inner_hits": {
	          "comments.votes": { 
	            "hits": {
	              "total" : {
	                  "value": 1,
	                  "relation": "eq"
	              },
	              "max_score": 0.6931471,
	              "hits": [
	                {
	                  "_index": "test",
	                  "_type": "_doc",
	                  "_id": "1",
	                  "_nested": {
	                    "field": "comments",
	                    "offset": 1,
	                    "_nested": {
	                      "field": "votes",
	                      "offset": 0
	                    }
	                  },
	                  "_score": 0.6931471,
	                  "_source": {
	                    "value": 1,
	                    "voter": "kimchy"
	                  }
	                }
	              ]
	            }
	          }
	        }
	      }
	    ]
	  }
	}
嵌套内部命中仅支持此间接引用。

### Parent/child inner hits

父/子 inner_hits 可用于包括父或子：

	PUT test
	{
	  "mappings": {
	    "properties": {
	      "my_join_field": {
	        "type": "join",
	        "relations": {
	          "my_parent": "my_child"
	        }
	      }
	    }
	  }
	}
	
	PUT test/_doc/1?refresh
	{
	  "number": 1,
	  "my_join_field": "my_parent"
	}
	
	PUT test/_doc/2?routing=1&refresh
	{
	  "number": 1,
	  "my_join_field": {
	    "name": "my_child",
	    "parent": "1"
	  }
	}
	
	POST test/_search
	{
	  "query": {
	    "has_child": {
	      "type": "my_child",
	      "query": {
	        "match": {
	          "number": 1
	        }
	      },
	      "inner_hits": {}    # 内部命中定义类似于嵌套示例中的内容
	    }
	  }
	}

可以从上述搜索请求中生成响应片段的示例：


	{
	  ...,
	  "hits": {
	    "total": {
	      "value": 1,
	      "relation": "eq"
	    },
	    "max_score": 1.0,
	    "hits": [
	      {
	        "_index": "test",
	        "_type": "_doc",
	        "_id": "1",
	        "_score": 1.0,
	        "_source": {
	          "number": 1,
	          "my_join_field": "my_parent"
	        },
	        "inner_hits": {
	          "my_child": {
	            "hits": {
	              "total": {
	                "value": 1,
	                "relation": "eq"
	              },
	              "max_score": 1.0,
	              "hits": [
	                {
	                  "_index": "test",
	                  "_type": "_doc",
	                  "_id": "2",
	                  "_score": 1.0,
	                  "_routing": "1",
	                  "_source": {
	                    "number": 1,
	                    "my_join_field": {
	                      "name": "my_child",
	                      "parent": "1"
	                    }
	                  }
	                }
	              ]
	            }
	          }
	        }
	      }
	    ]
	  }
	}

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/inner-hits.html
