## Elasticsearch 7.10 之 Search multiple data streams and indices


要搜索多个数据流和索引，请将其作为逗号分隔的值添加到搜索 API 的请求路径中。

以下请求搜索 **my-index-000001** 和 **my-index-000002** 索引。

	GET /my-index-000001,my-index-000002/_search
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
您还可以使用索引模式搜索多个数据流和索引。

以下请求针对 **my-index-*** 索引模式。该请求将搜索群集中以 **my-index-** 开头的所有数据流或索引。

	GET /my-index-*/_search
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
要搜索群集中的所有数据流和索引，请从请求路径中省略目标。或者您可以使用 _all 或 * 。

以下请求是等效的，并搜索集群中的所有数据流和索引。

	GET /_search
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
	
	GET /_all/_search
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
	
	GET /*/_search
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
### Index boost

搜索多个索引时，可以使用 **indexs_boost** 参数来提高一个或多个指定索引的结果。当来自某些索引的匹配比来自其他索引的匹配更重要时，这很有用。

NOTE: 您不能对数据流使用 indexs_boost 。

	GET /_search
	{
	  "indices_boost": [
	    { "my-index-000001": 1.4 },
	    { "my-index-000002": 1.3 }
	  ]
	}
 
索引别名和索引模式也可以使用：

	GET /_search
	{
	  "indices_boost": [
	    { "my-alias":  1.4 },
	    { "my-index*": 1.3 }
	  ]
	}
 
如果找到多个匹配项，则将使用第一个匹配项。例如，如果 alias1 中包含索引并且匹配 **my-index*** 模式，则将应用1.4的提升值。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-multiple-indices.html
