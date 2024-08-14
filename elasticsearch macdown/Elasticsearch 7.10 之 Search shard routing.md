## Elasticsearch 7.10 之 Search shard routing


为了防止硬件故障并提高搜索容量，Elasticsearch 可以跨多个节点上的多个分片存储索引数据的副本。运行搜索请求时，Elasticsearch 选择一个包含索引数据副本的节点，并将搜索请求转发到该节点的分片。此过程称为搜索分片路由或路由。

### Adaptive replica selection
默认情况下，Elasticsearch 使用自适应副本选择来路由搜索请求。此方法使用分片分配意识和以下条件选择合格的节点：

* 协调节点与合格节点之间的先前请求的响应时间
* 合格节点运行先前的搜索所花费的时间
* 合格节点的搜索线程池的队列大小

自适应副本选择旨在减少搜索延迟。但是您可以通过使用 [cluster settings API](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/cluster-update-settings.html) 将 **cluster.routing.use\_adaptive\_replica\_selection** 设置为 **false** 来禁用自适应副本选择。如果禁用，Elasticsearch 将使用循环方法路由搜索请求，这可能会导致搜索速度变慢。

### Set a preference

默认情况下，自适应副本选择从所有合格的节点和分片中进行选择。但是，您可能只希望来自本地节点的数据，或者想基于其硬件将搜索路由到特定节点。或者，您可能希望将重复搜索发送到同一分片以利用缓存。

要限制符合搜索请求条件的节点和分片集，请使用搜索 API 的 [preference](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-search.html#search-preference) 查询参数。

例如，以下请求使用 preference \_local 搜索 my-index-000001 。这将搜索限制为本地节点上的分片。如果本地节点不包含索引数据的分片副本，则请求将使用对另一个合格节点的自适应副本选择作为后备。

	GET /my-index-000001/_search?preference=_local
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
您还可以使用 **preference** 参数根据提供的字符串将搜索路由到特定的分片。如果集群状态和选定的分片没有更改，则使用相同 preference 字符串的搜索将以相同顺序路由到相同分片。

我们建议使用唯一的 preference 字符串，例如 用户名 或 Web session ID。该字符串不能以 _ 开头。

TIP: 您可以使用此选项为经常使用且占用大量资源的搜索提供缓存的结果。如果分片的数据保持不变，则使用相同的 preference 字符串进行重复搜索会从相同的分片请求缓存中检索结果。对于时间序列用例，例如日志记录，较旧索引中的数据很少更新，可以直接从此缓存中提供。

以下请求使用 preference 字符串 my-custom-shard-string 搜索 my-index-000001 索引。

	GET /my-index-000001/_search?preference=my-custom-shard-string
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
	
NOTE：如果集群状态或选定的分片发生更改，则相同的 preference 字符串可能不会以相同顺序将搜索路由到相同的分片。发生这种情况可能有多种原因，包括分片重定位和分片故障。节点还可以拒绝搜索请求， Elasticsearch 会将其重新路由到另一个节点。

### Use a routing value

为文档建立索引时，可以指定一个可选的 [routing value](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/mapping-routing-field.html) ，该值将文档路由到特定的分片。

例如，以下索引请求使用 my-routing-value 路由文档。

	POST /my-index-000001/_doc?routing=my-routing-value
	{
	  "@timestamp": "2099-11-15T13:12:00",
	  "message": "GET /search HTTP/1.1 200 1070000",
	  "user": {
	    "id": "kimchy"
	  }
	}
 
您可以在搜索 API 的路由查询参数中使用相同的路由值。这样可以确保搜索在用于索引文档的同一分片上运行。

	GET /my-index-000001/_search?routing=my-routing-value
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
您还可以提供多个逗号分隔的路由值：

	GET /my-index-000001/_search?routing=my-routing-value,my-routing-value-2
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
 
### Search concurrency and parallelism

默认情况下，Elasticsearch 不会根据请求命中的分片数量拒绝搜索请求。但是命中大量分片会大大增加 CPU 和内存使用率。

TIP：有关防止具有大量分片的索引的提示，请参阅 [Avoid oversharding](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/avoid-oversharding.html) 。

您可以使用 **max\_concurrent\_shard\_requests** 查询参数来控制搜索请求可在每个节点上命中的并发分片的最大数量。这样可以防止单个请求使集群过载。该参数默认最大为 5 。

	GET /my-index-000001/_search?max_concurrent_shard_requests=3
	{
	  "query": {
	    "match": {
	      "user.id": "kimchy"
	    }
	  }
	}
	
您还可以使用 **action.search.shard_count.limit** 群集设置来设置搜索分片限制，并拒绝命中太多分片的请求。您可以使用  [cluster settings API](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/cluster-update-settings.html) 配置 **action.search.shard_count.limit** 。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-shard-routing.html
