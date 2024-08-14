## Elasticsearch 7.10 之 Slow Log

### Search Slow Log

分片级别的慢速搜索日志允许将慢速搜索（查询和提取阶段）记录到专用日志文件中。

可以为执行的查询阶段和获取阶段设置阈值，这里是一个示例：

	index.search.slowlog.threshold.query.warn: 10s
	index.search.slowlog.threshold.query.info: 5s
	index.search.slowlog.threshold.query.debug: 2s
	index.search.slowlog.threshold.query.trace: 500ms
	
	index.search.slowlog.threshold.fetch.warn: 1s
	index.search.slowlog.threshold.fetch.info: 800ms
	index.search.slowlog.threshold.fetch.debug: 500ms
	index.search.slowlog.threshold.fetch.trace: 200ms
	
	index.search.slowlog.level: info
	
以上所有设置都是动态的，可以使用 **[update indices settings API](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-update-settings.html)** 为每个索引进行设置。例如：

	PUT /my-index-000001/_settings
	{
	  "index.search.slowlog.threshold.query.warn": "10s",
	  "index.search.slowlog.threshold.query.info": "5s",
	  "index.search.slowlog.threshold.query.debug": "2s",
	  "index.search.slowlog.threshold.query.trace": "500ms",
	  "index.search.slowlog.threshold.fetch.warn": "1s",
	  "index.search.slowlog.threshold.fetch.info": "800ms",
	  "index.search.slowlog.threshold.fetch.debug": "500ms",
	  "index.search.slowlog.threshold.fetch.trace": "200ms",
	  "index.search.slowlog.level": "info"
	}
 
默认情况下，均未启用（设置为 -1 ）。级别（ **warn**, **info**, **debug**, **trace** ）允许控制将在哪个级别下记录日志。不需要全部配置（例如，只设置 **warn** 阈值）。多个级别的好处是能够针对违反的特定阈值快速 “grep”。

日志记录在分片级别范围内完成，这意味着在特定分片内执行搜索请求。它不包含整个搜索请求，可以将其广播到多个分片以便执行。分片级别日志记录的一些好处是，与请求级别相比，特定机器上实际执行的关联。

默认情况下，使用以下配置（在 **log4j2.properties** 中找到）配置日志文件：

	appender.index_search_slowlog_rolling.type = RollingFile
	appender.index_search_slowlog_rolling.name = index_search_slowlog_rolling
	appender.index_search_slowlog_rolling.fileName = ${sys:es.logs.base_path}${sys:file.separator}${sys:es.logs.cluster_name}_index_search_slowlog.log
	appender.index_search_slowlog_rolling.layout.type = PatternLayout
	appender.index_search_slowlog_rolling.layout.pattern = [%d{ISO8601}][%-5p][%-25c] [%node_name]%marker %.-10000m%n
	appender.index_search_slowlog_rolling.filePattern = ${sys:es.logs.base_path}${sys:file.separator}${sys:es.logs.cluster_name}_index_search_slowlog-%i.log.gz
	appender.index_search_slowlog_rolling.policies.type = Policies
	appender.index_search_slowlog_rolling.policies.size.type = SizeBasedTriggeringPolicy
	appender.index_search_slowlog_rolling.policies.size.size = 1GB
	appender.index_search_slowlog_rolling.strategy.type = DefaultRolloverStrategy
	appender.index_search_slowlog_rolling.strategy.max = 4
	
	logger.index_search_slowlog_rolling.name = index.search.slowlog
	logger.index_search_slowlog_rolling.level = trace
	logger.index_search_slowlog_rolling.appenderRef.index_search_slowlog_rolling.ref = index_search_slowlog_rolling
	logger.index_search_slowlog_rolling.additivity = false
	
**Identifying search slow log origin**

识别触发缓慢运行查询的原因通常很有用。如果使用 **X-Opaque-ID** 标头发起了呼叫，则用户 ID 将作为附加 ID 字段包含在 Search Slow 日志中（向右滚动）。

	[2030-08-30T11:59:37,786][WARN ][i.s.s.query              ] [node-0] [index6][0] took[78.4micros], took_millis[0], total_hits[0 hits], stats[], search_type[QUERY_THEN_FETCH], total_shards[1], source[{"query":{"match_all":{"boost":1.0}}}], id[MY_USER_ID],
用户 ID 也包含在 JSON 日志中。

	{
	  "type": "index_search_slowlog",
	  "timestamp": "2030-08-30T11:59:37,786+02:00",
	  "level": "WARN",
	  "component": "i.s.s.query",
	  "cluster.name": "distribution_run",
	  "node.name": "node-0",
	  "message": "[index6][0]",
	  "took": "78.4micros",
	  "took_millis": "0",
	  "total_hits": "0 hits",
	  "stats": "[]",
	  "search_type": "QUERY_THEN_FETCH",
	  "total_shards": "1",
	  "source": "{\"query\":{\"match_all\":{\"boost\":1.0}}}",
	  "id": "MY_USER_ID",
	  "cluster.uuid": "Aq-c-PAeQiK3tfBYtig9Bw",
	  "node.id": "D7fUYfnfTLa2D7y-xw6tZg"
	}
	
### Index Slow log

索引慢日志，功能类似于搜索慢日志。日志文件名以 **\_index\_indexing\_slowlog.log** 结尾。日志和阈值的配置方式与搜索慢速日志相同。索引慢日志示例：

	index.indexing.slowlog.threshold.index.warn: 10s
	index.indexing.slowlog.threshold.index.info: 5s
	index.indexing.slowlog.threshold.index.debug: 2s
	index.indexing.slowlog.threshold.index.trace: 500ms
	index.indexing.slowlog.level: info
	index.indexing.slowlog.source: 1000
以上所有设置都是动态的，可以使用 **update indices settings** API 为每个索引进行设置。例如：

	PUT /my-index-000001/_settings
	{
	  "index.indexing.slowlog.threshold.index.warn": "10s",
	  "index.indexing.slowlog.threshold.index.info": "5s",
	  "index.indexing.slowlog.threshold.index.debug": "2s",
	  "index.indexing.slowlog.threshold.index.trace": "500ms",
	  "index.indexing.slowlog.level": "info",
	  "index.indexing.slowlog.source": "1000"
	}
 
默认情况下，Elasticsearch 将在慢日志中记录 \_source 的前 1000 个字符。您可以使用 **index.indexing.slowlog.source** 进行更改。将其设置为 **false** 或 **0** 将完全跳过对源的日志记录，将其设置为 **true** 将不考虑大小而记录整个源。默认情况下，原始 **_source** 会重新格式化，以确保它适合单个日志行。如果保留原始文档格式很重要，则可以通过将 **index.indexing.slowlog.reformat** 设置为 **false** 来关闭重新格式化，这将导致源按“原样”记录，并可能跨越多个日志行。

默认情况下，在 **log4j2.properties** 文件中配置索引慢日志文件：

	appender.index_indexing_slowlog_rolling.type = RollingFile
	appender.index_indexing_slowlog_rolling.name = index_indexing_slowlog_rolling
	appender.index_indexing_slowlog_rolling.fileName = ${sys:es.logs.base_path}${sys:file.separator}${sys:es.logs.cluster_name}_index_indexing_slowlog.log
	appender.index_indexing_slowlog_rolling.layout.type = PatternLayout
	appender.index_indexing_slowlog_rolling.layout.pattern = [%d{ISO8601}][%-5p][%-25c] [%node_name]%marker %.-10000m%n
	appender.index_indexing_slowlog_rolling.filePattern = ${sys:es.logs.base_path}${sys:file.separator}${sys:es.logs.cluster_name}_index_indexing_slowlog-%i.log.gz
	appender.index_indexing_slowlog_rolling.policies.type = Policies
	appender.index_indexing_slowlog_rolling.policies.size.type = SizeBasedTriggeringPolicy
	appender.index_indexing_slowlog_rolling.policies.size.size = 1GB
	appender.index_indexing_slowlog_rolling.strategy.type = DefaultRolloverStrategy
	appender.index_indexing_slowlog_rolling.strategy.max = 4
	
	logger.index_indexing_slowlog.name = index.indexing.slowlog.index
	logger.index_indexing_slowlog.level = trace
	logger.index_indexing_slowlog.appenderRef.index_indexing_slowlog_rolling.ref = index_indexing_slowlog_rolling
	logger.index_indexing_slowlog.additivity = false

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-slowlog.html 
