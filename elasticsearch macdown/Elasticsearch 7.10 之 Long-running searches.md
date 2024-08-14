## Elasticsearch 7.10 之 Long-running searches

Elasticsearch 通常允许您快速搜索大量数据。 在某些情况下，搜索可能在许多分片上执行，可能针对冻结的索引并跨越多个远程群集，因此预期结果不会在毫秒内返回。 当您需要执行长时间运行的搜索时，同步等待返回结果是不理想的。 相反，异步搜索使您可以提交异步执行的搜索请求，监视请求的进度并在以后的阶段检索结果。 您也可以在部分结果可用时但在搜索完成之前检索它们。

您可以使用提交异步搜索 API 提交异步搜索请求。 使用 get async search API ，您可以监视异步搜索请求的进度并检索其结果。 正在进行的异步搜索可以通过 Delete async search API 删除。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/async-search-intro.html
