## Elasticsearch 7.10 之 RequestOptions

**RestHighLevelClient** 中的所有 API 都接受一个 **RequestOptions** ，您可以使用它们以不会改变 Elasticsearch 执行请求的方式自定义请求。 例如，在这里您可以指定 NodeSelector 来控制哪个节点接收请求。有关自定义选项的更多示例，请参见 [low level client documentation](https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-low-usage-requests.html#java-rest-low-usage-request-options) 。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-high-getting-started-request-options.html
