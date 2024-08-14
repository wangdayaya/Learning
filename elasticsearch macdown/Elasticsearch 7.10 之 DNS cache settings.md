## Elasticsearch 7.10 之 DNS cache settings




Elasticsearch 在安装了安全管理器的情况下运行。 有了安全管理器， JVM 默认将无限期缓存正主机名解析，并且默认将负主机名解析缓存十秒钟。 Elasticsearch 使用默认值覆盖此行为，以将正向查找缓存六十秒，并将负向查找缓存十秒。 这些值应适用于大多数环境，包括 DNS 分辨率随时间变化的环境。 如果不是，则可以在  **[JVM options](https://www.elastic.co/guide/en/elasticsearch/reference/current/jvm-options.html)** 中编辑值 **es.networkaddress.cache.ttl** 和 **es.networkaddress.cache.negative.ttl** 。 请注意，除非您删除 **es.networkaddress.cache.ttl** 和 **es.networkaddress.cache.negative.ttl** ，否则 Elasticsearch 将忽略 [Java security policy](https://docs.oracle.com/javase/8/docs/technotes/guides/security/PolicyFiles.html) 中的值 **[networkaddress.cache.ttl=\<timeout>](https://docs.oracle.com/javase/8/docs/technotes/guides/net/properties.html)** 和 **[networkaddress.cache.negative.ttl=\<timeout>](https://docs.oracle.com/javase/8/docs/technotes/guides/net/properties.html)** 。 

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/networkaddress-cache-ttl.html
