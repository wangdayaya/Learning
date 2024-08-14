## Elasticsearch 7.10 之 Compatibility

The Java High Level REST Client 至少需要 Java 1.8，并依赖于 Elasticsearch 核心项目。客户端版本与为其开发客户端的 Elasticsearch 版本相同。它接受与 TransportClient 相同的请求参数，并返回相同的响应对象。如果需要将应用程序从 TransportClient 迁移到新的 REST 客户端，请参阅 [Migration Guide](https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-high-level-migration.html) 。

确保高级客户端能够与在相同主要版本和较大或相等的次要版本上运行的任何 Elasticsearch 节点进行通信。它不需要与其通信的 Elasticsearch 节点使用相同的次要版本，因为它具有前向兼容性，这意味着它支持与比其开发的版本更高的 Elasticsearch 通信。

6.0 客户端可以与任何 6.x Elasticsearch 节点进行通信，而 6.1 客户端可以与 6.1、6.2 和任何更高版本的 6.x 版本进行通信，但是与先前的 Elasticsearch 节点进行通信时可能会出现不兼容问题如果 6.1 客户端支持 6.0 节点不知道的某些 API 的新请求正文字段，则版本介于 6.1 和 6.0 之间。

建议将 Elasticsearch 集群升级到新的主要版本时升级 High Level Client ，因为 REST API 的重大更改可能会导致意外结果，具体取决于请求所命中的节点，并且新添加的 API 仅受支持。客户端的较新版本。一旦集群中的所有节点都已升级到新的主要版本，客户端应始终最后更新。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-high-compatibility.html
