## Elasticsearch 7.10 之 Java High Level REST Client

The Java High Level REST Client 在 Java Low Level REST client 之上工作。 它的主要目的是公开 API 特定的方法，这些方法接受请求对象作为参数并返回响应对象，以便由客户端本身来处理请求和响应。

每个 API 可以同步或异步调用。 同步方法返回一个响应对象，而名称以 async 后缀结尾的异步方法则需要一个侦听器参数，一旦接收到响应或错误，该参数就会被通知（在低级客户端管理的线程池上）。

Java High Level REST Client 取决于 Elasticsearch 核心项目。 它接受与 TransportClient 相同的请求参数，并返回相同的响应对象。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-high.html
