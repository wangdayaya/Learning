## Elasticsearch 7.10 之 Initialization


**RestHighLevelClient** 实例需要 [REST low-level client builder](https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-low-usage-initialization.html) 按以下方式构建：

	RestHighLevelClient client = new RestHighLevelClient(
	        RestClient.builder(
	                new HttpHost("localhost", 9200, "http"),
	                new HttpHost("localhost", 9201, "http")));
	                
高级客户端将基于提供的构建器在内部创建用于执行请求的低级客户端。该低级客户端维护一个连接池并启动一些线程，因此您应该在完全正确地使用它之后关闭高级客户端，这反过来又将关闭内部低级客户端以释放那些资源。 这可以通过关闭完成：

	client.close();
	
在本文档的其余部分中，有关 Java High Level Client 的信息，将把 **RestHighLevelClient** 实例称为客户端。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-high-getting-started-initialization.html
