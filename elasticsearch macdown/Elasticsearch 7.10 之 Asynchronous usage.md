## Elasticsearch 7.10 之 Asynchronous usage

跨不同客户端的所有方法都以传统的同步和异步变体形式存在。不同之处在于，异步请求使用 REST Low Level Client 中的异步请求。如果您正在执行多个请求或正在使用例如 rx java，Kotlin 协同例程或类似框架是有用的。

可以通过以下事实来识别异步方法：它们的名称中带有单词 “Async” ，并返回一个 Cancellable 实例。异步方法接受与同步变量相同的请求对象，并接受通用 ActionListener<T> ，其中 T 是同步方法的返回类型。

所有异步方法都返回一个带有 Cancel 方法的 Cancellable 对象，在您要中止请求时可以调用该方法。取消不再需要的请求是避免对 Elasticsearch 施加不必要负载的好方法。

使用 Cancellable 实例是可选的，如果不需要，可以放心地忽略它。一个用例是将其与例如 Kotlin 的 suspantCancellableCoRoutine。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-high-getting-started-asynchronous-usage.html
