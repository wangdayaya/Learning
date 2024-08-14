##Elasticsearch 7.10 之 Important System Configuration


理想情况下，Elasticsearch 应该在服务器上单独运行并使用所有可用资源。为此您需要配置您的操作系统，以允许运行 Elasticsearch 的用户访问比默认允许更多的资源。

进入生产之前，必须考虑以下设置：

* Disable swapping
* Increase file descriptors
* Ensure sufficient virtual memory
* Ensure sufficient threads
* JVM DNS cache settings
* Temporary directory not mounted with noexec
* TCP retransmission timeout

### Development mode vs production mode



默认情况下，Elasticsearch 假定您正在开发模式下工作。如果以上任何设置的配置都不正确，将在日志文件中写入警告，但是您将能够启动和运行 Elasticsearch 节点。

一旦您配置了诸如 **network.host** 之类的网络设置，Elasticsearch 就会假定您即将投入生产，并将上述警告升级为异常。这些异常将阻止您的 Elasticsearch 节点启动。这是一项重要的安全措施，以确保不会因服务器配置错误而丢失数据。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/system-config.html

