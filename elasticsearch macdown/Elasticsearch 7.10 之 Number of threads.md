## Elasticsearch 7.10 之 Number of threads



Elasticsearch 对不同类型的操作使用许多线程池。 能够在需要时创建新线程很重要。 确保 Elasticsearch 用户可以创建的线程数至少为 4096 。

这可以通过在启动 Elasticsearch 之前将 **ulimit -u 4096** 设置为 root 或在 **/etc/security/limits.conf** 中将 **nproc** 设置为 4096 来完成。

软件包分发在 **systemd** 下作为服务运行时，将自动为 Elasticsearch 进程配置线程数。 无需其他配置。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/max-number-of-threads.html
