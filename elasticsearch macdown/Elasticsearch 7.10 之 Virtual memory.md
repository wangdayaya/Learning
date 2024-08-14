## Elasticsearch 7.10 之 Virtual memory



Elasticsearch 默认使用 **mmapfs** 目录存储其索引。 默认的操作系统对 mmap 计数的限制可能太低，这可能会导致内存不足异常。

在 Linux 上，您可以通过以 root 用户身份运行以下命令来增加限制：

	sysctl -w vm.max_map_count=262144

要永久设置此值，请更新 **/etc/sysctl.conf** 中的**vm.max_map_count** 设置。 要在重新引导后进行验证，请运行 **sysctl vm.max_map_count** 。

RPM 和 Debian 软件包将自动配置此设置。 不需要进一步的配置。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/vm-max-map-count.html
