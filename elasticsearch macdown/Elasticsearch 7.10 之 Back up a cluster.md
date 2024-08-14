## Elasticsearch 7.10 之 Back up a cluster

您不能仅通过复制其所有节点的数据目录来备份 Elasticsearch 集群。 Elasticsearch 可能正在运行时对其数据目录的内容进行更改；复制其数据目录不能期望捕获其内容的一致。如果尝试从此类备份还原群集，则该群集可能会失败并报告损坏和/或丢失文件。另外，它或许成功了，但它默默地丢失了一些数据。备份集群的唯一可靠方法是使用快照和还原功能。

要为您的群集进行完整备份：

* 备份资料
* 备份集群配置
* 备份安全配置

要从备份还原群集：

* 恢复数据
* 恢复安全配置

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/backup-cluster.html
