## Elasticsearch 7.10 之 Set up a cluster for high availability

您的数据对您很重要。确保其安全和可用对 Elasticsearch 很重要。有时，您的群集可能会遇到硬件故障或断电。为了帮助您 Elasticsearch 提供了许多功能，即使出现故障也可以实现高可用性。

* 通过适当的计划，可以设计集群有弹性以应对许多常见问题，例如单个节点的丢失或网络连接丢失，甚至区域范围内的断电（例如断电）。
* 您可以使用 [cross-cluster replication](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/xpack-ccr.html) 将数据复制到远程跟随者集群，该跟随者集群可能与领导者集群位于不同的数据中心，甚至位于不同的大陆。跟随者群集充当热备用服务器，在灾难严重到导致领导者集群发生故障的情况下，您可以进行故障转移。追随者集群还可以充当地理副本，以服务来自附近客户的搜索。
* 防止数据丢失的最后一道防线是对集群进行 [regular snapshots](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/backup-cluster.html) ，以便您可以在需要时在其他位置还原它的完整副本。



详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/high-availability.html
