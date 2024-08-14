## Elasticsearch 7.10 之 Designing for resilience

诸如 Elasticsearch 之类的分布式系统，旨在即使它们的某些组件出现故障也可以继续工作。只要有足够的连接良好的节点来接管其职责，如果 Elasticsearch 群集的某些节点不可用或已断开连接，它们就可以继续正常运行。

弹性群集的大小有一个限制。所有 Elasticsearch 集群都需要：

* 一个选出的 [master node](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/modules-discovery-quorums.html)  节点
* 每个 [role](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/modules-node.html) 至少有一个节点。
* 每个 [shard](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/scalability.html) 至少有一个副本。

弹性集群需要每个所需集群组件的冗余。这意味着弹性群集必须具有：

* 至少三个主节点
* 每个角色至少两个节点
* 每个分片至少有两个副本（一个主副本和一个或多个副本）

一个有弹性的集群需要三个符合主机资格的节点，因此，如果其中一个节点发生故障，那么其余两个节点仍占多数，并且可以举行一次成功的选举。

同样，每个角色的节点的冗余意味着如果某个特定角色的节点发生故障，则另一个节点可以承担其责任。

最后，一个弹性集群应给每个分片至少分配两个副本。如果一个副本失败，则应该再接一个好副本。 Elasticsearch 会在其余节点上自动重建任何失败的分片副本，以便在发生故障后将集群恢复到完全运行状况。

故障会暂时减少集群的总容量。此外，发生故障后，集群必须执行其他后台活动才能使其恢复健康。您应该确保即使某些节点发生故障，集群也具有处理工作量的能力。

根据您的需求和预算，Elasticsearch 集群可以包含一个节点，数百个节点或两者之间的任意数量。设计较小的群集时，通常应集中精力使其对单节点故障具有弹性。大型群集的设计者还必须考虑多个节点同时发生故障的情况。以下页面为构建各种规模的弹性集群提供了一些建议：

* [Resilience in small clusters](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/high-availability-cluster-small-clusters.html)
* [Resilience in larger clusters](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/high-availability-cluster-design-large-clusters.html)

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/high-availability-cluster-design.html
