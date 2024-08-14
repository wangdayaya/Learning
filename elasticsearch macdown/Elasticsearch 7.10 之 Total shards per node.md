## Elasticsearch 7.10 之 Total shards per node


集群级分片分配器尝试将单个索引的分片分布在尽可能多的节点上。但是，根据您拥有的分片和索引的数量以及它们的大小，可能并不总是能够均匀地分布分片。

以下动态设置允许您从每个节点允许的单个索引中指定分片总数的硬限制：

 **index.routing.allocation.total_shards_per_node**
将分配给单个节点的最大分片数（副本和主副本）。默认为无限。

您还可以限制节点可以拥有的分片数量，而与索引无关：

 **cluster.routing.allocation.total_shards_per_node**
（动态）分配给每个节点的主要和副本碎片的最大数量。默认为 -1（无限制）。

Elasticsearch 在分片分配期间检查此设置。例如，一个集群的 **cluster.routing.allocation.total_shards_per_node** 设置为 100 ，三个节点具有以下分片分配：

* 节点 A ：100 个分片
* 节点 B ：98 个分片
* 节点 C ：1 个分片

如果节点 C 发生故障，Elasticsearch 将其分片重新分配给节点 B
。将分片重新分配给节点 A 将超出节点 A 的分片限制。

WARNING: 这些设置施加了硬限制，这可能导致某些分片无法分配。请谨慎使用。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/allocation-total-shards.html
