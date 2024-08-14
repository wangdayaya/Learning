## Elasticsearch 7.10 之 Index Shard Allocation



此模块提供每个索引的设置，以控制分片到节点的分配：

* [Shard allocation filtering](https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-allocation-filtering.html) ：控制将哪些分片分配给哪些节点。
* [Delayed allocation](https://www.elastic.co/guide/en/elasticsearch/reference/current/delayed-allocation.html) ：由于节点离开而导致延迟分配未分配的分片。
* [Total shards per node](https://www.elastic.co/guide/en/elasticsearch/reference/current/allocation-total-shards.html) ：每个节点的相同索引的分片数量的硬限制。
* [Data tier allocation](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tier-shard-filtering.html) ：控制索引到 [data tiers](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html) 的分配。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-allocation.html
