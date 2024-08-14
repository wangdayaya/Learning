## Elasticsearch 7.10 之 Index-level data tier allocation filtering

您可以使用索引级别的分配设置来控制将索引分配到的 [data tier](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tiers.html) 。数据层分配器是一个 [shard allocation filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-allocation-filtering.html) ，它使用两个内置节点属性：**_tier** 和 **\_tier\_preference** 。

这些层属性是使用数据节点角色设置的：

* [data_content](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-node.html#data-content-node)
* [data_hot](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-node.html#data-hot-node)
* [data_warm](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-node.html#data-warm-node)
* [data_cold](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-node.html#data-cold-node)

NOTE: 数据角色不是有效的数据层，不能用于数据层过滤。

### Data tier allocation settingse

 **index.routing.allocation.include._tier**: 将索引分配给其 **node.roles** 配置至少具有逗号分隔值之一的节点
 
 **index.routing.allocation.require._tier**: 将索引分配给其 **node.roles** 配置具有所有逗号分隔值的节点

 **index.routing.allocation.exclude._tier**: 将索引分配给其 **node.roles** 配置没有任何逗号分隔值的节点
 
 **index.routing.allocation.include.\_tier\_preference**: 
将索引分配给列表中具有可用节点的第一层。如果首选层中没有可用的节点，这可以防止索引保持未分配状态。例如，如果将 **index.routing.allocation.include.\_tier\_preference** 设置为 **data_warm** ，**data_hot**，则在存在具有 **data_warm** 角色的节点的情况下，将索引分配给 warm tier 。如果热层中没有节点，但是有具有 **data_hot** 角色的节点，则将索引分配给 hot tier 。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tier-shard-filtering.html
