## Elasticsearch 7.10 之 Index-level shard allocation filtering



您可以使用分片分配过滤器来控制 Elasticsearch 在何处分配特定索引的分片。这每个索引过滤器与集群范围的分配过滤和 [allocation awareness](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-cluster.html#shard-allocation-awareness) 结合使用。

分片分配过滤器可以基于自定义节点属性或内置的 \_name, \_host\_ip, \_publish\_ip, \_ip, \_host, \_id, \_tier 和 \_tier_preference 属性。 [Index lifecycle management](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html) 使用基于自定义节点属性的过滤器来确定在阶段之间移动时如何重新分配分片。

**cluster.routing.allocation** 设置是动态的，可将活着的索引从一组节点移动到另一组节点。仅在可能的情况下重新分配分片，而又不会破坏另一个路由约束，例如从不将主分片和副本分片分配在同一节点上。

例如，您可以使用自定义节点属性来指示节点的性能特征，并使用分片分配过滤将特定索引的分片路由到最合适的硬件类别。

### Enabling index-level shard allocation filtering

要基于定制节点属性进行过滤：

1. 在每个节点的 elasticsearch.yml 配置文件中使用自定义节点属性指定过滤器特征。例如，如果您有 **small** , **medium** 和 **big** 节点，则可以添加一个 **size** 属性以根据节点大小进行过滤。

		node.attr.size: medium
您还可以在启动节点时设置自定义属性：

		`./bin/elasticsearch -Enode.attr.size=medium
2. 向索引添加路由分配过滤器。 **index.routing.allocation** 设置支持三种类型的过滤器：**include** , **exclude** 和 **require** 。例如，要告诉 Elasticsearch 将 **test** 索引中的分片分配给 **big** 或 **medium** 节点，请使用 **index.routing.allocation.include** ：

		PUT test/_settings
		{
		  "index.routing.allocation.include.size": "big,medium"
		}
 
	如果您指定多个过滤器，则节点必须同时满足以下条件，才能将分片重定位到该过滤器：

* 	如果指定了任何 **require** 类型条件，则必须全部满足
* 	如果指定了任何 **exclude** 类型条件，则可能都不满足
* 	如果指定了任何 **include** 类型条件，则必须至少满足其中一个条件
  
	例如，要将 **test** 索引移动到 **rack1** 中的 **big** 节点，可以指定：
	
		PUT test/_settings
		{
		  "index.routing.allocation.require.size": "big",
		  "index.routing.allocation.require.rack": "rack1"
		}
 
### Index allocation filter settings

 **index.routing.allocation.include.{attribute}**
 : 将索引分配给其 {attribute} 具有至少一个逗号分隔值的节点。
 
 **index.routing.allocation.require.{attribute}**: 将索引分配给其 {attribute} 具有所有逗号分隔值的节点。
 
**index.routing.allocation.exclude.{attribute}**: 将索引分配给其 {attribute} 没有任何逗号分隔值的节点。

索引分配设置支持以下内置属性：

_name: 通过节点名称匹配节点

\_host\_ip: 通过主机 IP 地址（与主机名关联的 IP ）匹配节点

\_publish\_ip: 通过发布的 IP 地址匹配节点

\_ip: 匹配 \_host\_ip 或 \_publish\_ip

_host: 通过主机名匹配节点

_id: 通过节点 ID 匹配节点

_tier: 通过节点的数据层角色来匹配节点。有关更多详细信息，请参见 [data tier allocation filtering](https://www.elastic.co/guide/en/elasticsearch/reference/current/data-tier-shard-filtering.html)

NOTE: _tier 过滤基于节点角色。角色的一个子集是数据层角色，并且通用数据角色将匹配任何层过滤。

指定属性值时，可以使用通配符，例如：

	PUT test/_settings
	{
	  "index.routing.allocation.include._ip": "192.168.2.*"
	}

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-allocation-filtering.html
