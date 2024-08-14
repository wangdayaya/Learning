## Elasticsearch 7.10 之 Delaying allocation when a node leaves


当节点出于任何原因（有意或无故）离开集群时，主服务器会做出以下反应：

* 将副本分片提升为主分片，以代替所有之前在该节点上的主分片。
* 分配副本分片以替换丢失的副本（假设有足够的节点）。
* 在其余节点之间平均分配碎片。

这些操作旨在通过确保尽快完全复制每个分片来保护集群免受数据丢失。

即使我们在节点级别和集群级别都限制了并发恢复，但是这种 “shard-shuffle” 仍会给集群增加很多额外的负载，如果丢失的节点可能很快就会返回，则可能没有必要。想象一下这种情况：

* 节点 5 失去网络连接
* 主节点将副本分片提升主分片以代替之前在节点 5 上的主分片
* 主服务器将新副本分配给集群中的其他节点
* 每个新副本都会在网络上创建主分片的完整副本
* 更多分片将移至不同节点以重新平衡群集
* 几分钟后，节点 5 返回
* 主服务器通过向节点 5 分配分片来重新平衡群集

如果主服务器刚刚等待了几分钟，那么丢失的分片可能已经以最少的网络流量重新分配给了节点 5 。对于已自动同步刷新的空闲分片（未接收索引请求的分片），此过程甚至更快。

可以使用 **index.unassigned.node\_left.delayed_timeout** 动态设置（默认值为 1m ）来延迟由于节点离开而变成未分配的副本分片的分配。

可以在实时索引（或所有索引）上更新此设置：

	PUT _all/_settings
	{
	  "settings": {
	    "index.unassigned.node_left.delayed_timeout": "5m"
	  }
	}

 
启用延迟分配后，上述情况将变为：

* 节点 5 失去网络连接
* 主节点将副本分片提升主分片以代替之前在节点 5 上的主分片
* 主服务器记录一条消息，指出未分配的分片的分配已延迟，并且延迟了多长时间。
* 集群保持黄色，因为存在未分配的副本分片。
* 几分钟后，在超时到期之前，节点 5 返回。
* 丢失的副本将重新分配给节点 5（同步刷新的分片几乎立即恢复）。

	
		NOTE: 此设置将不会影响将副分片升级为主分片，也不会影响之前未分配的副本的分配。特别是，在整个集群重新启动后，延迟分配不会生效。同样，在主节点故障转移情况下，经过的延迟时间会被遗忘（即重置为完整的初始延迟）。

### Cancellation of shard relocation

如果延迟的分配超时，则主服务器将丢失的分片分配给另一个将开始恢复的节点。如果丢失的节点重新加入集群，并且其分片仍具有与主节点相同的 sync-id ，则将取消分片重定位，并将同步的分片用于恢复。

因此，默认的 timeout 设置为仅一分钟：即使分片重定位开始，取消恢复以支持同步分片也很 cheap 。


### Monitoring delayed unassigned shards

可以通过  [cluster health API](https://www.elastic.co/guide/en/elasticsearch/reference/current/cluster-health.html) 查看其因超时设置而被延迟分配的分片数量：

	GET _cluster/health    # 该请求将返回 delay_unassigned_shards 值

 

### Removing a node permanently

如果节点不打算返回，并且您希望 Elasticsearch 立即分配丢失的分片，只需将超时更新为零：

	PUT _all/_settings
	{
	  "settings": {
	    "index.unassigned.node_left.delayed_timeout": "0"
	  }
	}
 
您可以在丢失的分片开始恢复后立即重置超时。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/delayed-allocation.html
