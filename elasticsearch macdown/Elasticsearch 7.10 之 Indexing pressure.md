## Elasticsearch 7.10 之 Indexing pressure


将文档索引到 Elasticsearch 中会以内存和 CPU 负载的形式引入系统负载。每个索引操作都包括协调，主要和复制阶段。这些阶段可以跨集群中的多个节点执行。

索引压力可以通过外部操作（例如索引请求）或内部机制（例如恢复和跨集群复制）来产生。如果将过多的索引工作引入系统，则集群可能会变得饱和。这可能会对其他操作产生不利影响，例如搜索，群集协调和后台处理。

为避免这些问题，Elasticsearch 在内部监视索引负载。当负载超过某些限制时，新的索引工作将被拒绝

### Indexing stages

外部索引操作经历三个阶段：协调，主索引和副本。请参阅 [Basic write model](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-replication.html#basic-write-model)。

### Memory limits

**indexing_pressure.memory.limit** 节点设置限制可用于未完成的索引请求的字节数。此设置默认为堆的 10％。

在每个索引阶段的开始，Elasticsearch 会考虑索引编制请求所消耗的字节。仅在索引阶段结束时才发布此统计。这意味着上游阶段将占据资源，直到所有下游阶段完成。例如，在主阶段和副本阶段完成之前，协调请求将一直负责。在必要时，将一直处理主要请求，直到每个同步副本都响应以启用副本重试为止。

当未完成的协调索引，主索引和副本索引字节的数量超过配置的限制时，节点将在协调或主阶段开始拒绝新的索引工作。

当未完成的副本索引字节数超过配置的限制的 1.5 倍时，节点将在副本阶段开始拒绝新的索引工作。这种设计意味着随着索引压力在节点上产生，它们自然会停止接受协调和主要工作，而转而支持未完成的复本工作。

indexing_pressure.memory.limit 设置的 10％ 默认限制大小合适。您应该在仔细考虑后再进行更改。仅索引请求会导致此限制。这意味着存在额外的索引开销（缓冲区，侦听器等），这些开销也需要堆空间。  Elasticsearch 的其他组件也需要内存。将该限制设置得太高会拒绝其他操作和组件的运行内存。

### Monitoring

您可以使用 [node stats API](https://www.elastic.co/guide/en/elasticsearch/reference/current/cluster-nodes-stats.html#cluster-nodes-stats-api-response-body-indexing-pressure) 检索索引压力指标。

### Indexing pressure settings

**indexing_pressure.memory.limit** ：索引请求可能消耗的未完成字节数。当达到或超过此限制时，节点将拒绝新的协调和主要操作。当副本操作消耗此限制的 1.5 倍时，该节点将拒绝新的副本操作。默认为堆的 10％ 。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-indexing-pressure.html
