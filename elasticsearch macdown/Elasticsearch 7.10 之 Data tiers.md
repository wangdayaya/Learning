## Elasticsearch 7.10 之 Data tiers

数据层是具有相同数据角色的节点的集合，这些节点通常共享相同的硬件配置文件：

* 内容层节点：处理诸如产品目录之类的内容的索引和查询负载。
* 热层节点：处理诸如日志或指标之类的时间序列数据的索引负载，并保存您最近，最常访问的数据。
* 暖层节点：保存的时间序列数据访问频率较低，并且很少需要更新。
* 冷层节点：保存时间序列数据，这些数据偶尔会被访问，并且通常不会更新。

当您将文档直接索引到特定索引时，它们会无限期地保留在内容层节点上。

当您将文档索引到数据流时，它们最初位于热层节点上。您可以配置索引生命周期管理（ILM）策略，以根据性能、弹性和数据保留要求自动通过热、暖和冷层转换时间序列数据。

节点的数据角色是在 elasticsearch.yml 中配置的。例如，可以将群集中性能最高的节点分配给热层和内容层：

	node.roles: ["data_hot", "data_content"]

### Content tier
存储在内容层中的数据通常是项目的集合，例如产品目录或文章档案。与时间序列数据不同，内容的价值在一段时间内保持相对恒定，因此，随着时间的流逝，将其转移到具有不同性能特征的层中是没有意义的。内容数据通常具有很长的数据保留要求，并且您希望能够快速检索项目，无论它们有多旧。

内容层节点通常针对查询性能进行了优化，它们将处理能力置于 IO 吞吐量之上，因此它们可以处理复杂的搜索和聚合并快速返回结果。尽管它们还负责索引编制，但通常不会以与时间序列数据（例如日志和指标）一样高的速率摄取内容数据。从弹性角度来看，该层中的索引应配置为使用一个或多个副本。

除非新索引是数据流的一部分，否则它们会自动分配给内容层。

### Hot tier
热层是时间序列数据的 Elasticsearch 入口点，并保存您最近，最频繁搜索的时间序列数据。热层中的节点在读取和写入时都需要快速，这需要更多的硬件资源和更快的存储（SSD）。为了具有弹性，应将热层中的索引配置为使用一个或多个副本。

属于数据流的新索引会自动分配给热层。

### Warm tier
一旦查询时间序列数据的频率低于热层中最近索引的数据，就可以将其移至此层。暖层通常保存最近几周的数据。仍然允许进行更新，但可能很少。通常，暖层中的节点不需要像热层中的节点一样快。为了实现弹性，应将暖层中的索引配置为使用一个或多个副本。

### Cold tier
一旦不再更新数据，它就可以从暖层移到冷层，并在余下的时间内保留下来。冷层仍然是响应查询层，但是冷层中的数据通常不会更新。随着数据过渡到冷层，可以对其进行压缩和缩小。为了具有弹性，冷层中的索引可以依赖可搜索的快照，从而无需副本。

### Data tier index allocation
创建索引时，默认情况下，Elasticsearch 将 **index.routing.allocation.include.\_tier\_preference** 设置为 **data\_content** ，以将索引分片自动分配给内容层。

当 Elasticsearch 创建索引作为数据流的一部分时，默认情况下， Elasticsearch 将 **index.routing.allocation.include.\_tier\_preference** 设置为 **data\_hot** ，以自动将索引分片分配给热层。

您可以通过在创建索引请求或与新索引匹配的索引模板中指定分片分配过滤设置来覆盖基于自动层的自动分配。

您还可以显式设置 **index.routing.allocation.include.\_tier\_preference** 以选择退出默认的基于层的分配。如果将层首选项设置为 **null** ，则 **Elasticsearch** 在分配期间将忽略数据层角色。

### Automatic data tier migration
ILM 使用迁移操作自动在可用数据层之间过渡托管索引。默认情况下，此操作会在每个阶段自动注入。您可以显式指定迁移操作以覆盖默认行为，也可以使用分配操作手动指定分配规则。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/data-tiers.html
