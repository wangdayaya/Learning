## Elasticsearch 7.10 之 Use index sorting to speed up conjunctions



索引排序对于组织 Lucene doc ID（不与 **_id** 进行合并）可能会很有用，以使连接（ a AND b AND ...）更有效的方式进行。 为了高效，连词依赖于以下事实：如果任何子句不匹配，则整个连词都不匹配。 通过使用索引排序，我们可以将不匹配的文档放在一起，这将有助于有效地跳过不符合连接词的大范围文档 ID 。

此技巧仅适用于低基数字段。 一条经验法则是，您应首先对基数都很低且经常用于过滤的字段进行排序。 排序顺序（ asc 或 desc ）无所谓，因为我们只关心将与相同子句匹配的值彼此靠近。

例如，如果您要索引要出售的汽车，则按燃料类型，车身类型，品牌，注册年份以及最终里程来分类可能会很有趣。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-index-sorting-conjunctions.html
