## Elasticsearch 7.10 之 Metrics Aggregations

此族中的聚合基于从要聚合的文档中以一种或另一种方式提取的值来计算指标。 这些值通常从文档的字段中提取（使用字段数据），但是也可以使用脚本生成。

数值指标聚合是一种特殊类型的指标聚合，可输出数值。 一些聚合输出单个数字指标（例如 avg ），称为单值数字指标聚合，其他聚合生成多个指标（例如 stats ），并称为多值数字指标聚合。 当这些值充当某些桶聚合的直接子聚合（某些桶聚合使您可以基于每个桶中的数字度量对返回的桶进行排序）时，单值和多值数字度量聚合之间的区别将发挥作用。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/search-aggregations-metrics.html
