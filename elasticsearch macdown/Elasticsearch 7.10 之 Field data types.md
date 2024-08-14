## Elasticsearch 7.10 之 Field data types


每个字段都有一个字段数据类型或字段类型。此类型指示字段包含的数据类型（例如字符串或布尔值）及其预期用途。例如，您可以将字符串索引到 **text** 字段和 **keyword** 字段。分析文本字段值以进行全文搜索，而将关键字字符串保持原样以进行过滤和排序。

字段类型按 family 分组。同一族中的类型支持相同的搜索功能，但可能具有不同的空间使用或性能特征。

当前，唯一的类型族是 **keyword** ，它由 **keyword** ，**constant_keyword** 和 **wildcard** 字段类型组成。其他类型族只有一个字段类型。例如，**boolean** 类型族由一个字段类型组成：**boolean**。

### Common types

[binary](https://www.elastic.co/guide/en/elasticsearch/reference/current/binary.html) ：二进制值编码为 Base64 字符串

[boolean](https://www.elastic.co/guide/en/elasticsearch/reference/current/boolean.html) ：true 和 false 的值

[Keywords](https://www.elastic.co/guide/en/elasticsearch/reference/current/keyword.html) ：关键字族，包括 keyword ，constant_keyword 和 wildcard

[Numbers](https://www.elastic.co/guide/en/elasticsearch/reference/current/number.html) ：数字类型，例如 long 和 double ，用于表示数量 

Dates ：日期类型，包括 [date](https://www.elastic.co/guide/en/elasticsearch/reference/current/date.html) 和 [date_nanos](https://www.elastic.co/guide/en/elasticsearch/reference/current/date_nanos.html)

[alias](https://www.elastic.co/guide/en/elasticsearch/reference/current/alias.html) ：为现有字段定义别名

### Objects and relational types

[object](https://www.elastic.co/guide/en/elasticsearch/reference/current/object.html) ：JSON 对象

[flattened](https://www.elastic.co/guide/en/elasticsearch/reference/current/flattened.html) ：整个 JSON 对象作为单个字段值

[nested](https://www.elastic.co/guide/en/elasticsearch/reference/current/nested.html) ：保留其子字段之间关系的 JSON 对象

[join](https://www.elastic.co/guide/en/elasticsearch/reference/current/parent-join.html) ：为同一索引中的文档定义父/子关系

### Structured data types

[Range](https://www.elastic.co/guide/en/elasticsearch/reference/current/range.html) ：范围类型，例如 long\_range ， double\_range ，date\_range 和 ip\_range

[ip](https://www.elastic.co/guide/en/elasticsearch/reference/current/ip.html) ：IPv4 和 IPv6 地址

[version](https://www.elastic.co/guide/en/elasticsearch/reference/current/version.html) ：软件版本。支持语义版本控制优先级规则

[murmur3](https://www.elastic.co/guide/en/elasticsearch/plugins/7.10/mapper-murmur3.html) ：计算并存储值的哈希

### Aggregate data types

[histogram](https://www.elastic.co/guide/en/elasticsearch/reference/current/histogram.html) ：预汇总的数值

### Text search types

[text](https://www.elastic.co/guide/en/elasticsearch/reference/current/text.html) ：分析的非结构化文本

[annotated-text](https://www.elastic.co/guide/en/elasticsearch/plugins/7.10/mapper-annotated-text.html) ：包含特殊标记的文本。用于标识命名实体

[completion](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-suggesters.html#completion-suggester) ：用于自动完成建议

[search_as_you_type](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-as-you-type.html) ：像文字一样的类型，可按需输入

[token_count](https://www.elastic.co/guide/en/elasticsearch/reference/current/token-count.html) ：文本中的令牌计数

### Document ranking types

[density_vector](https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html) ：记录浮点值的密集向量

[sparse_vector](https://www.elastic.co/guide/en/elasticsearch/reference/current/sparse-vector.html) ：记录浮点值的稀疏向量

[rank_feature](https://www.elastic.co/guide/en/elasticsearch/reference/current/rank-feature.html) ：记录数字特点以提高查询时的点击率

[rank_features](https://www.elastic.co/guide/en/elasticsearch/reference/current/rank-features.html) ：记录数字特点以提高查询时的点击率

### Spatial data types

[geo_point](https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-point.html) ：纬度和经度点
[geo_shape](https://www.elastic.co/guide/en/elasticsearch/reference/current/geo-shape.html) ：复杂的形状，例如多边形
[point](https://www.elastic.co/guide/en/elasticsearch/reference/current/point.html) ：任意笛卡尔点
[shape](https://www.elastic.co/guide/en/elasticsearch/reference/current/shape.html) ：任意笛卡尔几何

### Other types
[percolator](https://www.elastic.co/guide/en/elasticsearch/reference/current/percolator.html) ：索引以查询 DSL 编写的查询

### Array

在 Elasticsearch 中，数组不需要专用的字段数据类型。默认情况下，任何字段都可以包含零个或多个值，但是，数组中的所有值都必须具有相同的字段类型。请参阅 [Arrays](https://www.elastic.co/guide/en/elasticsearch/reference/current/array.html) 。

### Multi-fields

为不同的目的以不同的方式对同一字段建立索引通常很有用。例如，字符串字段可以映射为用于全文本搜索的 **text** 字段，也可以映射为用于排序或聚合的 **keyword** 字段。或者您可以使用  **standard analyzer** ，**english analyzer** 和  **french analyzer** 为文本字段建立索引。

这是多字段的目的。大多数字段类型通过 [fields](https://www.elastic.co/guide/en/elasticsearch/reference/current/multi-fields.html) 参数支持多字段。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-types.html
