## Elasticsearch 7.10 之 Metadata fields

每个文档都有与其关联的元数据，例如 \_index ，_type 和 _id 元数据字段。创建映射类型时，可以自定义其中一些元数据字段的行为。

### Identity metadata fields

[_index](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-index-field.html) ：文档所属的索引。

[_type](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-type-field.html) ：文档的映射类型。

[_id](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-id-field.html) ：文件编号。

### Document source metadata fields

[_source](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-source-field.html) ：表示文档正文的原始 JSON 。

[_size](https://www.elastic.co/guide/en/elasticsearch/plugins/7.10/mapper-size.html) ：_source 字段的大小（以字节为单位），由 [mapper-size plugin](https://www.elastic.co/guide/en/elasticsearch/plugins/7.10/mapper-size.html) 提供。

### Indexing metadata fields

[\_field_names](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-field-names-field.html) ：文档中包含非空值的所有字段。

[_ignored](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-ignored-field.html) ：由于 [ignore_malformed](https://www.elastic.co/guide/en/elasticsearch/reference/current/ignore-malformed.html) ，在索引时已被忽略的文档中的所有字段。


### Routing metadata field
[_routing](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-routing-field.html) ：一个自定义的路由值，用于将文档路由到特定的分片。

### Other metadata field

[_meta](https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-meta-field.html) ：特定于应用程序的元数据。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-fields.html
