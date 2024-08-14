## Elasticsearch 7.10 之 Dynamic Mapping



Elasticsearch 的最重要功能之一是它试图摆脱束缚，让您尽快开始探索数据。 要为文档建立索引，您不必先创建索引、定义映射类型并定义字段——您只需为文档建立索引，那么索引、类型和字段就会自动生效：

	PUT data/_doc/1   # 创建 data 索引，_doc 映射类型以及一个名为 count 的字段，其数据类型为 long
	{ "count": 5 }
 



自动检测和添加新字段称为动态映射。 可通过以下方式自定义动态映射规则以适合您的目的：

[Dynamic field mappings](https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic-field-mapping.html) ：管理动态字段检测的规则。

[Dynamic templates](https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic-templates.html) ：用于为动态添加的字段配置映射的自定义规则。

TIP ：索引模板允许您配置新索引的默认映射、设置和别名，无论是自动创建还是显式创建。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic-mapping.html
