## Elasticsearch 7.10 之 Text analysis

文本分析是将非结构化文本（例如电子邮件的正文或产品说明）转换为针对搜索优化的结构化格式的过程。

### When to configure text analysis

当索引或搜索 text 字段时，Elasticsearch 执行文本分析。

如果您的索引不包含文本字段，则无需进一步设置； 您可以跳过本节中的页面。

但是，如果您使用 text 字段或文本搜索未返回预期的结果，则配置文本分析通常会有所帮助。 如果您要使用 Elasticsearch 执行以下操作，则还应研究分析配置：

* 建立一个搜索引擎
* 挖掘非结构化数据
* 优化搜索特定语言
* 进行词典或语言研究

### In this section

* [Overview](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-overview.html)
* [Concepts](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-concepts.html)
* [Configure text analysis](https://www.elastic.co/guide/en/elasticsearch/reference/current/configure-text-analysis.html)
* [Built-in analyzer reference](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-analyzers.html)
* [Tokenizer reference](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-tokenizers.html)
* [Token filter reference](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-tokenfilters.html)
* [Character filters reference](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-charfilters.html)
* [Normalizers](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-normalizers.html)

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis.html
