## Elasticsearch 7.10 之 Scripting


使用脚本，您可以在 Elasticsearch 中评估自定义表达式。例如，您可以使用脚本在搜索请求中返回“脚本字段”或评估查询的自定义分数。

默认的脚本语言是 [Painless](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting-painless.html) 。其他 **lang** 插件使您可以运行以其他语言编写的脚本。在可以使用脚本的任何地方，都可以包含 lang 参数以指定脚本的语言。

### General-purpose languages

这些语言可在脚本 API 中用于任何目的，并提供最大的灵活性。

Language|Sandboxed|Required plugin
---|---|---
painless|yes|built-in

### Special-purpose languages

这些语言不太灵活，但是对于某些任务通常具有更高的性能。


Language|Sandboxed|Required plugin|Purpose
---|---|---|---
[expression](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting-expression.html)|yes|built-in|fast custom ranking and sorting
[mustache](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-template.html)|yes|built-in|templates
[java](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting-engine.html)|n/a|you write it!|expert API

### Scripts and security
沙盒语言在设计时考虑到安全性。但是，非沙盒语言可能是一个安全问题，请阅读 [Scripting and security](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting-security.html) 以获取更多详细信息。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-scripting.html
