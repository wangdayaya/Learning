## Elasticsearch 7.10 之 Back up a cluster’s configuration

除了备份集群中的数据外，备份其配置也很重要，特别是在集群变大且难以重构时。

配置信息驻留在每个集群节点上的常规文本文件中。在二进制安全容器 elasticsearch.keystore 文件中指定了敏感设置值，例如 Watcher 通知服务器的密码；某些设置值是指向相关配置数据的文件路径，例如摄取 geo ip 数据库。所有这些文件都包含在 ES\_PATH\_CONF 目录中。

对配置文件的所有更改都是通过手动编辑文件或使用命令行实用程序完成的，而不是通过 API 。实际上，这些更改在初始设置后很少发生。

我们建议您使用所选的文件备份软件对 Elasticsearch config（$ ES\_PATH\_CONF）目录进行定期（理想情况下，每天）备份。

我们建议您为这些配置文件制定一个配置管理计划。您可能希望将它们的检查放到到版本控制中，或者通过选择配置管理工具来配置它们。

这些文件中的某些文件可能包含敏感数据，例如密码和 TLS 密钥，因此您应调查备份软件 and/or 存储解决方案是否能够加密此数据。

配置文件中的某些设置可能会被集群设置覆盖。您可以通过指定快照 API 的 **include\_global\_state：true**（默认）参数来捕获数据备份快照中的这些设置。或者，您可以使用 [get settings API](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/cluster-get-settings.html) 以文本格式提取这些配置值：

	GET _cluster/settings?pretty&flat_settings&filter_path=persistent
 
您可以将此输出与其他配置文件一起存储为文件。

* 暂态设置不用于备份。
* Elasticsearch 安全功能将配置数据（例如角色定义和 API 密钥）存储在专用的特殊索引中。此“系统”数据是对安全设置配置的补充，也应备份。
* 其他 Elastic Stack 组件（例如 Kibana 和 Machine learning ）将其配置数据存储在其他专用索引中。从 Elasticsearch 角度来看，这些只是数据，因此您可以使用常规数据备份过程。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/backup-cluster-configuration.html
