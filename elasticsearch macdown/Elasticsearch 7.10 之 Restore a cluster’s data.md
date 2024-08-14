## Elasticsearch 7.10 之 Restore a cluster’s data


您可以将快照还原到正在运行的集群，该集群默认情况下包括快照中的所有数据流和索引。 但是，您可以选择仅从快照还原集群状态或特定的数据流或索引。

如果您的集群启用了 Elasticsearch 安全功能，则还原 API 需要 manage 集群特权。还原过程没有定制角色。此特权是非常纵容的，并且仅应授予“管理员”类别中的用户。具体来说，它允许恶意用户将数据泄露到他们选择的位置。自动化工具不应被具有此特权的用户身份运行。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/restore-cluster-data.html
