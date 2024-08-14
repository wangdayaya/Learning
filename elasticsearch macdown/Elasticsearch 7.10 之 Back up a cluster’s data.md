## Elasticsearch 7.10 之 Back up a cluster’s data

要备份集群的数据，可以使用 [snapshot API](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/modules-snapshots.html) 。

快照是从正在运行的 Elasticsearch 集群中获取的备份。您可以对整个集群进行快照，包括其所有数据流和索引。您还可以仅对集群中的特定数据流或索引进行快照。

必须先 [register a snapshot repository](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/snapshots-register-repository.html) ，然后才能 [create snapshots](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/snapshots-take-snapshot.html)。

快照可以存储在本地或远程存储库中。远程存储库可以驻留在 Amazon S3，HDFS，Microsoft Azure，Google Cloud Storage 和存储库插件支持的其他平台上。

Elasticsearch 增量拍摄快照：快照过程仅将先前快照尚未复制到的数据拷贝到存储库中，从而避免了不必要的工作或存储空间重复。这意味着您可以安全地非常频繁地以最小的开销拍摄快照。但是，快照在逻辑上也是独立的：删除快照不会影响任何其他快照的完整性。

如果您的集群启用了 Elasticsearch 安全功能，则在备份数据时，必须授权快照 API 调用。

**snapshot_user** 角色是保留角色，可以分配给正在调用快照端点的用户。如果所有用户所做的只是作为备份过程一部分的定期快照，则这是唯一必要的角色。该角色包括列出所有现有快照（任何存储库）的特权以及列出和查看所有索引（包括 **.security** 索引）的设置的特权。它不授权创建存储库，还原快照或在索引内搜索的特权。因此，用户可以查看和快照所有索引，但不能访问或修改任何数据。

有关更多信息，请参见 [Security privileges](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/security-privileges.html) 和 [Built-in roles](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/built-in-roles.html) 。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/backup-cluster-data.html
