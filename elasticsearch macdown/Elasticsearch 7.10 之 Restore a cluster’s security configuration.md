## Elasticsearch 7.10 之 Restore a cluster’s security configuration

只有在 **.security** 索引的快照是在同一主要版本的先前次要版本中创建的，才可以还原该快照。每个主要版本的最后一个次要版本都可以转换和读取其主要版本和下一个索引的格式。

还原安全配置时，可以选择完全还原所有配置（包括非安全配置），也可以只还原 .security 索引的内容。如备份基于索引的安全性配置中所述，第二个选项仅包含资源类型的配置。第一种选择的优点是可以从过去的时间点将集群还原到明确定义的状态。第二个选项仅涉及安全性配置资源，但不能完全还原安全性功能。

要从备份还原安全配置，请首先确保已安装包含 .security 快照的存储库：

	GET /_snapshot/my_backup
 
	GET /_snapshot/my_backup/snapshot_1
 
然后登录到其中一个节点主机，导航到 Elasticsearch 安装目录，然后执行以下步骤：

1. 将具有超级用户内置角色的新用户添加到文件领域。

	例如，创建一个名为 **restore\_user** 的用户：

		bin/elasticsearch-users useradd restore_user -p password -r superuser
		
2. 使用以前创建的用户，删除现有的 .security-6 或 .security-7 索引。
 
		curl -u restore_user -X DELETE "localhost:9200/.security-*"
		
	此步骤之后，任何依赖于 .security 索引的身份验证将无法进行。这意味着所有向本机或保留用户进行身份验证的 API 调用都会失败，所有依赖于本机角色的用户也会失败。我们在上面的步骤中创建的文件领域用户将继续工作，因为它没有存储在 .security 索引中，而是使用内置的超级用户角色。

3. 使用同一用户，从快照还原 .security 索引。

		curl -u restore_user -X POST "localhost:9200/_snapshot/my_backup/snapshot_1/_restore" -H 'Content-Type: application/json' -d'
		 {
		    "indices": ".security-*",
		    "include_global_state": true 
		 }
		 '

	**include_global_state：true** 仅对于完整还原是必需的。这将还原全局集群​​元数据，其中包含整个集群的配置信息。如果将其设置为 false ，它将仅恢复 .security 索引的内容，例如用户名和密码，API 密钥，应用程序特权，角色和角色映射定义。

4. （可选）如果需要查看和覆盖快照中包含的设置（通过 include\_global\_state 标志），请选择并应用通过 GET _cluster/settings API 提取的持久性设置。
5. 如果要对集群进行完整的时间点还原，则还必须还原配置文件。同样，这还将恢复非安全性设置。

	这需要备份配置文件的直接文件系统副本，覆盖 $ES\_PATH\_CONF 的内容，然后重新启动节点。这需要在每个节点上完成。根据当前集群配置和还原的配置之间差异的程度，您可能无法执行滚动重启。如果要对配置目录执行完全还原，则建议完全重启集群是最安全的选择。或者，您可能希望将配置文件还原到磁盘上的单独位置，并使用文件比较工具来查看现有配置和还原的配置之间的差异。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/restore-security-configuration.html
