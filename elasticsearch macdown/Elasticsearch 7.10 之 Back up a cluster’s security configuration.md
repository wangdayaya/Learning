## Elasticsearch 7.10 之 Back up a cluster’s security configuration

安全配置信息位于两个位置：[files](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/security-backup.html#backup-security-file-based-configuration) 和 [indices](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/security-backup.html#backup-security-index-configuration)。

### Back up file-based security configuration
Elasticsearch 安全功能是使用 elasticsearch.yml 和 elasticsearch.keystore 文件中的 xpack.security 命名空间配置的。此外，在同一 ES\_PATH\_CONF 目录中还有其他几个其他配置文件。这些文件定义角色和角色映射并配置文件领域。一些设置指定了对安全敏感的数据的文件路径，例如用于 HTTP 客户端和节点间通信的 TLS 密钥和证书以及用于 SAML，OIDC 和 Kerberos 领域的私钥文件。所有这些都还存储在 ES_PATH_CONF 中；路径设置是相对的。

elasticsearch.keystore、TLS 密钥、 SAML 、OIDC 和 Kerberos 领域专用密钥文件需要机密性。将文件复制到备份位置时，这一点至关重要，因为这会增加恶意侦听的范围。

要备份所有此配置，可以使用常规的基于文件的备份，如上一节所述。

* 文件备份必须在每个集群节点上运行。
* 文件备份也将存储非安全性配置。不支持仅备份安全功能的配置。备份是完整配置状态的时间点记录。

### Back up index-based security configuration
Elasticsearch 安全功能将系统配置数据存储在专用索引中。该索引在 Elasticsearch 6.x 版本中名为 **.security-6** ，在 7.x 版本中名为 **.security-7** 。 **.security** 别名始终指向适当的索引。该索引包含配置文件中不可用的数据，并且无法使用标准文件系统工具可靠地备份这些数据。该数据描述：

* 本机领域中用户的定义（包括哈希密码）
* 角色定义（通过创建角色 API 定义）
* 角色映射（通过创建角色映射 API 定义）
* 应用程序权限
* API密钥

因此 .security 索引除了配置信息外，还包含资源和定义。完整的安全功能备份中需要所有这些信息。

像使用其他任何数据索引一样，使用标准的 Elasticsearch 快照功能来备份 .security 。为了方便起见，以下是完整的步骤：

1. 创建可用于备份 .security 索引的存储库。最好为此专用索引有一个专用的存储库。如果愿意，您还可以将其他 Elastic Stack 组件的系统索引快照到此存储库。

		PUT /_snapshot/my_backup
		{
		  "type": "fs",
		  "settings": {
		    "location": "my_backup_location"
		  }
		}
	 
	调用此 API 的用户必须具有提升的管理集群特权，以防止非管理员泄露数据。

2. 创建一个用户，并仅为其分配内置的 **snapshot_user** 角色。

	以下示例在本机领域中创建了一个新用户 snapshot_user ，但用户所属的领域并不重要：

		POST /_security/user/snapshot_user
		{
		  "password" : "secret",
		  "roles" : [ "snapshot_user" ]
		}
 
	创建授权为 snapshot_user 的增量快照。

3. 以下示例显示如何使用创建快照 API 将 .security 索引备份到 my_backup 存储库：

		PUT /_snapshot/my_backup/snapshot_1
		{
		  "indices": ".security",
		  "include_global_state": true 
		}
 

	此参数值捕获存储在全局集群元数据中的所有持久性设置以及其他配置（例如别名和存储的脚本）。请注意，这包括非安全性配置，是对文件系统配置文件备份的补充，但不能代替备份。

索引格式仅在单个主要版本中兼容，并且不能还原到其原始版本之前的版本。例如，您可以将安全快照从 6.6.0 还原到 6.7.0 集群中，但是不能将其还原到运行 Elasticsearch 6.5.0 或 7.0.0 的集群中。

### Controlling access to the backup repository
安全索引的快照通常将包含敏感数据，例如用户名和密码。由于密码是使用加密哈希存储的，因此快照的公开不会自动使第三方能够作为您的用户之一进行身份验证或使用 API ​​密钥。但是它将披露机密信息。

同样重要的是，保护备份的完整性，以防万一您需要还原它们。如果第三方能够修改存储的备份，则他们可以安装后门，如果将快照加载到 Elasticsearch 集群中，该后门将授予访问权限。

我们建议您：

* 在专用存储库中快照 **.security** 索引的快照，在该存储库中严格限制和审核读写访问。
* 如果有迹象表明已读取快照，请在本机领域中更改用户密码并撤消 API 密钥。
* 如果有迹象表明快照已被篡改，请不要还原它。 当前没有还原过程检测恶意篡改的选项。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/security-backup.html
