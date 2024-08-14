## Elasticsearch 7.10 之 File Descriptors


NOTE: 这仅与 Linux 和 macOS 有关，如果在 Windows上 运行 Elasticsearch，则可以安全地忽略它。 在 Windows 上，JVM 使用仅受可用资源限制的 API。

Elasticsearch 使用许多文件描述符或文件句柄。 文件描述符用尽可能是灾难性的，很可能导致数据丢失。 确保将运行 Elasticsearch 的用户的打开文件描述符的数量限制增加到 65536 或更高。

对于 **.zip** 和 **.tar.gz** 软件包，在启动 Elasticsearch 之前将 **ulimit -n 65535** 设置为 root ，或者在 **/etc/security/limits.conf** 中将 **nofile** 设置为 **65535** 。

在 macOS 上，您还必须将 JVM 选项 **-XX:-MaxFDLimit** 传递给 Elasticsearch ，以使其使用更高的文件描述符限制。

RPM 和 Debian 软件包已经默认将文件描述符的最大数量设置为 65535 ，并且不需要进一步配置。

您可以使用 Nodes stats API 检查为每个节点配置的 **max\_file_descriptors** ，其中：

	GET _nodes/stats/process?filter_path=**.max_file_descriptors

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/file-descriptors.html
