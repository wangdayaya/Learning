##Elasticsearch 7.10 之 Disable swapping

大多数操作系统尝试为文件系统缓存使用尽可能多的内存，并快速换出未使用的应用程序内存。这可能导致 JVM 堆的一部分甚至其可执行页面换出到磁盘上。

交换对性能，节点稳定性非常不利，应不惜一切代价避免交换。它可能导致垃圾回收持续几分钟而不是毫秒，并且可能导致节点响应缓慢甚至断开与群集的连接。在弹性分布式系统中，让操作系统杀死该节点更为有效。

有三种禁用交换的方法。首选选项是完全禁用交换。如果这不是一个选择，是否要尽量减少交换性而不是内存锁定，取决于您的环境。

### Disable all swap files

通常，Elasticsearch 是在盒子上运行的唯一服务，其内存使用量由 JVM 选项控制。无需启用交换功能。

在 Linux 系统上，可以通过运行以下命令暂时禁用交换：

	sudo swapoff -a
不需要重启 Elasticsearch 。

要永久禁用它，您将需要编辑 **/etc/fstab** 文件，并注释掉包含单词 **swap** 的所有行。

在 Windows 上，可以通过 **System Properties → Advanced → Performance → Advanced → Virtual memory** 完全禁用分页文件来实现等效操作。

### Configure swappiness

在 Linux 系统上可用的另一个选项是确保将 sysctl 值 **vm.swappiness** 设置为 1 。这可以减少内核的交换趋势，并且在正常情况下不应导致交换，尽管仍然允许整个系统在紧急情况下进行交换。

### Enable bootstrap.memory_lock

另一个选择是在 Linux / Unix 系统上使用 **mlockall** 或在 Windows 上使用 **VirtualLock** 尝试将进程地址空间锁定在 RAM 中，以防止任何 Elasticsearch 堆内存被换出。

	NOTE: 使用内存锁时，某些平台仍会交换堆外内存。为防止堆外内存交换，请禁用所有交换文件。
要启用内存锁定，请在 **elasticsearch.yml** 中将 **bootstrap.memory_lock** 设置为 **true**：

	bootstrap.memory_lock: true
WARNING: 如果 **mlockall** 尝试分配的内存超过可用内存，则可能导致 **JVM** 或 **Shell** 会话退出！

启动 Elasticsearch 之后，您可以通过检查此请求的输出中的 mlockall 值来查看是否成功应用了此设置：

	GET _nodes?filter_path=**.mlockall
 
如果看到 **mlockall** 为 **false** ，则表示 mlockall 请求已失败。您还将在日志中看到一行包含更多信息的行，其内容为 **Unable to lock JVM Memory** 。

在 Linux / Unix 系统上，最可能的原因是运行 Elasticsearch 的用户无权锁定内存。可以授予以下权限：

* .zip and .tar.gz

	在启动 Elasticsearch 之前将 **[ulimit -l unlimited](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#ulimit)** 设置为 root 用户，或者在 **[/etc/security/limits.conf](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#limits.conf)** 中将 **memlock** 设置为 **unlimited**

* RPM and Debian

	在 **[system configuration file](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#sysconfig)** 中将 **MAX\_LOCKED_MEMORY** 设置为 **unlimited**（对于使用 **systemd** 的系统，请参见下文）

* Systems using systemd

	在  [systemd configuration](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#systemd) 中将 **LimitMEMLOCK** 设置为 **infinity** 

**mlockall** 失败的另一个可能原因是 JNA 临时目录（通常是 /tmp 的子目录）是使用 noexec 选项安装的。这可以通过使用 ES\_JAVA_OPTS 环境变量为 JNA 指定新的临时目录来解决：

	export ES_JAVA_OPTS="$ES_JAVA_OPTS -Djna.tmpdir=<path>"
	./bin/elasticsearch
或在 jvm.options 配置文件中设置此 JVM 标志。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/setup-configuration-memory.html
