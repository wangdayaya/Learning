##Elasticsearch 7.10 之 Important Elasticsearch configuration


尽管 Elasticsearch 需要很少的配置，但是在投入生产之前，需要考虑许多设置。

进入生产之前，必须考虑以下设置：

* Path settings
* Cluster name setting
* Node name setting
* Network host settings
* Discovery settings
* Heap size settings
* JVM heap dump path setting
* GC logging settings
* Temporary directory settings
* JVM fatal error log setting

### Path settings

对于 **macOS .tar.gz**，**Linux .tar.gz** 和 **Windows .zip** 安装， Elasticsearch 默认将数据和日志写入相应的数据并记录 **$ES\_HOME** 的子目录。但是 **$ES_HOME** 中的文件可能会在升级期间被删除。

在生产中，强烈建议您将 **elasticsearch.yml** 中的 **path.data** 和 **path.logs** 设置为 **$ES_HOME** 之外的位置。

	TIP: Docker，Debian，RPM，macOS Homebrew 和 Windows .msi 安装默认情况下会写入数据并记录到 **$ES_HOME** 以外的位置。

**path.data** 和 **path.logs** 的值因平台而异：

Linux 和 macOS 安装支持 Unix 风格的路径：
		
	path:
	  data: /var/data/elasticsearch
	  logs: /var/log/elasticsearch
	  
如果需要，可以在 **path.data** 中指定多个路径。 Elasticsearch 跨所有提供的路径存储节点的数据，但将每个分片的数据保留在同一路径上。

	WARNNING: Elasticsearch 不能平衡节点数据路径上的分片。单个路径中的高磁盘使用率会触发整个节点的高磁盘使用率水印。触发后，即使节点的其他路径具有可用的磁盘空间，Elasticsearch 也不会将分片添加到该节点。如果需要额外的磁盘空间，建议您添加一个新节点，而不是其他数据路径。

Linux 和 macOS 安装在 path.data 中支持多个 Unix 风格的路径：

	path:
	  data:
	    - /mnt/elasticsearch_1
	    - /mnt/elasticsearch_2
	    - /mnt/elasticsearch_3
### Cluster name setting

当节点与集群中的所有其他节点共享其 **cluster.name** 时，该节点只能加入该集群。默认名称是 **elasticsearch** ，但是您应该将其更改为描述集群用途的适当名称。

	cluster.name: logging-prod
 	
IMPORTANT: 不要在不同的环境中重复使用相同的集群名称。否则，节点可能会加入错误的群集。
### Node name setting


Elasticsearch 使用 **node.name** 作为 Elasticsearch 特定实例的可读标识符。此名称包含在许多 API 的响应中。 Elasticsearch 启动时，节点名称默认为计算机的主机名，但可以在 **elasticsearch.yml** 中显式配置：

	node.name: prod-data-2
### Network host setting

默认情况下，Elasticsearch 仅绑定到环回地址，例如 **127.0.0.1** 和 **[:: 1]** 。此绑定足以在服务器上运行单个开发节点。

TIP: 可以从单个节点上相同的 **$ES_HOME** 位置启动多个节点。此设置对于测试 Elasticsearch 形成集群的能力很有用，但不建议用于生产环境。

要与其他服务器上的节点形成集群，您的节点将需要绑定到非环回地址。虽然有许多网络设置，但通常只需配置 **network.host** ：

	network.host: 192.168.1.10
**network.host** 设置还了解一些特殊值，例如 \_local_ ，\_site_ ，\_global_ 以及诸如 :ip4 和 :ip6 的修饰符。请参阅 [Special values for network.host](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-network.html#network-interface-values)。

当您为 **network.host** 提供自定义设置时，Elasticsearch 会假设您正在从开发模式转换为生产模式，并将许多系统启动检查从警告升级为异常。查看 [development and production modes](https://www.elastic.co/guide/en/elasticsearch/reference/current/system-config.html#dev-vs-prod)。

### Discovery and cluster formation settings


在投入生产之前，请配置两个重要的发现和集群形成设置，以便集群中的节点可以彼此发现并选举一个主节点。

**discovery.seed_hosts**

开箱即用，无需任何网络配置，Elasticsearch 将绑定到可用的环回地址，并扫描本地端口 9300 至 9305 以与在同一服务器上运行的其他节点连接。此行为无需进行任何配置即可提供自动群集体验。

如果要与其他主机上的节点组成集群，请使用 [static](https://www.elastic.co/guide/en/elasticsearch/reference/current/settings.html#static-cluster-setting) 中的 **discovery.seed_hosts** 设置。此设置提供了集群中其他主机节点的列表，这些节点具有成为 master 的资格，并且很可能处于活动状态，而且可以联系以播种 [discovery process](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-discovery-hosts-providers.html) 。此设置接受集群中所有具有成为 master 资格节点的 YAML 序列或地址数组。每个地址可以是 IP 地址，也可以是通过 DNS 解析为一个或多个 IP 地址的主机名。

	discovery.seed_hosts:
	   - 192.168.1.10:9300    
	   - 192.168.1.11    # 端口是可选的，默认为 9300，但可以覆盖
	   - seeds.mydomain.com    # 如果主机名解析为多个 IP 地址，则该节点将尝试在所有解析的地址处发现其他节点
	   - [0:0:0:0:0:ffff:c0a8:10c]:9301    # IPv6 地址必须放在方括号中


如果具有能成为 master 资格的节点没有固定的名称或地址，请使用 [alternative hosts provider](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-discovery-hosts-providers.html#built-in-hosts-providers) 动态查找其地址。

**cluster.initial\_master_nodes**

首次启动 Elasticsearch 集群时，[cluster bootstrapping](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-discovery-bootstrap-cluster.html) 步骤将确定具有能成为 master 资格的节点集，这些节点的投票将在第一次选举中进行计数。在 [development mode](https://www.elastic.co/guide/en/elasticsearch/reference/current/bootstrap-checks.html#dev-vs-prod-mode) ，未配置发现设置，此步骤由节点自身自动执行。

因为自动引导本质上是 [inherently unsafe](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-discovery-quorums.html) ，所以在生产模式下启动新集群时，必须明确列出能够成为 master 节点的列表，它们的投票要在第一次选举中进行计算。您可以使用 **cluster.initial\_master_nodes** 来设置此列表。

IMPORTANT: 集群首次成功形成后，从每个节点的配置中删除 **cluster.initial\_master_nodes** 设置。重新启动集群或将新节点添加到现有集群时，请勿使用此设置。

	discovery.seed_hosts:
	   - 192.168.1.10:9300
	   - 192.168.1.11
	   - seeds.mydomain.com
	   - [0:0:0:0:0:ffff:c0a8:10c]:9301
	cluster.initial_master_nodes:    # 通过默认的主机名 node.name 标识初始 master 节点。确保 cluster.initial_master_nodes 中的值与 node.name 完全匹配。如果您使用完全限定的域名（FQDN），例如 master-node-a.example.com 作为节点名称，则必须在此列表中使用 FQDN 。相反，如果 node.name 是没有任何尾随限定符的裸主机名，则还必须在 cluster.initial_master_nodes 中省略尾随限定符。
	   - master-node-a
	   - master-node-b
	   - master-node-c



请参阅 [bootstrapping a cluster](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-discovery-bootstrap-cluster.html) 以及 [discovery and cluster formation settings](https://www.elastic.co/guide/en/elasticsearch/reference/current/modules-discovery-settings.html) 。

### Heap size settings

默认情况下，Elasticsearch 告诉 JVM 使用最小和最大的大小都为 1 GB 的堆。在进入生产阶段时，配置堆大小以确保 Elasticsearch 有足够的可用堆非常重要。

Elasticsearch 将通过 Xms（最小堆大小）和 Xmx（最大堆大小）设置分配 jvm.options 中指定的整个堆。这两个设置必须彼此相等。

这些设置的值取决于服务器上可用的 RAM 数量：

* 将 **Xmx** 和 **Xms** 设置为不超过物理 RAM 的 50％ 。 Elasticsearch 出于其他目的而需要的内存超过了 JVM 堆，因此为此留出空间很重要。例如 Elasticsearch 使用堆外缓冲区来进行有效的网络通信，依靠操作系统的文件系统缓存来有效地访问文件，而 JVM 本身也需要一些内存。通常观察到 Elasticsearch 进程使用的内存多于 **Xmx** 设置配置的限制。
* 将 **Xmx** 和 **Xms** 设置为不超过 JVM 用于压缩对象指针（压缩 oop ）的阈值。确切的阈值有所不同，但接近 32 GB 。您可以通过在日志中查找如下一行来验证您是否处于阈值以下：

		heap size [1.9gb], compressed ordinary object pointers [true]
	
* 将 **Xmx** 和 **Xms** 设置为不大于基于零的压缩 oop 的阈值。确切的阈值会有所不同，但在大多数系统上 26 GB 是安全的，在某些系统上可能高达 30 GB。您可以通过使用 JVM 选项 -XX:+UnlockDiagnosticVMOptions -XX:+PrintCompressedOopsMode       启动 Elasticsearch 并查找类似于以下内容的行来验证您是否处于此阈值以下：

		heap address: 0x000000011be00000, size: 27648 MB, zero based Compressed Oops
		
	此行显示启用了从零开始的压缩 oop 。如果未启用从零开始的压缩 oop ，则会看到类似以下内容的行：

		heap address: 0x0000000118400000, size: 28672 MB, Compressed Oops with base: 0x00000001183ff000
		
Elasticsearch 可用的堆越多，它可用于其内部缓存的内存就越多，但可供操作系统用于文件系统缓存的内存就越少。同样，较大的堆可能导致较长的垃圾回收暂停。

这是如何通过 **jvm.options.d/** 文件设置堆大小的示例：

	-Xms2g    # 将最小堆大小设置为 2g
	-Xmx2g    # 将最大堆大小设置为 2g


使用 **jvm.options.d** 是为生产部署配置堆大小的首选方法。

也可以通过 **ES\_JAVA_OPTS** 环境变量来设置堆大小。通常不建议在生产部署中使用它，但对测试很有用，因为它会覆盖所有其他设置 JVM 选项的方法。

	ES_JAVA_OPTS="-Xms2g -Xmx2g" ./bin/elasticsearch    # 将最小和最大堆大小设置为 2 GB
	ES_JAVA_OPTS="-Xms4000m -Xmx4000m" ./bin/elasticsearch    # 将最小和最大堆大小设置为 4000 MB


NOTE: Windows 服务的堆配置与上述不同。最初为 Windows 服务填充的值可以按上述配置，但是在安装服务后会有所不同。 有关其他详细信息，请查阅 [Windows service documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/zip-windows.html#windows-service) 。


### JVM heap dump path setting

默认情况下，Elasticsearch 将 JVM 配置为将内存不足异常中的堆转储到默认数据目录。在 RPM 和 Debian 软件包中，数据目录为/var/lib/elasticsearch 。在 Linux，MacOS 和 Windows 发行版上，数据目录位于 Elasticsearch 安装目录的根目录下。

如果此路径不适合接收堆转储，请修改 **jvm.options** 中的 **-XX:HeapDumpPath=...** 条目：

* 如果指定目录，那么 JVM 将基于正在运行的实例的 PID 为堆转储生成文件名。
* 如果指定固定文件名而不是目录，则当 JVM 需要在内存不足异常时执行堆转储时，该文件必须不存在。否则，堆转储将失败。

### GC logging settings

默认情况下，Elasticsearch 启用垃圾收集（GC）日志。它们在 jvm.options 中配置，并输出到与 Elasticsearch 日志相同的默认位置。默认配置每 64 MB 轮换一次日志，最多可消耗 2 GB 磁盘空间。

您可以使用  [JEP 158: Unified JVM Logging](https://openjdk.java.net/jeps/158) 中描述的命令行选项重新配置 JVM 日志记录。除非您直接更改默认的 **jvm.options** 文件，否则除了您自己的设置外，还将应用 Elasticsearch 的默认配置。要禁用默认配置，请首先通过提供 **-Xlog:disable** 选项来禁用日志记录，然后提供您自己的命令行选项。这将禁用所有 **JVM** 日志记录，因此请确保检查可用选项并启用所需的所有功能。

要查看原始 JEP 中未包含的其他选项，请参阅使用  [Enable Logging with the JVM Unified Logging Framework](https://docs.oracle.com/en/java/javase/13/docs/specs/man/java.html#enable-logging-with-the-jvm-unified-logging-framework) 。

**例子**

通过创建带有一些示例选项的 **$ES_HOME/config/jvm.options.d/gc.options** ，将默认 GC 日志输出位置更改为 **/opt/my-app/gc.log** ：

	# Turn off all previous logging configuratons
	-Xlog:disable
	
	# Default settings from JEP 158, but with `utctime` instead of `uptime` to match the next line
	-Xlog:all=warning:stderr:utctime,level,tags
	
	# Enable GC logging to a custom location with a variety of options
	-Xlog:gc*,gc+age=trace,safepoint:file=/opt/my-app/gc.log:utctime,pid,tags:filecount=32,filesize=64m

配置 Elasticsearch [Docker container](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) 以将 GC 调试日志发送到标准错误（stderr）。这使容器协调器可以处理输出。如果使用 **ES\_JAVA_OPTS** 环境变量，请指定：

	MY_OPTS="-Xlog:disable -Xlog:all=warning:stderr:utctime,level,tags -Xlog:gc=debug:stderr:utctime"
	docker run -e ES_JAVA_OPTS="$MY_OPTS" # etc
### Temporary directory settings

默认情况下，Elasticsearch 使用启动脚本在系统临时目录下立即创建的私有临时目录。

在某些 Linux 发行版中，如果最近没有访问过，则系统实用程序将从 **/tmp** 中清除文件和目录。如果要求临时目录的使用不是长时间持续的，此行为可能会导致在运行 Elasticsearch 时删除私有临时目录。如果功能要求随后使用该目录，则删除私有临时目录会导致问题。

如果您使用 **.deb** 或 **.rpm** 软件包安装 Elasticsearch ，并在 **systemd** 下运行，则 Elasticsearch 使用的私有临时目录会在定期清理中去掉。

如果您打算长时间在 Linux 或 MacOS 上运行 **.tar.gz** 发行版，请考虑为 Elasticsearch 创建专用的临时目录，该目录不在即将清除旧文件和目录的路径下。该目录应设置权限，以便只有运行 Elasticsearch 的用户才能访问它。然后在启动 Elasticsearch 之前，将 **$ES_TMPDIR** 环境变量设置为指向此目录。

### JVM fatal error log setting

默认情况下，Elasticsearch 将 JVM 配置为将致命错误日志写入默认日志目录。在 RPM 和 Debian 软件包中，该目录为 **/var/log/elasticsearch** 。在 Linux ，MacOS 和 Windows 发行版上，**logs** 目录位于 Elasticsearch 安装目录的根目录下。

这些是 JVM 在遇到致命错误（例如分段错误）时生成的日志。如果此路径不适合接收日志，请修改 **jvm.options** 中的 **-XX:ErrorFile=...**  条目。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/important-settings.html

