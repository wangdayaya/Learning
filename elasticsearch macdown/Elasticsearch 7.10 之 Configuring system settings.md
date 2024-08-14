##Elasticsearch 7.10 之 Configuring system settings

在何处配置系统设置取决于您用于安装 Elasticsearch 的软件包以及所使用的操作系统。

使用 .zip 或 .tar.gz 软件包时，可以配置系统设置：

* 暂时使用 **[ulimit](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#ulimit)**
* 永久在 **[/etc/security/limits.conf](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#limits.conf)** 中

使用 RPM 或 Debian 软件包时，大多数系统设置是在 [system configuration file](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#sysconfig) 中设置的。但是，使用 systemd 的系统要求在 [systemd configuration file](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#systemd) 中指定系统限制。

### ulimit

在 Linux 系统上，可以使用 **ulimit** 临时更改资源限制。在切换到将要运行 Elasticsearch 的用户之前，通常需要将限制设置为 root 。例如要将打开的文件句柄数（ulimit -n）设置为 65536 ，可以执行以下操作：

	sudo su    # 成为root
	ulimit -n 65535     # 更改打开文件的最大数量
	su elasticsearch    # 成为 elasticsearch 用户，以便启动Elasticsearch

新限制仅适用于当前会话。

可以使用 **ulimit -a** 查询所有当前应用的限制。

### /etc/security/limits.conf

在 Linux 系统上，可以通过编辑 **/etc/security/limits.conf** 文件来为特定用户设置永久限制。要将 elasticsearch 用户的最大打开文件数设置为 65535 ，请在 limits.conf 文件中添加以下行：

	elasticsearch  -  nofile  65535
该更改仅在 **elasticsearch** 用户下次打开新会话时才生效。

**NOTE: Ubuntu 和 limits.conf**

对于由 init.d 启动的进程，Ubuntu 忽略 limits.conf 文件。要启用 limits.conf 文件，请编辑 /etc/pam.d/su 并取消注释以下行：
	# session    required   pam_limits.so

### Sysconfig file

使用 RPM 或 Debian 软件包时，可以在系统配置文件中指定系统设置和环境变量，该文件位于：

	RPM			/etc/sysconfig/elasticsearch
	
	Debian		/etc/default/elasticsearch

但是，对于使用 **systemd** 的系统，需要通过 **[systemd](https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html#systemd)** 指定系统限制。

### Systemd configuration

在使用 systemd 的系统上使用 RPM 或 Debian 软件包时，必须通过 systemd 指定系统限制。

systemd 服务文件（**/usr/lib/systemd/system/elasticsearch.service**）包含默认情况下应用的限制。

要覆盖它们，请添加一个名为 **/etc/systemd/system/elasticsearch.service.d/override.conf** 的文件（或者，您可以运行 **sudo systemctl edit elasticsearch** ，这会在默认编辑器中自动打开该文件）。设置此文件中的所有更改，例如：

	[Service]
	LimitMEMLOCK=infinity
完成后，运行以下命令以重新加载单元：

	sudo systemctl daemon-reload



详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/setting-system-settings.html
