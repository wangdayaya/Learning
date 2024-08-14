## Elasticsearch 7.10 之 JNA temporary directory not mounted with noexec



NOTE: 这仅与 Linux 有关。
Elasticsearch 使用 Java 本机访问（JNA）库来执行一些平台相关的本机代码。在 Linux 上，在运行时从 JNA 存档中提取支持该库的本机代码。默认情况下，此代码被提取到 Elasticsearch 临时目录，该目录默认为 **/tmp** 的子目录。或者可以使用 JVM 标志 **-Djna.tmpdir=<path>** 控制此位置。由于本机库以可执行文件的形式映射到 JVM 虚拟地址空间，因此不得使用 **noexec** 挂载此代码提取到的位置的基础安装点，因为这会阻止 JVM 进程将其映射为可执行文件。在某些加固的 Linux 安装中，这是 /tmp 的默认安装选项。指示使用 noexec 挂载了基础挂载的一种迹象是，JNA 在启动时将无法通过 **java.lang.UnsatisfiedLinkerError** 异常加载，并带有一条消息 **failed to map segment from shared object** 。请注意，在 JVM 版本之间，异常消息可能有所不同。此外，依赖于通过 JNA 执行本机代码的 Elasticsearch 组件将失败，并显示消息，表明这是因为 **because JNA is not available** 。如果看到此类错误消息，则必须重新挂载用于 JNA 的临时目录，以使其不通过 **noexec** 挂载。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/executable-jna-tmpdir.html
