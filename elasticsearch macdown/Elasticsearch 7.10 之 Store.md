## Elasticsearch 7.10 之 Store

使用存储模块，您可以控制如何在磁盘上存储和访问索引数据。

NOTE：这是一个低级别的设置。某些存储实现的并行性很差，或者禁用了堆内存使用的优化。我们建议坚持默认设置。

### File system storage types

有不同的文件系统实现或存储类型。默认情况下，Elasticsearch 将根据操作环境选择最佳实施。

还可以通过在 **config/elasticsearch.yml** 文件中配置存储类型，为所有索引显式设置存储类型：

	index.store.type: hybridfs
这是一个静态设置，可以在创建索引时基于每个索引进行设置：

	PUT /my-index-000001
	{
	  "settings": {
	    "index.store.type": "hybridfs"
	  }
	}

 
WARNING：这是仅限专家的设置，以后可能会删除。
以下各节列出了支持的所有不同存储类型。

**fs** ：默认文件系统实现。这将根据操作环境选择最佳的实现，当前的操作环境在所有受支持的系统上都是混合的，但可能会发生变化。

**simplefs** ：Simple FS 类型是使用随机访问文件直接实现文件系统存储（映射到 Lucene SimpleFsDirectory ）。此实现的并行性能较差（多个线程将成为瓶颈），并禁用了一些针对堆内存使用的优化。

**niofs** ：NIO FS 类型使用 NIO 在文件系统上存储分片索引（映射到 Lucene NIOFSDirectory ）。它允许多个线程同时读取同一文件。由于 SUN Java 实现中存在错误，因此不建议在 Windows 上使用它，并且会禁用堆内存使用的某些优化。

**mmapfs** ：MMap FS 类型通过将文件映射到内存（mmap）将分片索引存储在文件系统上（映射到 Lucene MMapDirectory ）。内存映射将占用您进程中虚拟内存地址空间的一部分，该空间等于要映射的文件的大小。在使用此类之前，请确保您已允许足够的虚拟地址空间。

**hybridfs** ：hybridfs 类型是 niofs 和 mmapfs 的混合类型，它根据读取访问模式为每种文件类型选择最佳的文件系统类型。当前，只有 Lucene 术语词典，规范和 doc 值文件才进行内存映射。使用Lucene NIOFSDirectory 打开所有其他文件。与 mmapfs 相似，请确保您已提供足够的虚拟地址空间。

您可以通过设置 node.store.allow_mmap 来限制 mmapfs 和相关的 hybridfs 存储类型的使用。这是一个布尔设置，指示是否允许内存映射。默认值为允许。则此设置很有用，例如，如果您在无法控制创建大量内存映射的能力的环境中，因此需要禁用使用内存映射的能力。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-store.html
