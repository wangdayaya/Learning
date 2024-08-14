## Elasticsearch 7.10 之 Near real-time search

文档和索引的概述章节的内容表明，当文档存储在 Elasticsearch 中时，将在 1 秒内以几乎实时地可以对其进行索引和完全搜索。什么定义了近实时搜索？

Lucene 是 Elasticsearch 所依仗的底层 Java 库，它引入了按段搜索的概念。段类似于倒排索引，但是 Lucene 中的索引一词的意思是“段的集合加上一个提交点“。提交后，将新段添加到提交点并清除缓冲区。

文件系统缓存位于 Elasticsearch 和磁盘之间。内存中的索引缓冲区（图1）中的文档被写入新段（图2）。新段首先写入文件系统缓存（操作代价低），然后才刷新到磁盘（操作代价昂贵）。但是将文件放入高速缓存后，可以像打开其他文件一样打开和读取该文件。


![图1](https://www.elastic.co/guide/en/elasticsearch/reference/current/images/lucene-in-memory-buffer.png)


Lucene 允许编写和打开新的段，使包含的文档可见，无需执行完整的提交即可搜索。与提交磁盘相比，此过程要轻松得多，并且可以经常执行而不会降低性能。


![图2](https://www.elastic.co/guide/en/elasticsearch/reference/current/images/lucene-written-not-committed.png)


在 Elasticsearch 中，这种写入和打开新段的过程称为 refresh 。刷新使自上次刷新以来对索引执行的所有操作都可用于搜索。您可以通过以下方式控制刷新：

* 等待刷新间隔
* 设置 ?refresh 选项
* 使用 Refresh API 显式完成刷新（POST _refresh）

默认情况下，Elasticsearch 会定期每秒刷新一次索引，但仅在最近 30 秒内已收到一个或多个搜索请求的索引上刷新。这就是为什么我们说 Elasticsearch 具有几乎实时的搜索功能：文档更改不可见，无法立即搜索，但是在此时间范围内将变为可见。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/near-real-time.html

