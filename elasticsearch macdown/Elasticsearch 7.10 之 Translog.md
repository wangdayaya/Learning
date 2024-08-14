## Elasticsearch 7.10 之 Translog

对 Lucene 的更改仅在 Lucene 提交期间才保留在磁盘上，这是一个相对昂贵的操作，因此无法在每次索引操作或删除操作之后执行更改操作。如果发生进程退出或硬件故障，Lucene 会把在一次提交之后到另一次提交之前发生的更改从索引中删除。

Lucene 的提交太昂贵而无法执行每个单独的更改操作，因此每个分片副本还将操作写入其事务日志中，称为 **translog** 。在由内部 Lucene 索引处理之后但在确认之前，所有索引和删除操作都将写入事务日志。在崩溃的情况下，当分片恢复时，已被确认但尚未包括在上次 Lucene 提交中的近期操作将从事务日志中恢复。

Elasticsearch [flush](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-flush.html) 是执行 Lucene 提交并开始生成新的 translog 的过程。刷新会在后台自动执行，以确保 translog 不会变得太大，这会使恢复过程中的重放操作花费大量时间。手动执行刷新的功能也通过 API 执行，尽管很少需要这样做。

### Translog settings

仅当同步和提交事务日志时，事务日志中的数据才会持久保存到磁盘。如果发生硬件故障、操作系统崩溃、JVM 崩溃或分片故障，则自上一个事务记录提交以来写入的所有数据都将丢失。

默认情况下，将 **index.translog.durability** 设置为 **request** ，这意味着 Elasticsearch 仅在成功对主分片和每个已分配副本进行事务同步并提交事务后，才将索引、删除、更新或批量请求的成功报告给客户端。 如果将 **index.translog.durability** 设置为 **async** ，则 Elasticsearch 同步和提交事务日志仅在每个 **index.translog.sync_interval** 时执行，这意味着当节点恢复时，在崩溃之前执行的任何操作都可能会丢失。

以下可动态更新的每个索引设置控制事务日志的行为：

**index.translog.sync_interval** ：不管写操作如何，将 Translog 多长时间同步到磁盘并提交一次。默认为 **5s** 。不允许小于 **100ms** 的值。

**index.translog.durability** ：在每个索引、删除、更新或批量请求之后是否同步和提交事务日志。此设置接受以下参数：

 * request ：（默认）同步并在每个请求后提交。如果发生硬件故障，所有已确认的写入将已经提交到磁盘。
 * async ：同步和提交在每个 **sync_interval** 时之行一次。如果发生故障，则自上次自动提交以来所有已确认的写入将被丢弃。

**index.translog.flush_threshold_size** ：事务日志存储尚未安全地持久化在 Lucene 中的所有操作（即不是 Lucene 提交点的一部分）。尽管可以读取这些操作，但是如果分片已停止并且必须恢复，则需要重播它们。此设置控制这些操作的最大总大小，以防止恢复花费太长时间。一旦达到最大大小，将进行刷新，从而生成新的 Lucene 提交点。默认为 **512mb** 。

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-translog.html
