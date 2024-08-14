## Elasticsearch 7.10 之 Merge

Elasticsearch 中的分片是 Lucene 索引，Lucene 索引分为多个部分。段是索引中存储索引数据的内部存储元素，并且是不可变的。较小的段会定期合并为较大的段，以保持索引大小不变并清除删除的。

合并过程使用自动限制来平衡合并和其他活动（例如搜索）之间硬件资源的使用。

### Merge scheduling

合并调度程序（ConcurrentMergeScheduler）在需要时控制合并操作的执行。合并在单独的线程中运行，并且当达到最大线程数时，直到合并线程可用时才会继续进行进一步的合并，。

合并调度程序支持以下动态设置：

**index.merge.scheduler.max\_thread_count**

单个分片上可能一次合并的最大线程数。默认为 **Math.max(1, Math.min(4, \<\<node.processors, node.processors>> / 2))** ，对于良好的固态磁盘（SSD）来说效果很好。如果您的索引是在旋转盘片驱动器上，则将其减小到 1 。


详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-merge.html
