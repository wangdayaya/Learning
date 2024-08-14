## Elasticsearch 7.10 之 Index recovery prioritization

只要有可能，就会按优先级顺序恢复未分配的分片。索引按优先级排序如下：

* 可选的 **index.priority** 设置（从高到低）
* 索引创建日期（从高到低）
* 索引名称（从高到低）

这意味着，默认情况下，较新的索引将在较旧的索引之前恢复。

使用每个索引的动态可更新 **index.priority** 设置来自定义索引优先级顺序。例如：

	PUT index_1
	
	PUT index_2
	
	PUT index_3
	{
	  "settings": {
	    "index.priority": 10
	  }
	}
	
	PUT index_4
	{
	  "settings": {
	    "index.priority": 5
	  }
	}
 
在上面的示例中：

* **index_3** 将首先被恢复，因为它具有最高的 **index.priority**
* 下一个将恢复 **index_4** ，因为它具有次高的优先级
* 下一个将恢复 **index_2** ，因为它是最近创建的
* **index_1** 将最后恢复

此设置可以接受整数，并且可以使用 [update index settings API](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-update-settings.html) 在实时索引上进行更新：

	PUT index_4/_settings
	{
	  "index.priority": 1
	}

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/recovery-prioritization.html#recovery-prioritization
