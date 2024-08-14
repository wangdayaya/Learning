## Elasticsearch 7.10 之 Index management

Kibana 的 **Index Management** 功能是管理集群的索引、数据流和索引模板的简便方法。练习良好的索引管理可确保您的数据以尽可能最具成本效益的方式正确存储。

### What you’ll learn

您将学习如何：

* 查看和编辑索引设置。
* 查看索引的映射和统计信息。
* 执行索引级操作，例如刷新和冻结。
* 查看和管理数据流。
* 创建索引模板以自动配置新的数据流和索引。

### Required permissions
如果使用 Elasticsearch 安全功能，则需要以下安全特权：

* **monitor** 集群特权，可访问 Kibana 的 **Index Management** 功能。
* **view\_index\_metadata** 和 **manage** 索引特权以查看数据流或索引的数据。
* **manage\_index\_templates** 集群特权用于管理索引模板。

要在 Kibana 中添加这些特权，请转到 **Stack Management > Security > Roles** 。

### View and edit indices
打开 Kibana 的主菜单，然后单击  **Stack Management > Index Management** 。

![索引管理界面](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/images/index-mgmt/management_index_labels.png)

**Index Management** 页面包含索引的概述。徽章指示索引是 [frozen](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/frozen-indices.html) 、 [follower](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/ccr-put-follow.html) 还是 [rollup](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/rollup-get-rollup-index-caps.html) 。

单击徽章将列表缩小为仅该类型的索引。您也可以使用搜索栏过滤索引。

您可以深入研究每个索引以研究索引的设置、映射和统计信息。在此视图中，您还可以编辑索引设置。

![索引管理界面](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/images/index-mgmt/management_index_details.png)

### Perform index-level operation
使用 **Manage** 菜单执行索引级操作。在索引详细信息视图中，或者在概述页面上选择一个或多个索引的复选框时，此菜单可用。该菜单包括以下操作：

* [Close index](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/indices-close.html)
* [Force merge index](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/indices-forcemerge.html)
* [Refresh index](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/indices-refresh.html)
* [Flush index](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/indices-flush.html)
* [Freeze index](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/freeze-index-api.html)
* [Delete index](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/indices-delete-index.html)
* Add [lifecycle policy](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/set-up-lifecycle-policy.html)

### Manage data streams
**Data Streams** 视图列出了您的数据流，并允许您检查或删除它们。

要查看有关数据流的更多信息，例如其生成或当前索引生命周期策略，请单击数据流的名称。

![数据流详细信息](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/images/index-mgmt/management_index_data_stream_stats.png)

要查看有关流的后备索引的信息，请单击 **Indices** 列中的数字。

![支持指数](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/images/index-mgmt/management_index_data_stream_backing_index.png)

### Manage index templates
**Index Templates** 视图列出了您的模板，并允许您检查、编辑、克隆和删除它们。对索引模板所做的更改不会影响现有索引。

![索引模板](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/images/index-mgmt/management-index-templates.png)
如果没有任何模板，则可以使用 **Create template** 向导创建一个模板。

#### Try it: Create an index template
在本教程中，您将创建一个索引模板，并使用它来配置两个新索引。

##### Step 1. Add a name and index pattern

1. 在 **Index Templates** 视图中，打开 **Create template** 向导。

	![创建向导](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/images/index-mgmt/management_index_create_wizard.png)

2. 在 **Name** 字段中，输入 **my-index-template** 。
3. 将 **Index pattern** 设置为 **my-index-*** ，以便模板将任何与该索引模式匹配的索引匹配。
4. 将 **Data Stream** 、**Priority** 、 **Version** 、**_meta field** 字段保留为空白或保持原样。

##### Step 2. Add settings, mappings, and index aliases

1. 将组件模板添加到索引模板。

	组件模板是预配置的映射、索引设置和索引别名集，您可以在多个索引模板之间重用。标记指示组件模板是否包含映射（M），索引设置（S），索引别名（A）或这三者的组合。

	组件模板是可选的。对于本教程，请勿添加任何组件模板。

	![组件模板页面](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/images/index-mgmt/management_index_component_template.png)

2. 定义索引设置。这些是可选的。对于本教程，请将本节留空。
3. 定义一个映射，其中包含一个名为 geo 的对象字段和一个名为 coordinates 的子地理点字段：

	![映射字段页面](https://www.elastic.co/guide/en/elasticsearch/reference/7.10/images/index-mgmt/management-index-templates-mappings.png)

	或者，您可以单击 Load JSON 链接并将映射定义为JSON：

		{
		  "properties": {
		    "geo": {
		      "properties": {
		        "coordinates": {
		          "type": "geo_point"
		        }
		      }
		    }
		  }
		}
您可以在动态模板和高级选项选项卡中创建其他映射配置。对于本教程，请勿创建任何其他映射。

4. 定义一个名为 my-index 的索引别名：

		{
		  "my-index": {}
		}	
	
5. 在查看页面上，查看摘要。如果一切正常，请单击 **Create template** 。

##### Step 3. Create new indices

现在，您可以使用索引模板创建新索引了。

1. 为以下文档建立索引以创建两个索引：**my-index-000001** 和 **my-index-000002**。

		POST /my-index-000001/_doc
		{
		  "@timestamp": "2019-05-18T15:57:27.541Z",
		  "ip": "225.44.217.191",
		  "extension": "jpg",
		  "response": "200",
		  "geo": {
		    "coordinates": {
		      "lat": 38.53146222,
		      "lon": -121.7864906
		    }
		  },
		  "url": "https://media-for-the-masses.theacademyofperformingartsandscience.org/uploads/charles-fullerton.jpg"
		}
		
		POST /my-index-000002/_doc
		{
		  "@timestamp": "2019-05-20T03:44:20.844Z",
		  "ip": "198.247.165.49",
		  "extension": "php",
		  "response": "200",
		  "geo": {
		    "coordinates": {
		      "lat": 37.13189556,
		      "lon": -76.4929875
		    }
		  },
		  "memory": 241720,
		  "url": "https://theacademyofperformingartsandscience.org/people/type:astronauts/name:laurel-b-clark/profile"
		}
	
2. 使用 get index API 查看新索引的配置。 使用先前创建的索引模板配置索引。

		GET /my-index-000001,my-index-000002
		
详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/7.10/index-mgmt.html
