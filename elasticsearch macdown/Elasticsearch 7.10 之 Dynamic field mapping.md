## Elasticsearch 7.10 之 Dynamic field mapping

默认情况下，当在文档中找到以前看不见的字段时，Elasticsearch 会将新字段添加到类型映射中。通过将 [dynamic](https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic.html) 参数设置为 false （忽略新字段）或 strict（如果遇到未知字段，则引发异常），可以在文档和对象级别禁用此行为。

假设启用了动态字段映射，则使用一些简单的规则来确定字段应具有的数据类型：

|  JSON data type   | Elasticsearch data type  |
|  ----      | ----  |
| null | 没有添加任何字段 |
| true or false  | boolean 字段 |
| floating point number  | float 字段 |
| integer  | long 字段 |
| object  | object 字段 |
| array  | 取决于数组中的第一个非空值 |
| string  | date 字段（如果该值通过日期检测），double 或 long 字段（如果该值通过数字检测）或带有 keyword 子字段的 text 字段 |


这些是唯一可以动态检测的字段数据类型。所有其他数据类型必须显式映射。

除了下面列出的选项之外，还可以使用 [dynamic_templates](https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic-templates.html) 自定义动态字段映射规则。

### Date detection

如果启用了 **date\_detection**（默认），则将检查新的字符串字段，以查看其内容是否与 dynamic\_date_formats 中指定的任何日期模式匹配。如果找到匹配项，则会添加具有相应格式的新日期字段。

dynamic\_date\_formats 的默认值为：[ "strict_date_optional_time","yyyy/MM/dd HH:mm:ss Z||yyyy/MM/dd Z"]

例如：

	PUT my-index-000001/_doc/1
	{
	  "create_date": "2015/09/02"
	}
	
	GET my-index-000001/_mapping    # create_date 字段已添加为日期字段，其格式为："yyyy/MM/dd HH:mm:ss Z||yyyy/MM/dd Z"
 


### Disabling date detection

可以通过将 date_detection 设置为 false 来禁用动态日期检测：

	PUT my-index-000001
	{
	  "mappings": {
	    "date_detection": false
	  }
	}
	
	PUT my-index-000001/_doc/1    # create_date 字段已添加为文本字段
	{
	  "create": "2015/09/02"
	}
 


### Customising detected date formats

另外，可以自定义 dynamic\_date\_formats 以支持您自己的日期格式：

	PUT my-index-000001
	{
	  "mappings": {
	    "dynamic_date_formats": ["MM/dd/yyyy"]
	  }
	}
	
	PUT my-index-000001/_doc/1
	{
	  "create_date": "09/25/2015"
	}	
 
### Numeric detection

尽管 JSON 支持本地浮点数和整数数据类型，但某些应用程序或语言有时可能会将数字呈现为字符串。通常，正确的解决方案是显式映射这些字段，但是可以启用数字检测（默认情况下处于禁用状态）以自动执行此操作：

	PUT my-index-000001
	{
	  "mappings": {
	    "numeric_detection": true
	  }
	}
	
	PUT my-index-000001/_doc/1
	{
	  "my_float":   "1.0",    # 将 my_float 字段添加为 float 字段
	  "my_integer": "1"    # my_integer 字段被添加为 long 字段
	}
 

详情见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/dynamic-field-mapping.html
