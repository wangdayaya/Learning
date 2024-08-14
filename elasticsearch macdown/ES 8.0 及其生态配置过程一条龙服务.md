# 背景

前两天长三角某一线城市的数据泄漏了，没有上热搜被压下去了，但是圈内传开了，我们 leader 给我打了个紧急电话，询问我们 ES 架构的安全性，尽管我说了我们使用的 7.x 我已经开启了安全配置，但是他还是恳求我升到最高版本，没办法只能升级了，顺便说一句数据泄漏真的是够过很严重，我在某网站上通过输入我的手机号查到了近几年的账号和住址信息等，更可怕的是这个网站如果付费能查到更详细的信息，细思极恐。




# 下载 

根据自己的硬件选择合适的安装包进行下载，我这边是 mac x86 版本：

* Elasticsearch 8.0.0 ：https://www.elastic.co/downloads/past-releases/elasticsearch-8-0-0
* Kibana 8.0.0：https://www.elastic.co/downloads/past-releases/kibana-8-0-0

# 解压

将下载好的两个压缩包进行解压，可以的到两个文件夹

	kibana-8.0.0
	elasticsearch-8.0.0
# Elasticsearch
### 配置 elasticsearch.yml
修改 elasticsearch 主目录下的  config/elasticsearch.yml 文件

	# ---------------------------------- Cluster -----------------------------------
	cluster.name: qjfy
	# ------------------------------------ Node ------------------------------------
	node.name: node-1
	# ----------------------------------- Memory -----------------------------------
	bootstrap.memory_lock: true
	# ---------------------------------- Network -----------------------------------
	network.host: localhost
	http.port: 9200


### 启动 elasticsearch


在 es 主目录中 ，运行 ：

	 ./bin/elasticsearch
	
稍等片刻，出现以下信息：

* 这里是为了说明 8.x 自动开启了安全设置，给出了 elastic 用户的初始密码，可以通过命令 bin/elasticsearch-reset-password -u elastic -i 重新修改
* 另外给出了 HTTP CA 证书
* 如果需要安装 kibana （下面的章节会讲到），我们只需要启动 kibana 点击给出的网址之后将这里给出的一长串 token 拷贝进去即可，切记只有 30 分钟有效，如果超时，执行命令重新生成即可，执行命令即可返回一长串新的 token

		 ./bin/elasticsearch-create-enrollment-token -s kibana 
* 如果想要其他节点加入集群，按照以下的操作进行即可
	
		✅ Elasticsearch security features have been automatically configured!
		✅ Authentication is enabled and cluster connections are encrypted.
		
		ℹ️  Password for the elastic user (reset with `bin/elasticsearch-reset-password -u elastic`):
		  OxhN+dqEpl+MR_UaVUgV
		
		ℹ️  HTTP CA certificate SHA-256 fingerprint:
		  d5f97e829d095c89a8eeb03df6b17792a9e073e5a85448258697b647da7a752b
		
		ℹ️  Configure Kibana to use this cluster:
		• Run Kibana and click the configuration link in the terminal when Kibana starts.
		• Copy the following enrollment token and paste it into Kibana in your browser (valid for the next 30 minutes):
		  eyJ2ZXIiOiI4LjAuMCIsImFkciI6WyIxMjcuMC4wLjE6OTIwMCIsIls6OjFdOjkyMDAiXSwiZmdyIjoiZDVmOTdlODI5ZDA5NWM4OWE4ZWViMDNkZjZiMTc3OTJhOWUwNzNlNWE4NTQ0ODI1ODY5N2I2NDdkYTdhNzUyYiIsImtleSI6IjJSZ1Z5SUVCVkFIZWZJc3JCZXd2OmQ1MUtlN0ZpUnRLYk56SU9Dd2lURGcifQ==
		
		ℹ️  Configure other nodes to join this cluster:
		• On this node:
		  ⁃ Create an enrollment token with `bin/elasticsearch-create-enrollment-token -s node`.
		  ⁃ Uncomment the transport.host setting at the end of config/elasticsearch.yml.
		  ⁃ Restart Elasticsearch.
		• On other nodes:
		  ⁃ Start Elasticsearch with `bin/elasticsearch --enrollment-token <token>`, using the enrollment token that you generated.



### 回看 elasticsearch.yml 

我们打开一个新的终端，再看配置文件 elasticsearch.yml  （如下所示），可以发现上面的部分是我们自己配置的内容，下面的部分系统自动给我们写入了一些 “SECURITY AUTO CONFIGURATION” 配置，这些内容就是系统自动默认为我们开启了 HTTP API 客户端连接开启加密，集群间加密传输和身份验证、自动加入集群等配置。这些在以前都是需要手动配置的，现在自动生成倒也方便。

	# ---------------------------------- Cluster -----------------------------------
	cluster.name: qjfy
	# ------------------------------------ Node ------------------------------------
	node.name: node-1
	# ----------------------------------- Memory -----------------------------------
	bootstrap.memory_lock: true
	# ---------------------------------- Network -----------------------------------
	network.host: localhost
	http.port: 9200
	
	#----------------------- BEGIN SECURITY AUTO CONFIGURATION -----------------------
	#
	# The following settings, TLS certificates, and keys have been automatically      
	# generated to configure Elasticsearch security features on 04-07-2022 07:19:54
	#
	# --------------------------------------------------------------------------------
	
	# Enable security features
	xpack.security.enabled: true
	
	xpack.security.enrollment.enabled: true
	
	# Enable encryption for HTTP API client connections, such as Kibana, Logstash, and Agents
	xpack.security.http.ssl:
	  enabled: true
	  keystore.path: certs/http.p12
	
	# Enable encryption and mutual authentication between cluster nodes
	xpack.security.transport.ssl:
	  enabled: true
	  verification_mode: certificate
	  keystore.path: certs/transport.p12
	  truststore.path: certs/transport.p12
	# Create a new cluster with the current node only
	# Additional nodes can still join the cluster later
	cluster.initial_master_nodes: ["node-1"]
	
	#----------------------- END SECURITY AUTO CONFIGURATION -------------------------
	
	
# Kibana
### 配置 kibana.yml

修改 kibana 主目录下面的 config/kibana.yml 文件 

	server.port: 5601
	server.host: "localhost"
	elasticsearch.hosts: ["http://localhost:9200"]

### 启动 Kibana 

在 kibana 主目录启动，命令：

	./bin/kibana

稍等片刻我们发现，会打印如下的信息，我们将链接 http://localhost:5601/?code=157557 拷贝到浏览器，然后将上面启动 es 时候得到的 token （如果超时按照上面的方法重新生成一个即可）拷贝进去，然后进行自动配置即可。

	i Kibana has not been configured.
	
	Go to http://localhost:5601/?code=157557 to get started.

如果配置成功，会出现登录页面，此时输入账号和密码即可，账号是 elastic ，密码就是上面章节自动生成的密码 OxhN+dqEpl+MR_UaVUgV 。

到此为止 ES 和 Kibana 成功启动了，当然了我这里只是单机版本，集群版本还需要摸索。

### 对比 7.x 和 8.x 配置

其实对于单节点来说，从配置上来看我没发现 elasticsearch 8.x 有比 7.x 更安全的配置，因为这些配置我在 7.x 的时候都手动配置了，所以我觉得 8.x 只是实现了“一键配置”这个功能，至于说在相同配置文件下 8.x 有比 7.x 更安全的内在防护或者对安全进行了升级，这个我没有研究不能下定论。

# IK
### 下载 IK 插件

从这里下载 8.0 版本的 ik 插件的源码

* 	 https://github.com/medcl/elasticsearch-analysis-ik/releases/tag/v8.0.0

### 修改源码

（细节太多，篇幅过长，如果有兴趣，我可以再写一篇）下载源码之后，导入 idea ，对 pom.xml 执行 reload project ，稍等片刻，然后对源码进行修改，主要目的是能够定时扫描连接的数据库中的词典，更新 IK 分词器所使用到的词库。然后 mvn package 打包得到的压缩包，解压之后将文件夹改为 ik 放入 ES 的 plugins 目录下。重启 ES 之后终端会打印一下信息，表明 ES 加载插件运行成功：

	[2022-07-05T15:07:13,619][INFO ][o.e.p.PluginsService     ] [node-1] loaded plugin [analysis-ik]
	...
	[2022-07-05T15:07:28,065][INFO ][stdout                   ] [node-1] 准备!
	[2022-07-05T15:07:28,212][INFO ][stdout                   ] [node-1] 连接服务器中的数据库成功!
	[2022-07-05T15:07:38,924][INFO ][stdout                   ] [node-1] 加载表中分词


### 测试分词器

在 Kibana 命令行中进行测试：

	GET _analyze
	{
	  "text": "杭州市文三路15号",
	  "analyzer": "ik_smart"
	}
结果可以看出来分词器已经加载了我数据库中的词表，对“杭州市文三路15号”进行了分词。

	{
	  "tokens" : [
	    {
	      "token" : "杭州市",
	      "start_offset" : 0,
	      "end_offset" : 3,
	      "type" : "CN_WORD",
	      "position" : 0
	    },
	    {
	      "token" : "文三路",
	      "start_offset" : 3,
	      "end_offset" : 6,
	      "type" : "CN_WORD",
	      "position" : 1
	    },
	    {
	      "token" : "15号",
	      "start_offset" : 6,
	      "end_offset" : 9,
	      "type" : "CN_WORD",
	      "position" : 2
	    }
	  ]
	}

# pinyin
因为我做的业务有用到汉语拼音插件的功能，所以这里也记录一下配置过程。

### 下载 pinyin 插件

从这里下载 pinyin 8.0.0 ：

* https://github.com/medcl/elasticsearch-analysis-pinyin/releases/tag/v8.0.0

### 配置

将下载好的压缩包解压得到文件夹改名为 pinyin ，并且放到 ES 的 plugins 下
将 pinyin 文件夹下的 plugin-descriptor.properties 中的 elasticsearch.version 改为 8.0.0 （和你的 ES 版本一样） ，然后重新启动 ES 加载改插件，打印如下信息表明加载成功：
	
	[2022-07-05T15:25:13,591][INFO ][o.e.p.PluginsService     ] [node-1] loaded plugin [analysis-pinyin]

### 测试
在 Kibana 终端中使用 pinyin 分析器测试
	
	GET _analyze
	{
	  "text": "杭州市",
	  "analyzer": "pinyin"
	}
	
可以看到打印出了相关汉字的单字拼音还有首字母的缩写。
	
	{
	  "tokens" : [
	    {
	      "token" : "hang",
	      "start_offset" : 0,
	      "end_offset" : 0,
	      "type" : "word",
	      "position" : 0
	    },
	    {
	      "token" : "hzs",
	      "start_offset" : 0,
	      "end_offset" : 0,
	      "type" : "word",
	      "position" : 0
	    },
	    {
	      "token" : "zhou",
	      "start_offset" : 0,
	      "end_offset" : 0,
	      "type" : "word",
	      "position" : 1
	    },
	    {
	      "token" : "shi",
	      "start_offset" : 0,
	      "end_offset" : 0,
	      "type" : "word",
	      "position" : 2
	    }
	  ]
	}
	
# dynamic-synonym

### 编译
从这里 clone 下来代码进入 idea 

	https://github.com/bells/elasticsearch-analysis-dynamic-synonym

然后进行编译打包 mvn package 得到压缩包，将该项目 target/releases 下的压缩包进行解压得到文件夹，改名为 dynamic-synonym 移动到 ES 的 plugins 下

### 配置

将 dynamic-synonym 文件夹下的 plugin-descriptor.properties 文件中的最下面的 ES 版本改为 8.0.0 ， 重启 ES ，如果出现下面日志说明 ES 加载插件成功：

	 [2022-07-05T15:52:35,707][INFO ][o.e.p.PluginsService     ] [node-1] loaded plugin [analysis-dynamic-synonym]
	...
	[2022-07-05T16:04:25,799][INFO ][dynamic-synonym          ] [node-1] start reload local synonym from /Users/wys/elasticsearch-8.0.0/config/synonyms.txt.



### 测试

在 ES 的 config 目录下创建一个存储同义词的文件 synonyms.txt ，写入一下内容：

	kfc,肯德基
	mc,麦当劳
	hz,杭州

在 Kibana 中进行测试，先要创建一个索引，里面有分词器，分词器中把加进去：

	PUT /test
	{
	  "settings":{
	    "analysis":{
	      
	      "filter": {
	          "my_synonym" : {
	            "type" : "dynamic_synonym",
	            "synonyms_path" : "synonyms.txt"
	          }
	      },
	      "analyzer":{
	        "ik_search_analyzer":{
	          "type": "custom",
	          "tokenizer":"ik_smart",
	          "filter": [ "my_synonym" ]
	        }
	      }
	    }
	  }
	}
	
	GET test/_analyze
	{
	  "text": "杭州肯德基",
	  "analyzer": "ik_search_analyzer"
	}

打印结果，可以看到先通过 ik_smart 进行分词，然后将同义词也都找了出来：

	{
	  "tokens" : [
	    {
	      "token" : "杭州",
	      "start_offset" : 0,
	      "end_offset" : 2,
	      "type" : "CN_WORD",
	      "position" : 0
	    },
	    {
	      "token" : "hz",
	      "start_offset" : 0,
	      "end_offset" : 2,
	      "type" : "SYNONYM",
	      "position" : 0
	    },
	    {
	      "token" : "肯德基",
	      "start_offset" : 2,
	      "end_offset" : 5,
	      "type" : "CN_WORD",
	      "position" : 1
	    },
	    {
	      "token" : "kfc",
	      "start_offset" : 2,
	      "end_offset" : 5,
	      "type" : "SYNONYM",
	      "position" : 1
	    }
	  ]
	}

随时可以在 synonyms.txt 中按照上面的格式加新的同义词， ES 会在几秒内重新加载同义词，并打印信息：

	[2022-07-05T16:05:25,802][INFO ][dynamic-synonym          ] [node-1] start reload local synonym from /Users/wys/elasticsearch-8.0.0/config/synonyms.txt.
	[2022-07-05T16:05:25,821][INFO ][dynamic-synonym          ] [node-1] success reload synonym
	
# 尾记

本文主要记录了安装 ES 8.0.0 、安装 Kibana 8.0.0 、安装 ik 8.0.0 、安装 pinyin 8.0.0 、安装 dynamic_synonym 7.16.0（可以当作 8.0.0 使用）等过程。