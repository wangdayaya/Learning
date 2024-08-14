



# 下载 

根据自己的硬件选择合适的安装包进行下载，我这边是 linux x86 版本：

* Elasticsearch 8.0.0 ：https://www.elastic.co/downloads/past-releases/elasticsearch-8-0-0
* Kibana 8.0.0：https://www.elastic.co/downloads/past-releases/kibana-8-0-0

# Elasticsearch 

### 解压
将下载好的压缩包进行解压，可以的到文件夹


	elasticsearch-8.0.0

### 修改 elasticsearch.yml

	cluster.name: qjfy
	node.name: node-1
	bootstrap.memory_lock: true
	network.host: localhost
	http.port: 9400
### 配置

这里就是一些常见的配置，创建非 root 用户，修改目录所有用户，修改系统设置等等，这里不赘述。

### 启动 

切换自己创建的非 root 用户，我这里是 es ，然后在 ES 主目录下命令行启动：

	 ./bin/elasticsearch
	 
稍等片刻，出现以下信息，分别是：

*  8.x 自动开启了安全设置，给出了 elastic 用户的初始密码，可以使用命令进行修改
 	
	 	 bin/elasticsearch-reset-password -u elastic -i 
	

* 另外给出了 HTTP CA 证书


* 如果需要安装 kibana （下面的章节会讲到），我们只需要启动 kibana 点击给出的网址之后将这里给出的一长串 token 拷贝进去即可，切记只有 30 分钟有效，如果超时，执行命令重新生成即可，执行命令即可返回一长串新的 token

	   	./bin/elasticsearch-create-enrollment-token -s kibana 


* 如果想要其他节点加入集群，按照以下的操作进行即可
	
		Elasticsearch security features have been automatically configured!
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

	cluster.name: qjfy
	node.name: node-1
	bootstrap.memory_lock: true
	network.host: localhost
	http.port: 9400	
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

	
### 浏览器查看

在浏览器中输入以下网址，并且将账号和密码输入即可看到成功部署的界面：

	https://localhost:9400


	
# Kibana 

### 解压

将下载好的压缩包进行解压，可以的到文件夹
	
	kibana-8.0.0

### 修改 kibana.yml

	server.port: 5601
	server.host: "localhost"
	elasticsearch.hosts: ["http://localhost:9400"]
### 配置 kibana

这里也是常见的创建非 root 用户、修改目录所属用户等操作，不做赘述。
### 启动

这里如果我们直接启动 kibana ，会爆错（如下），因为 kibana 默认是在能打开浏览器的操作系统上运行的，linux 一般没有浏览器无法进行到这一步，所以没有正确的权限去安全访问上面已经有了安全配置的 elasticsearch ：

	[ERROR][elasticsearch-service] Unable to retrieve version information from Elasticsearch nodes. 
	
这是一个大坑，我在这里耗了两天的时间，因为现在网上的文档都是说因为 elasticsearch.yml 的安全配置关掉，这简直就是没脑子的人才想的出来的办法，这么做的话虽然能启动 kibana ，但是 elasticsearch 岂不是在公网果奔了，肯定不是这么改的，我在网上找了各种办法，最后在这里找到了解决办法，详情看网页：

	https://www.elastic.co/guide/en/elasticsearch/reference/current/configuring-stack-security.html#stack-start-with-security
	
我们在没有浏览器的 linux 服务器上，想要 kibana 能连接到 elasticsearch ，要先通过以下命令行，将上面生成的 enrollment token 传入 kibana 中，如果超时重新生成一个即可：

	bin/kibana-setup --enrollment-token eyJ2ZXIiOiI4LjAuMCIsImFkciI6WyIxMjcuMC4wLjE6OTIwMCIsIls6OjFdOjkyMDAiXSwiZmdyIjoiZDVmOTdlODI5ZDA5NWM4OWE4ZWViMDNkZjZiMTc3OTJhOWUwNzNlNWE4NTQ0ODI1ODY5N2I2NDdkYTdhNzUyYiIsImtleSI6IjJSZ1Z5SUVCVkFIZWZJc3JCZXd2OmQ1MUtlN0ZpUnRLYk56SU9Dd2lURGcifQ==
	
然后终端会打印如下信息，表示 kibana 连接 elasticsearch 安全配置成功：

	Kibana configured successfully!
	
	To start Kibana run:
		bin/kibana

这时候我们正常运行 kibana 即可启动，然后在浏览器中输入网址 http://localhost:5601 ，还有 elastic 账户和对应的密码即可正常进入 kibana 界面。

### 回看 kibana.yml

和 elasticsearch.yml 一样，后半部分都是系统自动添加的安全配置