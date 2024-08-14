
### 安装 es 节点

**准备：jdk1.8 安装包、es 7.3 安装包、kibana 7.3 安装包、ik、pinyin、elasticsearch-head 插件等**

假设有三个节点的 host 分别为 ：100.100.100.61、100.100.100.65、100.100.100.66

1. 在 100.100.100.66 服务器上面安装 **java 1.8** ，并在 **/etc/profile** 文件最后加入下面配置，保存后执行 **source /etc/profile** 生效，并查看 **java -version** 确认是否安装成功：
	
		export JAVA_HOME=/data/jdk
		export PATH=$PATH:$JAVA_HOME/bin
		export CLASSPATH=.:$JAVA_HOME/lib/tools.jar:$JAVA_HOME/lib/dt.jar:$CLASSPATH

2. 添加用户 es

		useradd es
		
3. 用 jar 包安装 es ，并且配置 elasticsearch.yml 

		cluster.name: qjfy
		node.name: node_66
		node.master: true
		node.data: true
		path.data: /usr/local/data
		path.logs: /usr/local/logs
		http.port: 9200
		transport.tcp.port: 9300
		network.host: 0.0.0.0
		discovery.seed_hosts: ["100.100.100.66:9200", "100.100.100.61:9200", "100.100.100.65:9200"]
		cluster.initial_master_nodes: ["node_66","node_65","node_61"]
		
4. 配置 jvm.options
	
		-Xms4g   
		-Xmx4g 
	
		
5. 在 root 权限下修改 /etc/sysctl.conf ，在最后加入

		vm.max_map_count=655360    # 限制一个进程拥有虚拟内存区域的大小
	
	保存后执行
	
		sudo sysctl -p
		
6. 在 root 权限下修改 /etc/security/limits.conf，添加以下两行
	
		* hard nofile 65536
		* soft nofile 65536
		* 
7. 在 root 权限下修改 data 读写权限和用户
	
		chmod 777 /usr/local/data（可选）
		chmod 777 /usr/local/logs（可选）
		chown -R es /usr/local/data
		chown -R es /usr/local/logs


8. 切换到 es 用户下启动 es ，浏览器输入以下命令检查是否安装成功

		100.100.100.66:9200	

10. 其他两个数据节点 100.100.100.61 和 100.100.100.65 服务器上除 elasticsearch.yml 中 node.name 属性不同，其他操作一样

	
参数配置有不懂的地方可参考：https://blog.csdn.net/xiaoge335/article/details/100575925 中的解释，或者直接看官方文档解释
    
### 100.100.100.66 安装 kibana 
		
1. 用 jar 包安装 kibana ，配置 kibana.yml

	server.host: "0.0.0.0"
	elasticsearch.hosts:["http://100.100.100.66:9200"]
	
2. 修改 kibana 目录用户

		chown -R es /usr/local/kibana-7.3.0
		
3. 切换到 es 用户，先启动 es ，再启动 kibana ，在浏览器中输入以下命令，检查 kibana 是否成功安装

		100.100.100.66:5601
	
### 100.100.100.66 节点上安装 head
1. 在 git 上面下载 git://github.com/mobz/elasticsearch-head.git  解压后的文件上传到 100.100.100.66 的 /usr/local 文件夹下，并改名为 head
2. 使用 uname -a  命令查看服务器系统位数，去 http://nodejs.cn/download/ 网页下下载对应的 nodejs tar 包，解压后放到 /usr/local ，文件夹名字改为 node 
3. 建立软连接，变为全局
	
		ln -s /usr/local/node/bin/npm /usr/local/bin/ 
		ln -s /usr/local/node/bin/node /usr/local/bin/

	使用 node -v 和 npm -v 检查是否安装成功
4. 在root 权限下，进入 head 文件夹，配置 npm 的代理、镜像、安装 grunt、安装插件（这个过程有点迷，总之能跑通，有懂的人可以评论告我）

		npm config set proxy http://代理 ip:端口 （不需要代理可跳过）
		npm config set registry http://registry.npm.taobao.org
		npm install cnpm
		npm install -g grunt-cli
		npm install  # 此时会报错，报错什么版本，就安装对应什么版本，我的是 npm install phantomjs-prebuilt@2.1.16 --ignore-scripts ，然后在执行 npm install
		
		
5. 在  Gruntfile.js 下的 connect.server.options 下增加

		hostname: "*",
		
	在 _site/app.js 文件中的
	
		this.base_uri = this.config.base_uri || this.prefs.get("app-base_uri") || "http://localhost:9200";
		
	改为 
		
		this.base_uri = this.config.base_uri || this.prefs.get("app-base_uri") || "http://100.100.100.66:9200";
	
		
6. 在安装 node 的节点 100.100.100.66 服务器上修改 usr/local/elasticsearch/config/elasticsearch.yml ，加入以下内容允许跨域

		http.cors.enabled: true
		http.cors.allow-origin: "*"
		
7. 重启 es ，在 head 文件夹下运行以下命令，浏览器访问 100.100.100.66:9100 ，如果集群健康值为 green 则表示安装成功，使用即可。

		npm run start
		nohup npm run start  & (后台运行)
		
8. （可选）如果 head 插件中集群状态为未连接，则用浏览器控制台查看跨域 CORS 报错的字段是什么,如 key ，见下报错信息

		CORS policy: Request header field key is not allowed by Access-Control-Allow-Header in preflight response
	
	在 100.100.100.66 的 elasticsearch.yml 中添加如下配置
	
		http.cors.allow-headers: "key"
		
	重启 es 和 node 即可，如果仍然报此错，再将报错的字段追加到上述配置中，重复过程直到集群健康值为 green 

		http.cors.allow-headers: "key,content-type,altitoken"
	
9. 集群运行过程中，如果停掉其中一个 es 节点，不影响数据的使用，重启也会自动连入集群。
10. 如果想使用 head 通过其他节点查看集群情况，则需要配置第 5 步到最后的类似操作