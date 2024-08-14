准备工作：三个节点 112 节点、105 节点和 103 节点已经成功安装并配置好了和集群有关的配置信息。


### 集群节点间加密传输


1.先进行证书生成

	./elasticsearch-certutil ca 

需要输入文件名，可回车跳过使用默认的文件名。
然后输入想设置的密码（如：123456） ，回车后在 es 主目录下会有 **elastic-stack-ca.p12** 文件生成

2.在 **config** 目录下面创建 **certs** 文件夹，在 es 主目录下运行

	bin/elasticsearch-certutil cert --ca elastic-stack-ca.p12 --dns *.*.*.* --ip *.*.*.112 --out config/certs/node_112.p12    （ cat /etc/resolv.conf 查看 dns）

需要输入两个不同的密码（我都是 123456），终端展示如下：
	
	Enter password for CA (elastic-stack-ca.p12) : 
	Enter password for node_112.p12 :


3.在 elasticsearch.yml 配置文件中追加

	xpack.security.enabled: true
	xpack.security.transport.ssl.enabled: true
	xpack.security.transport.ssl.keystore.path: certs/node_112.p12
	xpack.security.transport.ssl.truststore.path: certs/node_112.p12

然后执行
	
	./bin/elasticsearch-keystore create
	./bin/elasticsearch-keystore add xpack.security.transport.ssl.keystore.secure_password
	./bin/elasticsearch-keystore add xpack.security.transport.ssl.truststore.secure_password


4.执行命令为 103 节点、105 节点生成 certificates 
	
	./bin/elasticsearch-certutil cert --ca elastic-stack-ca.p12 --multiple
	
过程中会先后让你输入以下内容：

	Enter password for CA (elastic-stack-ca.p12) : 输入之前设置好的密码 123456 	
	Enter instance name: node_105
	Enter name for directories and files of node_105 [node_105]: node_105
	Enter IP Addresses for instance (comma-separated if more than one) []: *.*.*.105
	Enter DNS names for instance (comma-separated if more than one) []: （回车）
	Would you like to specify another instance? Press 'y' to continue entering instance information: y
	Enter instance name: node_103
	Enter name for directories and files of node_103 [node_103]: node_103
	Enter IP Addresses for instance (comma-separated if more than one) []: *.*.*.103
	Enter DNS names for instance (comma-separated if more than one) []: （回车）
	Would you like to specify another instance? Press 'y' to continue entering instance information: 
	Please enter the desired output file [certificate-bundle.zip]: （回车）
	Enter password for node_103/node_103.p12 : 输入密码 123456
	Enter password for node_105/node_105.p12 : 输入密码 123456


执行结束在 es 主目录生成一个 certificate-bundle.zip 

5.安装 unzip 工作，然后解压这个文件
	
	unzip certificate-bundle.zip

得到为两个节点生成的 node\_103 和 node\_105 文件夹下的 node\_103.p12 和 node\_105.p12 两个文件，分别拷贝到 103 和 105 config/certs 下


6.在 103 节点和 105 节点下分别都执行以下命令

	./bin/elasticsearch-keystore create （选 y）
	./bin/elasticsearch-keystore add xpack.security.transport.ssl.keystore.secure_password （输入 123456）
	./bin/elasticsearch-keystore add xpack.security.transport.ssl.truststore.secure_password （输入 123456）




### 集群设置密码

启动 es 之后在 103 节点（随便挑了一个节点）上设置密码

	./bin/elasticsearch-setup-passwords interactive
	
终端展示如下：

	future versions of Elasticsearch will require Java 11; your Java version from [/usr/local/jdk1.8.0_261/jre] does not meet this requirement
	Initiating the setup of passwords for reserved users elastic,apm_system,kibana,logstash_system,beats_system,remote_monitoring_user.
	You will be prompted to enter passwords as the process progresses.
	Please confirm that you would like to continue [y/N]y
	
	Enter password for [elastic]: （我的都是 123456）
	Reenter password for [elastic]: 
	Enter password for [apm_system]: 
	Reenter password for [apm_system]: 
	Enter password for [kibana]: 
	Reenter password for [kibana]: 
	Enter password for [logstash_system]: 
	Reenter password for [logstash_system]: 
	Enter password for [beats_system]: 
	Reenter password for [beats_system]: 
	Enter password for [remote_monitoring_user]: 
	Reenter password for [remote_monitoring_user]: 

设置完之后，访问 \*.\*.\*.103:9200 得登录账号 elastic 密码 123456，即可展示相关配置信息即表示成功，105 不用设置，直接用此账号密码也可以登录

具体步骤见官网：https://www.elastic.co/guide/en/elasticsearch/reference/current/encrypting-internode.html

### 报错解决或者注意事项

1.

	 join validation on cluster state with a different cluster uuid zsRDFPbBRx-8Rl98GJU8ug than local cluster uuid 55n6EVZySSSBSg_1Xc3Ouw, rejecting
	 
原因：因为之前 112 上运行过 es ，现在改了配置文件中的 cluster_name 之后，导致该节点加入不了我设置的集群，可能是由于之前生成的 data 文件夹造成了未知的影响

推测是因为该节点之前启动过 ES ，已经创建了 data 文件夹，与要加入的集群冲突。集群配置完成前建议不要启动单个 ES 实例。原因：默认参数启动会以单实例方式启动，创建各种文件夹、文件，可能干扰后续集群配置。

解决：删除data文件夹

2.还有就是内网跨组网也无法顺利建立集群


3.在浏览器输入 http://\*.\*.\*.112:9100/ 使用 head 插件没有效果，集群状态未连接，而且浏览器控制台报错

	Failed to load resource: the server responded with a status of 401 (Unauthorized)
 
只需要带上账号和密码
 
	http://*.*.*.112:9100/?auth_user=elastic&auth_password=123456

4.如果出现 

	Authentication of [elastic] was terminated by realm [reserved]

之类的问题，表示设置的密码失效，可能也是因为 data 文件夹中的索引有改动导致的，最粗暴的解决办法也是删除每个节点中的 data 文件夹之后重新创建，然后再任意一台节点使用

	./bin/elasticsearch-setup-passwords interactive
重新创建账号和密码之后，重新启动即可


5.保证相关的文件和文件夹都在相同的用户的权限下，如涉及到的 es、logs、data 等
