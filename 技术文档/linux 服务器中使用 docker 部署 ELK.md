## 准备

	安装好 docker
	下载好 jdk-11.0.11_linux-x64_bin.tar.gz ，和 Dockerfile 放到一个目录下

## Dockerfile

	FROM centos
	# 作者
	MAINTAINER sam
	# 设置 docker 工作目录
	WORKDIR /usr/local/share/applications
	
	# 配置 java 环境
	COPY jdk-11.0.11_linux-x64_bin.tar.gz .
	RUN yum -y install wget && yum install -y unzip zip \ 
		&& tar -xvf jdk-11.0.11_linux-x64_bin.tar.gz \
		&& rm jdk-11.0.11_linux-x64_bin.tar.gz \
		&& mv jdk-11.0.11 jdk11
	ENV JAVA_HOME /usr/local/share/applications/jdk11/ 
	ENV CLASSPATH $:CLASSPATH:$JAVA_HOME/lib 
	ENV PATH $PATH:$JAVA_HOME/bin
	
	# 创建 es 用户
	RUN adduser es
	
	# 下载 elasticsearch 和 pinyin 和 ik ，并将 ik 和 pinyin 两个插件放到指定位置 
	RUN yum install -y expect && yum -y install wget && yum install -y unzip zip \
		&& wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.3.0-linux-x86_64.tar.gz \
		&& tar -xvf elasticsearch-7.3.0-linux-x86_64.tar.gz \
		&& rm elasticsearch-7.3.0-linux-x86_64.tar.gz \
		&& mv elasticsearch-7.3.0 elasticsearch \
		&& mkdir /usr/local/share/applications/elasticsearch/plugins/pinyin \
		&& wget https://github.com/medcl/elasticsearch-analysis-pinyin/releases/download/v7.3.0/elasticsearch-analysis-pinyin-7.3.0.zip \
		&& unzip -d /usr/local/share/applications/elasticsearch/plugins/pinyin elasticsearch-analysis-pinyin-7.3.0.zip   \
		&& rm elasticsearch-analysis-pinyin-7.3.0.zip \
		&& unzip ik.zip \
		&& wget https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v7.3.0/elasticsearch-analysis-ik-7.3.0.zip \
		&& unzip elasticsearch-analysis-ik-7.3.0.zip \
		&& mv elasticsearch-analysis-ik-7.3.0 ik \
		&& mv ik /usr/local/share/applications/elasticsearch/plugins/ \
		&& rm elasticsearch-analysis-ik-7.3.0.zip
	 
	# 修改 elasticsearch.yml 和 ik 需要的的配置文件
	RUN echo -e 'node.name: node-1\nnetwork.host: 0.0.0.0\nhttp.port: 9200\ncluster.initial_master_nodes: ["node-1"]\nxpack.security.enabled: true\nxpack.security.transport.ssl.enabled: true\nxpack.security.transport.ssl.verification_mode: certificate\nxpack.security.transport.ssl.keystore.path: certs/elastic-certificates.p12\nxpack.security.transport.ssl.truststore.path: certs/elastic-certificates.p12' >/usr/local/share/applications/elasticsearch/config/elasticsearch.yml \
		&& sed -i '$i\\tpermission java.net.SocketPermission "*", "connect,resolve";' /usr/local/share/applications/jdk11/lib/security/default.policy
	# 工作目标切换到 elasticsearch 下
	WORKDIR /usr/local/share/applications/elasticsearch
	# 生成证书并复制到指定目录
	RUN echo -e '#!/bin/bash/expect\nspawn /usr/local/share/applications/elasticsearch/bin/elasticsearch-certutil ca\nexpect "Please enter the desired output file*" {send "\\r"}\nexpect "Enter password for elastic*" {send "\\r"}\nspawn /usr/local/share/applications/elasticsearch/bin/elasticsearch-certutil cert --ca elastic-stack-ca.p12\nexpect "Enter password for CA*" {send "\\r"}\nexpect "Please enter the desired output file*" {send "\\r"}\nexpect "Enter password for elastic*" {send "\\r"}\nexpect eof' >/usr/local/share/applications/elasticsearch/createEsCerts.sh \
		&& expect /usr/local/share/applications/elasticsearch/createEsCerts.sh \
		&& mkdir /usr/local/share/applications/elasticsearch/config/certs/ && cp /usr/local/share/applications/elasticsearch/elastic-certificates.p12 /usr/local/share/applications/elasticsearch/config/certs/
	# 设置密码并创建索引
	RUN echo -e '#!/bin/bash/expect\nspawn /usr/local/share/applications/elasticsearch/bin/elasticsearch-setup-passwords interactive\nexpect "Please confirm that you would like to continue" {send "y\\r"}\nexpect "Enter*" {send "qjfy123456\\r"}\nexpect "Reenter*" {send "qjfy123456\\r"}\nexpect "Enter*" {send "qjfy123456\\r"}\nexpect "Reenter*" {send "qjfy123456\\r"}\nexpect "Enter*" {send "qjfy123456\\r"}\nexpect "Reenter*" {send "qjfy123456\\r"}\nexpect "Enter*" {send "qjfy123456\\r"}\nexpect "Reenter*" {send "qjfy123456\\r"}\nexpect "Enter*" {send "qjfy123456\\r"}\nexpect "Reenter*" {send "qjfy123456\\r"}\nexpect "Enter*" {send "qjfy123456\\r"}\nexpect "Reenter*" {send "qjfy123456\\r"}\nexpect eof' >/usr/local/share/applications/elasticsearch/setSecurity.sh \
		&& echo -e '#!/bin/bash\ncd /usr/local/share/applications/elasticsearch/\nsu es -c "nohup /usr/local/share/applications/elasticsearch/bin/elasticsearch  &"\necho "starting es ..."\nsleep 30\necho "es successfully started"\nsu es -c "expect /usr/local/share/applications/elasticsearch/setSecurity.sh"\necho "security configuration successfully"\ncurl -XPUT "http://localhost:9200/your_index_name" -u elastic:qjfy123456 -H "Content-Type: application/json" -d"你的索引 mapping"\necho "create gree_engine successfully"\necho "starting kibana"\ncd /usr/local/share/applications/kibana\nsu es -c "nohup /usr/local/share/applications/kibana/bin/kibana &"\nsleep 30\necho "kibana successfully started"\necho "starting logstash"\ncd /usr/local/share/applications/logstash\nnohup /usr/local/share/applications/logstash/bin/logstash -f /usr/local/share/applications/logstash/config/green_engine.conf &\nsleep 30\necho "logstash successfully started"'>/usr/local/share/applications/init.sh 
	# 将 elasticsearch 及其下的所有文件的用户都改为 es
	RUN chown -R es /usr/local/share/applications/elasticsearch/ 
	# 切换到 applications 目录
	WORKDIR /usr/local/share/applications
	
	# 下载 kibana
	RUN wget https://artifacts.elastic.co/downloads/kibana/kibana-7.3.0-linux-x86_64.tar.gz \
		&& tar -xvf kibana-7.3.0-linux-x86_64.tar.gz \
		&& mv kibana-7.3.0-linux-x86_64 kibana \
		&& rm kibana-7.3.0-linux-x86_64.tar.gz 
	# 配置 kibana.yml 
	RUN echo -e 'server.host: "0.0.0.0"\nelasticsearch.hosts: ["http://宿主机 ip :9999"]\nelasticsearch.username: "kibana"\nelasticsearch.password: "qjfy123456"'>/usr/local/share/applications/kibana/config/kibana.yml
	# 将 kibana 及其下的所有文件的用户都改为 es
	RUN chown -R es /usr/local/share/applications/kibana/
	
	# 下载 logstash 及其所用到的包，从 pg 数据库中取数据导入到 elasticsearch
	RUN yum install -y wget \ 
		&& wget https://artifacts.elastic.co/downloads/logstash/logstash-7.3.0.tar.gz \
		&& tar -xvf logstash-7.3.0.tar.gz \
		&& mv logstash-7.3.0 logstash \
		&& rm logstash-7.3.0.tar.gz \
		&& wget https://jdbc.postgresql.org/download/postgresql-42.2.19.jar \
		&& mv postgresql-42.2.19.jar logstash/logstash-core/lib/jars 
	# 配置文件
	RUN echo -e "使用 logstash 向 elasticsearch 中导入数据时候执行的 sql 语句" >/usr/local/share/applications/logstash/config/green_engine.sql \
		&& echo -e 'input {\n    stdin {\n    }\n    jdbc {\n        jdbc_driver_library => "/usr/local/share/applications/logstash/logstash-core/lib/jars/postgresql-42.2.19.jar"\n        jdbc_driver_class => "org.postgresql.Driver"\n        jdbc_connection_string => "jdbc:postgresql://pg 的 ip:5432/spatial_semantic"\n        jdbc_user => "postgres"\n        jdbc_password => "qjfy_gis@Hky.com"\n        use_column_value => "true"\n        tracking_column => "update_time"\n        tracking_column_type => "timestamp"\n        jdbc_paging_enabled => "true"\n        jdbc_page_size => "10000"\n        record_last_run => "true"\n        last_run_metadata_path => "./last_run"\n        clean_run => "false"\n        jdbc_validate_connection => "true"\n        schedule => "*/30 * * * * *"\n        statement_filepath => "/usr/local/share/applications/logstash/config/green_engine.sql"\n    }\n}\nfilter {\n        ruby {\n                code => "event.timestamp.time.localtime"\n        }\n        mutate {\n                copy => { "guid" => "[@metadata][_id]"}\n                remove_field => [ "@timestamp", "@version"]\n        }\n}\noutput {\n    stdout {\n    }\n    elasticsearch {\n        hosts => ["10.5.1.101:9999"]\n        index => "green_engine"\n        document_type => "_doc"\n        user => "elastic"\n        password => "qjfy123456"\n        document_id => "%{[@metadata][_id]}"\n    }\n}\n'>/usr/local/share/applications/logstash/config/green_engine.conf
	

## 运行容器
	
	
	docker build -t myImages .
	docker run -itd -p 9999:9200 -p 9998:5601 myImages /bin/bash
	docker exec -it 容器id /bin/bash
	
	进入容器，执行脚本，启动 es 、kibana 、logstash 创建索引
	/bin/bash /usr/local/share/applications/init.sh
	

	
	
	
	
