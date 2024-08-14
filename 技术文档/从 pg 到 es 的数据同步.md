## 从 pg 到 es 的数据同步


1. pg 中创建一张表，保证往 es 中导数据的字段都在其内存在，且 pg 中的表必须要有一个随着行数据插入或者更新而更新到最新时间的时间戳字段，一般用触发器即可满足所需，如本文中的 update_time 。
2. 安装 es 和 Logstash 
3. 下载驱动，可参考：https://jdbc.postgresql.org/download.html， 我的是 postgresql-42.2.14.jar ，放入 logstash-7.3.0/logstash-core/lib/jars 下面
4. 在 logstash-7.3.0/config 下面新建文件 green_engine.conf

		
		input {
		    stdin {
		 
		 
		    }
		    jdbc {
		        jdbc_driver_library => '/nas/logstash-7.3.0/logstash-core/lib/jars/postgresql-42.2.14.jar'      # pg 的驱动 jar 包存放位置
		        jdbc_driver_class => 'org.postgresql.Driver'
		        jdbc_connection_string => 'jdbc:postgresql://pg 数据库 ip:5432/spatial_semantic'
		        jdbc_user => 'pg 用户名'
		        jdbc_password => 'pg 密码'
		        use_column_value => 'true'
		        tracking_column => 'update_time'    # 跟踪该时间戳字段，保证在表中存在，并随着数据的插入和更新而更新到最新的时间戳
		        tracking_column_type => 'timestamp'
		        jdbc_paging_enabled => "true"
		        jdbc_page_size => "10000"
		        record_last_run => 'true'
		        last_run_metadata_path => './last_run'         # 记录最新的时间戳
		        clean_run => 'false'
		        jdbc_validate_connection => 'true'
		        schedule => '*/5 * * * * *'            # 每 5 秒执行一次
		        statement => "SELECT * FROM address WHERE update_time > :sql_last_value AND update_time  < NOW() ORDER BY update_time ASC"       # 现将 update_time 升序排序找到最近一次的时间戳，然后将在其之后到现在的数据查找出来，用于倒入到 es 中对应的索引中
		    }
		}
		 
		filter {
		        ruby {
		                code => 'event.timestamp.time.localtime'
		        }
		        mutate {
		                copy => { 'guid' => '[@metadata][_id]'}
		                remove_field => [ '@timestamp', '@version']
		        }
		}
		 
		output {
		    stdout {
		 
		 
		    }
		    elasticsearch {
		        hosts => ['es host:9200']
		        index => '索引名'
		        document_type => '_doc'
		        user => 'es 账户'
		        password => 'es 密码'
		        document_id => '%{[@metadata][_id]}'
		    }
		}

5. 在 es 中创建好对应的索引，其中的字段数据来源于数据库中对应的字段数据
6. 保存之后，执行如下命令，执行完之后，查看 es 中如果有数据导入则成功，否则失败
	
		./bin/logstash -f ./config/green_engine.conf