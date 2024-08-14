## Elasticsearch 7.10 之 Maven Repository

The high-level Java REST client 托管在 Maven Central 上，所需的最低 Java 版本是 1.8。

The High Level REST Client 的发布周期与 Elasticsearch 相同。用所需的客户端版本替换该版本。

如果您正在寻找 SNAPSHOT 版本，则应将我们的快照存储库添加到您的 Maven 配置中：

	<repositories>
	    <repository>
	        <id>es-snapshots</id>
	        <name>elasticsearch snapshot repo</name>
	        <url>https://snapshots.elastic.co/maven/</url>
	    </repository>
	</repositories>
	
或在 Gradle 中：

	maven {
	        url "https://snapshots.elastic.co/maven/"
	}
	
### Maven configuration 

这是使用 maven 作为依赖项管理器来配置依赖项的方法。将以下内容添加到您的 pom.xml 文件中：

	<dependency>
	    <groupId>org.elasticsearch.client</groupId>
	    <artifactId>elasticsearch-rest-high-level-client</artifactId>
	    <version>7.10.2</version>
	</dependency>
	
### Gradle configuration

这是使用 gradle 作为依赖项管理器来配置依赖项的方法。将以下内容添加到您的 build.gradle 文件中：

	dependencies {
	    compile 'org.elasticsearch.client:elasticsearch-rest-high-level-client:7.10.2'
	}
	
### Lucene Snapshot repository

任何主要版本（如Beta）的最初发行版都可能建立在 Lucene Snapshot 版本的基础上。在这种情况下，您将无法解析客户端的 Lucene 依赖关系。

例如，如果要使用依赖于 Lucene 8.0.0-snapshot-83f9835的7.0.0-beta1 版本，则必须定义以下存储库。

对于 Maven：

	<repository>
	    <id>elastic-lucene-snapshots</id>
	    <name>Elastic Lucene Snapshots</name>
	    <url>https://s3.amazonaws.com/download.elasticsearch.org/lucenesnapshots/83f9835</url>
	    <releases><enabled>true</enabled></releases>
	    <snapshots><enabled>false</enabled></snapshots>
	</repository>
	
对于 Gradle：

	maven {
	    name 'lucene-snapshots'
	    url 'https://s3.amazonaws.com/download.elasticsearch.org/lucenesnapshots/83f9835'
	}

详情见官网：https://www.elastic.co/guide/en/elasticsearch/client/java-rest/current/java-rest-high-getting-started-maven.html
