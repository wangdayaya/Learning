这是我参与更文挑战的第28天，活动详情查看： [更文挑战](https://juejin.cn/post/6967194882926444557)
# 什么是 Neo4j
Neo4j 是一个很流行的图数据库，和图形理论类似，有三个主要的概念：

* 节点
* 关系
* 属性

节点和关系都包含属性，而关系连接节点，且可以单向和双向。在 Neo4j 的直观页面上，节点用圆圈表示，关系用线和键头表示。

# Neo4j 的优缺点
优点：

* 很容易存储大量的连接数据
* 检索数据也很快速，结果很直观
* 上手比较容易， CQL 语言可读性强
* 数据量不过千万的话，性能基本能够保证

缺点：

* 只能是单机版本，如果有海量数据，没法做分布式
* 图数据结构导致写入性能差，实时性读写跟不上

# Neo4j 用途
常见于以下用途：

* 社交网络
* 推荐系统
* 欺诈分析
* Web 安全（垃圾邮件等等）

# Neo4j 安装
很简单，进入官网，自行解决: https://neo4j.com

# CQL
CQL 是 Neo4j 图形数据库的查询语言，下面通过案例来讲解主要的命令。
### CREATE 命令
CREATE 命令主要功能是：

* 创建节点
* 创建关系
* 为节点或者关系创建标签

创建一个名为 dept ，标签为 Dept ，并且有 dname 和 location 两个属性的节点，dept 只是一个类似变量的东西，可以命名为 d 或者其他字符串，不是该节点的属性：

	CREATE (dept:Dept { dname:"qjfy",location:"hky" })

创建一个名为 emp ，标签为 Employee ，并且有 name 和 dname 两个属性的节点，emp 只是一个类似变量的东西，可以命名为 e 或者其他字符串，不是该节点的属性：

	CREATE (emp:Employee{name:"wys",dname:"qjfy"})
	
创建一个名为 emp1 ，标签为 Employee 和 Actor 的没有属性的节点，emp1 只是一个类似变量的东西，可以命名为 e1或者其他字符串，不是该节点的属性：

	CREATE (emp1: Employee:Actor)
	


### MATCH 命令
MATCH 命令主要功能是：

* 获取节点、关系、属性的数据


一般 MATCH 命令需要和 RETURN 命令结合使用，不能单独使用 MATCH 命令，否则会报错 Neo.ClientError.Statement.SyntaxError 。

### RETURN 命令
RETURN 命令主要功能是：

* 检索节点的属性
* 检索节点和关联关系的属性

同样的，一般 MATCH 命令需要和 RETURN 命令结合使用，不能单独使用 RETURN 命令，否则会报错 Neo.ClientError.Statement.SyntaxError 。

###  MATCH & RETURN 命令
MATCH & RETURN 命令的主要功能是：

* 检索节点的属性
* 检索节点和关联关系的属性

返回标签为 Dept 的所有节点的属性 dname 和 location ，因为只有一个节点，所以只返回了一行数据。这里的 dept 类似于对象名，可以任意其他的字符串代替，如换成 d ：

	MATCH (dept: Dept) RETURN dept.dname,dept.location

返回结果：

|  dept.dname   | dept.location  |
|  ----  | ----  |
| "qjfy"  | "hky" |


或者 ：

	MATCH (d: Dept) RETURN d.dname,d.location

返回结果：

|  d.dname   | d.location  |
|  ----  | ----  |
| "qjfy"  | "hky" |


在不指定返回属性的时候，会显示图结构，如下图所示，这里多出来的一个 <id> 属性是数据库内部设置的，和 mysql 中的 id 类似：

	MATCH (dept: Dept) RETURN dept
	
![](/Users/wys/Desktop/NLP/Neo4j-match-return.png)

### ID 属性

<id> 是节点和关系的默认的内部属性。 当创建一个新的节点或关系时，其内部会分配一个数字，如果再创建其他新的节点或者关系，它会自动递增。 ID 上限约为 35 亿。

### 创建关系
1.我们已经有了 Dept 和  Employee 两个节点，现在创建没有属性的关系连接它们，关系标签为 WORK_AT ，关系名称为 r ：

	MATCH (e: Employee),(d: Dept) CREATE (e)-[r:WORK_AT]->(d) 

创建成功，然后使用查询语句查看关系：

	MATCH (e)-[r:WORK_AT]->(d) RETURN r

结果中 start 是  id  为 5026 的 Employee 标签的节点，end 是 id 为 5024 的 Dept 标签的节点：
	
	{
	  "identity": 14452,
	  "start": 5026,
	  "end": 5024,
	  "type": "WORK_AT",
	  "properties": {
	
	  }
	}

2.我们现在创建两个节点之间有 sal 属性的关系：

	MATCH (e: Employee),(d: Dept) CREATE (e)-[r:WORK_AT{sal:1000}]->(d) 

创建成功，然后使用查询语句查看关系：

	MATCH  (e)-[r:WORK_AT{sal:1000}]->(d) return r
	
结果：

	{
	  "identity": 14453,
	  "start": 5026,
	  "end": 5024,
	  "type": "WORK_AT",
	  "properties": {
	  "sal": 1000
	  }
	}

3.创建不存在于数据库中不带属性的的两个节点和关系：

	CREATE (w1:WeChatUser)-[like:LIKES]->(w2:WeChatUser) 

创建成功之后，使用查询语句查看关系：

	MATCH (w1)-[like:LIKES]->(w2) RETURN like

结果中 start 是  id  为 5027 的 WeChatUser 标签的节点，end 是 id 为 5028 的 WeChatUser 标签的节点：

	{
	  "identity": 14454,
	  "start": 5027,
	  "end": 5028,
	  "type": "LIKES",
	  "properties": {
	
	  }
	}

4.创建不存在于数据库中带属性的的两个节点和关系：

	CREATE (w1:WeChatUser{name:"wys"})-[like:LIKES{rating:100}]->(w2:WeChatUser{name:"ppy"}) 

创建成功之后，使用查询语句查看关系：

	MATCH (w1)-[like:LIKES{rating:100}]->(w2) RETURN like
	
结果中 start 是  id  为 5029 的 WeChatUser 标签的节点，end 是 id 为 5030 的 WeChatUser 标签的节点：

	{
	  "identity": 14455,
	  "start": 5029,
	  "end": 5030,
	  "type": "LIKES",
	  "properties": {
	  "rating": 100
	  }
	}	

5.查询连接在关系 LIKES 两边的节点：

	MATCH (a)-[r:LIKES]->(b) RETURN a,b

在结果中的没有内容的两个节点，就是我们在第 3 步中没有使用属性直接创建的两个节点。内容为 ppy 和 wys 的两个节点就是我们在第 4 步中使用属性创建的两个节点。

![](/Users/wys/Desktop/NLP/Neo4j-relation.png)

### WHERE 命令
类似于 SQL ，WHERE 命令来过滤 MATCH 查询的结果。

过滤出 name 为 wys 或者 dname 为 qjfy 的 Employee 标签的节点：
	
	MATCH (e:Employee) WHERE e.name = 'wys' OR e.dname = 'qjfy' RETURN e
	
### DELETE 命令

DELETE 命令主要的功能是：

* 删除节点及其属性
* 删除节点之间的关系

删除现有的所有 Employee 标签的节点（前提是这些节点上没有其他关系连接，否则会报错）：

	MATCH (e: Employee) DELETE e
	
删除现有的所有 LIKE 关系及其有关的节点：
	
	MATCH (a:WeChatUser)-[r]->(b:WeChatUser) DELETE a,r,b

### REMOVE 命令

REMOVE 命令主要的功能是：

* 移除节点或者关系的属性
* 移除节点或者关系的标签

1.现在创建一个带有属性的节点：

	CREATE (emp:Employee{name:"wys",dname:"qjfy"})
	
移除 dname 属性：

	MATCH (emp:Employee { name:"wys" }) REMOVE emp.dname RETURN emp

2.创建一个有多个标签的节点：

	CREATE (emp:Employee:Actor)
	
移除了节点的 Actor 标签：

	MATCH (e: Employee)  REMOVE e: Actor
	
### SET 命令

 SET 命令的主要功能是：
 
* 向现有节点或关系添加新属性
* 添加或更新属性值

现在创建一个的节点：

	CREATE (emp:Employee{name:"wys"})

开始添加一个属性 age ，并设置值为 27 ：

	MATCH (e: Employee) WHERE e.name="wys" SET e.age = '27' RETURN e
	
### ORDER BY 命令
类似于 SQL ，ORDER BY 命令对 MATCH 查询返回的结果进行排序。

再创建一个年龄为 28 的 Employee 节点：
	
	CREATE (emp:Employee{name:"wyt",age:'28'})

按照 age 将 Employee 节点排序，默认升序排序：

	MATCH (e:Employee) RETURN e ORDER BY e.age
	


### LIMIT 和 SKIP 命令

类似于 SQL， LIMIT 命令限制查询返回的行数，SKIP 命令过滤掉查询返回的前面若干行。

只返回一个 Employee 节点：

	MATCH (emp:Employee) RETURN emp LIMIT 1
	
跳过第一个结果，把第二个及之后的结果返回：

	MATCH (emp:Employee) RETURN emp SKIP 1
	

### IN 命令
类似于 SQL， IN 命令允许在 WHERE 子句中规定多个值。

只返回名为 wys 的节点：

	MATCH (e:Employee)  WHERE e.name IN ['wys'] RETURN e

### NULL 值

NULL 值表对节点或关系的属性的缺失值或未定义值。

返回 Employee 标签中 age 属性为空的节点：

	MATCH (e:Employee)  WHERE e.age IS NULL RETURN e

### MERGE 命令
MERGE 类似于 CREATE 命令和 MATCH 命令的组合，主要功能是：

* 创建节点、关系或者属性
* 从数据库检索数据

如果我们使用 CREATE 重复执行两次，因为 CREATE 命令总是向数据库添加新的节点，所以会创建两个一样的节点，但是将下面的语句重复执行两次，则只会创建一个节点：

	MERGE (e: Employee{ name:"wangda"})

所以 MERGE 命令检查该节点在数据库中是否存在。 如果它不存在创建新节点。 否则它不创建新的。


### UNION 命令
UNION 命令主要有两个：

* UNION，它将两组结果中的公共行组合并返回到一组结果中。 它不返回两个节点中重复的行。结果列名称和数据类型应该相同。

返回的结果中的 name 列没有重复行：

	MATCH (a:Employee) return a.name as name UNION MATCH (b:Actor) return b.name as name

* UNION ALL，它结合并返回两个结果集的所有行组成一个结果集。它返回两个节点中的重复行。同上，列名称和数据类型应该是相同的。

返回的结果中的 name 列有重复行：

	MATCH (a:Employee) return a.name as name UNION ALL MATCH (b:Actor) return b.name as name