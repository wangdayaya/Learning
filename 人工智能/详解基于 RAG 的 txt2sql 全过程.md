

# 前文
本文使用通义千问大模型和 ChromaDB 向量数据库来实现一个完整的 text2sql 的项目，并基于实际的业务进行效果的展示。

# 准备

在进行项目之前需要准备下面主要的内容：

- `python 环境`
- `通义千问 qwen-max 模型的 api-key`
- `ChromaDB 向量数据库`
- `acge_text_embedding 嵌入模型`

# RAG 

首先在进行主要内容之前要先回顾一下基础知识，市面上 的 `text2sql` 项目的基本框架就是下图中展示的 `RAG` 框架图，也就是常说的`检索增强生成技术`。结合我们的 `text2sql` 业务数据，我们按照图中的 1 + 3 个步骤分别介绍。“1” 指的是要进行 RAG 的预先准备工作，“3” 是 RAG 的三个步骤。

1. 使用我们准备好的 `acge_text_embedding 嵌入模型` 将相关的`数据库表结构信息`、`字段使用方法`、供大模型参考的`question-sql 对`等信息都进行向量化，然后将向量存入`ChromaDB 向量数据库`。
2. 用户提出针对数据库的问题 `query` ，然后通过同样的  `acge_text_embedding 嵌入模型`  将 `query` 转化成向量，通过相关性计算算法，从`ChromaDB 向量数据库`中召回和 query 最相关的文本作为上下文 `context` ，这里的 `context` 理想状态下肯定是和问题相关的`表结构、字段信息`，或者相似的 `question-sql 对` ，这些信息会在后面输入进 `LLM` 中，供 `LLM` 理解。
3. 将用户的 `query` 和 `context` 拼接成一个完整的 `prompt` ，此时的  `prompt` 中既有供 LLM 参考的问题相关的可用信息，又有用户的问题 。
4. 将 `prompt` 给 `LLM` ，让其输出合理的结果，我们这里的结果其实就是预先想要得到的 `sql` 。



所以到现在我们应该能体会出来，RAG 的框架最核心的只有两个部分：
1. 第一就是能从向量数据库中召回最相关的上下文供 LLM 理解问题相关的上下文：
2. 第二就是大模型的理解能力，是否能在给出充足上下的情况下将问题解决。

![image.png](https://p1-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/b2f14ac63e514d5397370df44d0e1675~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=900&h=506&s=123488&e=png&b=f7f6f2)
# RAG 疑问

有的人可能会说为什么不跳过第一步，把数据库所有的信息都输入给大模型，理论上也是可以的。但是具体实施会有困难，原因如下：
1. 目前大模型输入 token 都有明确的限制，比如 `qwen-max` 模型只有 `8K` （尽管这些限制在逐渐消失，现在很多大模型的输入 token 都已经过百万 token 了）。
2. 另外就是考虑到成本，发送大量 token 是`非常昂贵`的操作，如果模型理解能力有限，更是毫无意义。
3. 最后就是从实际的研究，仅发送`少量的但是质量较高的相关信息`给大模型更有助于生成好的答案。


# 详细过程

## 数据准备

ddl.txt：这里面存放的都是业务范围内容的表结构。如下：

```
CREATE TABLE ai_prj_plan ( duty_party character varying(255) , pipeline_type character varying(255) , ... );
CREATE TABLE dtqjln (  xmbh character varying(100), jgsj integer, ...}
```

documentations.txt : 这里存放的是每个字段的详细说明或者注意事项。如下：

```
ai_prj_plan 表中的字段 id 表示工程计划的主键 id 。
ai_prj_plan 表中的字段 create_time 表示工程计划的创建时间。
...
dtqjln 表中的字段 jsdw 表示地铁线路或者地铁区间的建设单位名称。
dtqjln 表中的字段 sjdw 表示地铁线路或者地铁区间的设计单位名称。
```

question-sql.txt : 这里存放的是一些代表性的业务可能涉及到的问题-sql 对样本，如下：

```
已经投运的管线工程计划总长###select SUM(length::numeric) from ai_prj_plan where current_progress=5 and plan_type in (1,2,3)
查10条计划单独施工的工程名字###select project_name as "ai_prj_plan.project_name"  from ai_prj_plan where plan_type=1 limit 10
...
```

## 导入向量数据库

这里的三个文件，每一行都作为一个 doc ，然后将每一行使用预先准备的 `acge_text_embedding 嵌入模型` 转化成 1024 向量，也就是三个文件一共有多少行，就会有多少个 1024 的向量，然后都存入`ChromaDB 向量数据库`。

## 用户提问


用户提问“2023年入廊管线中前期项目的计划有多少”，会使用预先准备的 `acge_text_embedding 嵌入模型`，将问题转化为一个 1024 向量，将其与`ChromaDB 向量数据库` 中的所有 1024 向量进行相似性召回，分别从三个文件中找出最相关的内容，至于召回策略可以自己定义。根据我的自定义召回策略，然后将召回的内容和问题进行拼接组成下面的完整的 prompt ，从完整的 prompt 我们可以看到召回了将要使用的表结构 `ai_prj_plan` 以及相关字段 `plan_type 、annual_aim_json 、plan_category` 的使用说明，最后找出了两个可能对模型有用的 `question-sql 对`供模型参考。所以下面的内容是提供了足够完成用户提问的相关信息，最终模型也给我们生成了符合要求的 SQL ，说明我们的整体项目实现了既定的目标。
```
[
	{'role': 'system', 'content': '您是一名精通 SQL 的专家，用户会提出业务相关的问题，请根据相关信息回答合适的 SQL ，您将仅使用 SQL 代码进行回答，不进行任何解释。
        您可以使用以下展示出的表结构作为参考：\n\nCREATE TABLE ai_prj_plan\n(\n    id character varying(64)  NOT NULL,\n    create_time timestamp(6) without time zone,\n    update_time timestamp(6) without time zone,\n    remark character varying(255) ,\n    plan_type integer,\n    duty_party character varying(255) ,\n    pipeline_type character varying(255) ,\n    project_name character varying(255) ,\n    dlmc character varying(255) ,\n    start_end_point character varying(255) ,\n    ssqx character varying(100) ,\n    total_invest real,\n    length real,\n    plan_code character varying(255) ,\n    plan_category integer,\n    version integer,\n    accept integer,\n    verify_status integer,\n    refuse_reason character varying(255) ,\n    geom geometry(Geometry,4326), -- 几何使用 4326 坐标系\n    years character varying(255) ,\n    current_progress integer,\n    annual_aim_json text ,\n)\n\n
        您可以使用以下展示出的 documentation 作为参考，每个 documentation 解释了每个表的字段的名字和用法，使用他们以指导您有效准确地回答用户的问题，请务必遵循每个字段的使用方法和注意事项:\n\nai_prj_plan 表中的字段 plan_type 表示工程计划中涵盖的工程类型，我们规定只能枚举整数 1 、 2 、 3 、 4 、 5 ，整数 1 表示单独施工管线计划，整数 2 表示随道路施工管线工程计划，整数 3 表示入廊管线工程计划，整数 4 表示管廊工程计划，整数 5 表示互联互通工程计划，其中将整数 1 、 2 、 3 代表的三种工程计划合并起来统称为“管线工程计划”或者"管线计划"。\n\nai_prj_plan 表中的字段 plan_category 表示工程计划的计划分类，我们规定只能枚举整数 1 和 2 ，整数 1 表示工程计划在计划内，整数 2 表示工程计划在计划外。\n\nai_prj_plan 表中的字段 annual_aim_json 表示工程计划的每年详细计划列表，虽然该字段是字符串内容，但是存储格式是 json 列表。每个 json 中有三个字段 year、planProgress、 planInvest，表示打算计划在某年（year）给该项目一定的投资（planInvest），要将该项目推进到计划的进度（planProgress）。}, 
	{'role': 'user', 'content': '2024年入廊管线中前期项目的计划有多少'}, 
	{'role': 'assistant', 'content': "SELECT COUNT(*) FROM ai_prj_plan, jsonb_array_elements(annual_aim_json::jsonb) AS aim  WHERE (aim->>'planProgress')::integer = 1 AND plan_category = 1 AND plan_type = 3 AND (aim->>'year')::integer = 2024;"}, 
	{'role': 'user', 'content': '2024年入廊管线中已完成的前期项目有多少'}, 
	{'role': 'assistant', 'content': "SELECT COUNT(1) AS cnt FROM ai_prj_plan, jsonb_array_elements(annual_aim_json::jsonb) AS aim WHERE (aim->>'year')::int = 2024  and (aim->>'planProgress')::int <= current_progress and (aim->>'planProgress') is not null  and plan_type  = 3 and current_progress = 1    and plan_category = 1"}, 
 	{'role': 'user', 'content': '2023年入廊管线中前期项目的计划有多少'}
 ]
```

大模型结果输出：
```
Sql:SELECT COUNT(*) FROM ai_prj_plan, jsonb_array_elements(annual_aim_json::jsonb) AS aim  WHERE (aim->>'planProgress')::integer = 1 AND plan_category = 1 AND plan_type = 3 AND (aim->>'year')::integer = 2023;
```

完结撒花，希望上面的内容能给大家解释清楚相关的技术原理和细节。