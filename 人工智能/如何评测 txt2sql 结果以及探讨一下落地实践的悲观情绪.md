# 前言
最近一直在做 `txt2sql` 的相关工作，但是对于结果的评测是有很多坑要踩，而且最关键的是需要人工去核对结果，这是相当费时费力的操作，所以我们打算引入大模型让大模型来对 `benchmark` 的结果和 `txt2sql` 的结果进行比对，看结果是否符合预期，如果不符合预期给出理由。

# 具体实现 

我这里直接使用的是通义千问的 qwen-turbo 模型，具体的比对测试结果的 prompt 是我自己写的，大家可以参考。

```
def analyze_difference(question, sql_base, sql_aigc):
    messages = [
        {'role': 'system',
         'content': '您是一名精通于解析 SQL 的助手，基于我的问题，我会提供两个 SQL 分别是 sql_base 和 sql_aigc ，sql_base 是标准正确的 SQL ， sql_aigc 是生成的结果 SQL，以 sql_base 为准，如果两个 SQL 表达的含义不同，请解释两个 SQL 的不同之处。'},
        {'role': 'user',
         'content': f"您是一名精通于解析 SQL 的助手，基于我的问题，我会提供两个 SQL 分别是 sql_base 和 sql_aigc ，sql_base 是标准正确的 SQL ， sql_aigc 是生成的结果 SQL，以 sql_base 为准，如果两个 SQL 表达的含义不同，请解释两个 SQL 的不同之处。主要从 SQL 逻辑和字段使用是否正确合理来进行分析，对 SQL 中的别名忽略不做分析\n"

                    f"请按照下面的格式输出，如果两个 SQL 含义相同则输出内容为空字符串：【相同###经过分析从问题中可以看两个 SQL 的逻辑都符合问题的要求】，不做多余的解释。如果两个 SQL 含义不同则输出内容为：【不相同###（简单概括不同的原因）】\n"

                    f"例子：现在的问题是：统计萧山区管廊长度大于800m的施工单位个数，现在有 sql_base : select ssqx,count(distinct sgdw) as "施工单位个数" from gxsspy WHERE ssqx in ( '萧山区' ) and "length" >= 800 group by ssqx order by ssqx，sql_aigc： select ssqx,count(distinct sgdw) as "施工单位个数" from gxsspy WHERE ssqx in ( '萧山区' ) and "length" >= 800 group by ssqx order by ssqx，并分析是否相同，如果不同则给出原因\n"
                    f"按照规定的输出格式输出为：【相同###经过分析从问题中可以看两个 SQL 的逻辑都符合问题的要求】\n"

                    f"例子：现在的问题是：统计钱江华府5幢附近1000米范围内的地铁站个数，现在有 sql_base : select count(1) as "地铁站数量（个）"  from dtzpt where ST_DWithin(st_transform((select geom from aipoi where name ~ '钱江华府5幢' order by id limit 1),4549),st_transform(geom,4549),1000)，sql_aigc： select "ssqx",count(1) as "数量" from dtzpt WHERE ST_DWithin(ST_Transform(geom,4549),ST_Transform((SELECT geom FROM aipoi WHERE name like '%钱江华府5幢%' order by id limit 1),4549),1000) group by "ssqx" order by ssqx，并分析是否相同，如果不同则给出原因\n"
                    f"按敢规定的输出格式输出为：【不相同###sql_aigc 中多增加了一列 ssqx 并按其分组并排序，逻辑与 sql_base 不同】\n"

                    f"现在的问题是：{question}，现在有 sql_base :{sql_base}\n，sql_aigc：{sql_aigc}\n，并分析 sql_base 和 sql_aigc 是否相同，如果不同则给出原因，返回的内容按照格式输出，如果两个 SQL 含义相同则输出内容为空字符串：【相同】，不做多余的解释，如果两个 SQL 含义不同则输出内容为：【（简单概括不同的原因）】"}
    ]
    dashscope.api_key = ''
    try:
        response = Generation.call(model="qwen-turbo", messages=messages, result_format='message')
        resp = response.output.choices[0]['message']['content']
        if resp and resp.count("###") == 1:
            resp = resp.replace("【", "").replace("】", "").split("###")
            return response['usage']['total_tokens'], resp[0], resp[1]
        return response['usage']['total_tokens'],"不相同", "无法解析"
    except Exception:
        print(f"{question} 出现异常")
        return response['usage']['total_tokens'], "不相同", "异常"
```

# 结果展示

`question`:
```
查询西湖区地下综合管廊的信息
```
`sql_base`: 
```
select distinct "name" , "xmbh" , "qsdw" , "jsdw" , "sjdw" , "sgdw" , "kcdw" , "is_model" , "ztzgcdbbg" , "ztzscdbbg" , "glcssl" , "length" , "geom" from gxsspy WHERE ssqx in ( '西湖区' )
```

`sql_aigc`:
```
select distinct "zfzgbm" , "ssqx" , "name" , "xmbh" , "qsdw" , "jsdw" , "sjdw" , "sgdw" , "kcdw" , "is_model" , "ztzgcdbbg" , "ztzscdbbg" , "glcssl" , "length" , "geom" , "sjly" from gxsspy WHERE ssqx in ( '西湖区' ) order by "ssqx"
```

`大模型比对结果`：
```
sql_aigc 中添加了新的字段 "zfzgbm" 和额外的排序操作（order by "ssqx"），这与 sql_base 的查询结果不同，base SQL 仅包含了 "name" 到 "geom" 这些字段且没有排序
```

# 后续的坑

这个思路整体是出于节省时间和精力考虑的，出发点不错，思路在理论上也是正确的，但是在实际使用中还是会有很多的坑，比如：

1. `benchmark` 的 `sql` 和 `txt2sql` 的生成 `sql` 在逻辑上是一致的，但是`返回的字段数`不一样，其实结果都能满足用户的需求，但是在大模型评测的时候会认为是错的
2. `benchmark` 的 `sql` 和 `txt2sql` 的生成 `sql` 在逻辑上是一致的，但是`具体条件的语法`不一样，比如同样是模糊查询，一个使用 `~` ，一个使用 `%` ， 在大模型评测的时候会认为是错的
3. `benchmark` 的 `sql` 和 `txt2sql` 的生成 `sql` 在前面的写法上是一致的，但是一个最后加了 `order by` ，另一个没有加 `order by` ，结果的内容都一样，只是顺序变了，在大模型评测的时候会认为是错的
4. `benchmark` 的 `sql` 和 `txt2sql` 的生成 `sql` 在书写的时候`虽然使用的条件不一样`，但是逻辑都是对的，结果也是一样的，但是大模型评测的时候会认为是错的

等等，此类情况只有在实际踩坑才能发现，在实际使用过程中肯定还会出现各种各样的情况，这些都是要不断去迭代升级原始的 `prompt` ，才能尽量减少后续的类似的问题，提升大模型比对的质量。

# 落地讨论

其实在和一些交流群里和大佬在聊使用大模型比对 txt2sql 的测试结果的方案的时候，接触到了一些关于 txt2sql 实际落地的不同的声音，和大家分享交流一下：

1. `最简单最直接的方案`就是使用 txt2sql 的生成的 sql 去跑结果，看是否和 benchmark 的`结果是否完全一致`，这需要要求 benchmark 的质量相当高，这也是很多大厂直接评测的方案。
2. 普遍对 txt2sql 的`实际应用持悲观的态度`，因为面对测试集可能会过拟合，然后在实际的使用过程中效果很差，毕竟人的提问方式可是千奇百怪的，提问边界也是海阔天空不好控制，见过兄弟公司吹的很凶，找来了大厂技术背书，但是开会时候说实测试结果的时候基本都在说不太行。就怕投入人力去搞，最后是一地鸡毛。
3. txt2sql 涉及到一个实际`应用过程中的悖论`，txt2sql 的目标一般来说是根据问题生成 sql 去查数据库吧，要不要人工审核？如果要，生成的意义又是什么呢?如果不审核，生成的 sql 质量有没有保证？
4. txt2sql 如果效果不好，`生成的 sql 会不会删库、改数、低效SQL导致中台宕机`？运维人员每天捅你刀子都是轻的，关键是有安全隐患和安全责任。
