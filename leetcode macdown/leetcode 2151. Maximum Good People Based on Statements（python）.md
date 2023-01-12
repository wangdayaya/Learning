leetcode 2151. Maximum Good People Based on Statements （python）


### 前言



我比赛的时候只完成了前三题，最后一道题剩下一个多小时，自己在草稿上写写画画，一会觉得有头绪一会觉得一团乱麻，最后放弃治疗，我看榜单前面的大佬们解决这道通也就只用了不到三番钟，这就是我和大佬的差距。这是 Weekly Contest 277 的第四题，难度 Hard ，对于我来说是真 Hard ，对于大佬来说就是个弟弟。

### 描述


There are two types of persons:

* The good person: The person who always tells the truth.
* The bad person: The person who might tell the truth and might lie.

You are given a 0-indexed 2D integer array statements of size n x n that represents the statements made by n people about each other. More specifically, statements[i][j] could be one of the following:

* 0 which represents a statement made by person i that person j is a bad person.
* 1 which represents a statement made by person i that person j is a good person.
* 2 represents that no statement is made by person i about person j.

Additionally, no person ever makes a statement about themselves. Formally, we have that statements[i][i] = 2 for all 0 <= i < n.

Return the maximum number of people who can be good based on the statements made by the n people.




Example 1:


![](https://assets.leetcode.com/uploads/2022/01/15/logic1.jpg)

	Input: statements = [[2,1,2],[1,2,2],[2,0,2]]
	Output: 2
	Explanation: Each person makes a single statement.
	- Person 0 states that person 1 is good.
	- Person 1 states that person 0 is good.
	- Person 2 states that person 1 is bad.
	Let's take person 2 as the key.
	- Assuming that person 2 is a good person:
	    - Based on the statement made by person 2, person 1 is a bad person.
	    - Now we know for sure that person 1 is bad and person 2 is good.
	    - Based on the statement made by person 1, and since person 1 is bad, they could be:
	        - telling the truth. There will be a contradiction in this case and this assumption is invalid.
	        - lying. In this case, person 0 is also a bad person and lied in their statement.
	    - Following that person 2 is a good person, there will be only one good person in the group.
	- Assuming that person 2 is a bad person:
	    - Based on the statement made by person 2, and since person 2 is bad, they could be:
	        - telling the truth. Following this scenario, person 0 and 1 are both bad as explained before.
	            - Following that person 2 is bad but told the truth, there will be no good persons in the group.
	        - lying. In this case person 1 is a good person.
	            - Since person 1 is a good person, person 0 is also a good person.
	            - Following that person 2 is bad and lied, there will be two good persons in the group.
	We can see that at most 2 persons are good in the best case, so we return 2.
	Note that there is more than one way to arrive at this conclusion.
	




Note:

	n == statements.length == statements[i].length
	2 <= n <= 15
	statements[i][j] is either 0, 1, or 2.
	statements[i][i] == 2


### 解析

根据题意，现在有两种人：好人肯定说真话。坏人可能说真话和可能撒谎。给定一个 0 索引的 2D 整数数组 statements ，大小为 n x n，表示 n 个人对彼此的陈述。  statements[i][j] 可以是以下之一：

* 0 代表第 i 个人说第 j 个人是坏人
* 1 代表第 i  个人说第 j 个人是好人
* 2 表示第 i 个人没有关于第 j 个人的任何陈述

此外，没有人会发表关于自己的言论。 根据 n 个人的陈述，返回最多可能有几个好人。

看完这道题的陈述和例子一的文本长度我就知道，这不是一道好做的题（对我来说），当时做这道题的时候是按照图的方向去思考的，因为互相之间可能存在陈述，但是看了上面的例子一之后感觉好繁杂，如果出现了坏人，那么就会把很多有关联的陈述都推翻，觉得这个方向不靠谱。后来又想是不是有什么规律，但是找了半天也不靠谱。最后想到了递归，因为限制条件中 n 最大为 15 ，通过把不同的情况递归，可能找出最多的人，但是怎么设计递归是个问题。

看了某个大佬的解答才恍然大悟，真的是如醍醐灌顶，一语惊醒梦中人。题目中给出了 n 个，每个人的身份最多就是两种好人或者坏人，我们只需要通过 DFS 递归，参数为列表 persons ，表示当前递归分支可能出现的好人或者坏人组合。每一次将一个好人加入 persons ，或者将一个坏人加入 persons ，在不同的分支上，重复此递归操作，最后当新的列表的长度为 n 时，就停止递归。在这个递归函数中，我们不断更新最多可能的好人个数，这个功能我们用函数 hasContradiction 来定义，判断当前递归分支里的 persons 组合和 statements 是否有冲突，如果没有冲突说明这个分支上的好坏人组合是符合题意的，我们用当前好人的数量来和结果 result 进行比较取较大值更新 result 。

因为有 n 个人，每个人有两个不同的角色，所以有 2^n 种可能，每种可能我们都要进行 hasContradiction 函数中两层循环的判断，所以这个算法的时间复杂度为 O(2^N \* N^2) ，幸亏限制条件中 n 最大为 15 ，否则肯定超时了。空间复杂度为 O(N)，这个没啥好说的。

这里贴的是大佬的代码，真的是思路清晰，代码简介，佩服。
### 解答
				

	class Solution:
	    def maximumGood(self, statements: List[List[int]]) -> int:
	        BAD, GOOD, NO_STATEMENT = 0, 1, 2
	        N = len(statements)
	        
	        def hasContradiction(persons):
	            for i in range(N):
	                for j in range(N):
	                    if statements[i][j] == NO_STATEMENT:
	                        continue
	                    if persons[i] == GOOD:
	                        if statements[i][j] != persons[j]:
	                            return True
	            return False
	        
	        ans = 0
	        def dfs(persons):
	            nonlocal ans
	            if len(persons) == N:
	                if not hasContradiction(persons):
	                    ans = max(ans, persons.count(GOOD))
	            else:
	                dfs(persons+[GOOD])
	                dfs(persons+[BAD])
	                
	        dfs([])
	        return ans
            	      
			
### 运行结果

	
	91 / 91 test cases passed.
	Status: Accepted
	Runtime: 4252 ms
	Memory Usage: 14.3 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-277/problems/maximum-good-people-based-on-statements/


您的支持是我最大的动力
