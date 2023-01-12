leetcode 2055. Plates Between Candles （python）

### 描述

There is a long table with a line of plates and candles arranged on top of it. You are given a 0-indexed string s consisting of characters '*' and '|' only, where a '*' represents a plate and a '|' represents a candle.

You are also given a 0-indexed 2D integer array queries where queries[i] = [left<sub>i</sub>, right<sub>i</sub>] denotes the substring s\[left<sub>i</sub>...right<sub>i</sub>] (inclusive). For each query, you need to find the number of plates between candles that are in the substring. A plate is considered between candles if there is at least one candle to its left and at least one candle to its right in the substring.

For example, s = "||**||**|\*", and a query [3, 8] denotes the substring "\*||**|". The number of plates between candles in this substring is 2, as each of the two plates has at least one candle in the substring to its left and right.
Return an integer array answer where answer[i] is the answer to the i<sub>th</sub> query.



Example 1:

![](https://assets.leetcode.com/uploads/2021/10/04/ex-1.png)

	Input: s = "**|**|***|", queries = [[2,5],[5,9]]
	Output: [2,3]
	Explanation:
	- queries[0] has two plates between candles.
	- queries[1] has three plates between candles.

	
Example 2:

![](https://assets.leetcode.com/uploads/2021/10/04/ex-2.png)

	Input: s = "***|**|*****|**||**|*", queries = [[1,17],[4,5],[14,17],[5,11],[15,16]]
	Output: [9,0,0,0,0]
	Explanation:
	- queries[0] has nine plates between candles.
	- The other queries have zero plates between candles.



Note:

	
* 3 <= s.length <= 10^5
* s consists of '*' and '|' characters.
* 1 <= queries.length <= 10^5
* queries[i].length == 2
* 0 <= left<sub>i</sub> <= right<sub>i</sub> < s.length

### 解析

根据题意，有一张长桌上面放着一排盘子和蜡烛。给定一个索引为 0 的字符串 s ，由字符 '\*' 和 '|' 组成仅，其中 '*' 代表一个盘子，一个 '|'代表蜡烛。给出一个 0 索引的二维整数数组 queries ，其中  queries[i] = [left<sub>i</sub>, right<sub>i</sub>] 表示子字符串 s[left<sub>i</sub>...right<sub>i</sub>]（含）。对于每个 query ，需要找到子字符串中蜡烛之间的盘子数。如果在子字符串中，该盘子左侧至少有一根蜡烛，右侧至少有一根蜡烛，则认为该盘子位于蜡烛之间。

例如，s = "||\*\*||\*\*|\*"，查询[3, 8] 表示子串 "\*||\*\*|" 。此子串中蜡烛之间的盘子数为 2，因为两个盘子中的每一个在其左右的子串中至少有一根蜡烛。
返回一个整数数组 answer ，其中 answer[i] 是第 i<sub>th</sub> 个查询的答案。

读完题目，我们发现最后在计算子字符串中的有效盘子数量，只需要知道一个盘子的左边有蜡烛，并且右边有蜡烛就可以判断该盘子是有效的了，那么我们只需要提前准备好两个列表left 和 right ，left 表示 s[i] 左边最近的蜡烛位置， right 表示  s[i]  右边最近的蜡烛位置。然后使用前缀和列表 presum 提前计算出  s[i]  蜡烛之前的所有蜡烛总和，然后遍历 queries 进行计算有效蜡烛的个数即可。

### 解答
				
	
	class Solution(object):
	    def platesBetweenCandles(self, s, queries):
	        """
	        :type s: str
	        :type queries: List[List[int]]
	        :rtype: List[int]
	        """
	        N = len(s)
	        presum = [0]*N
	        left = [-1]*N
	        right = [-1]*N
	        
	        L = -1
	        for i in range(N):
	            if s[i] == '|':
	                L = i
	            left[i] = L
	            
	        R = -1
	        for i in range(N-1, -1, -1):
	            if s[i] == '|':
	                R = i
	            right[i] = R
	            
	        t = 0    
	        for i in range(N):
	            if s[i] == '*':
	                t += 1
	            presum[i] = t
	        
	        result = []
	        for a,b in queries:
	            x = right[a]
	            y = left[b]
	            if x<=y and x>=a and y<=b:
	                result.append(presum[y]-presum[x])
	            else:
	                result.append(0)
	                
	        return result
	                
	         
	                
            	      
			
### 运行结果
	
	Runtime: 1800 ms, faster than 81.51% of Python online submissions for Plates Between Candles.
	Memory Usage: 57.3 MB, less than 54.62% of Python online submissions for Plates Between Candles.


原题链接：https://leetcode.com/problems/plates-between-candles/



您的支持是我最大的动力
