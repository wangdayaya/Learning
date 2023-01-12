leetcode  990. Satisfiability of Equality Equations（python）




### 描述


You are given an array of strings equations that represent relationships between variables where each string equations[i] is of length 4 and takes one of two different forms: "x<sub>i</sub>==y<sub>i</sub>" or "x<sub>i</sub>!=y<sub>i</sub>".Here, x<sub>i</sub> and y<sub>i</sub> are lowercase letters (not necessarily different) that represent one-letter variable names. Return true if it is possible to assign integers to variable names so as to satisfy all the given equations, or false otherwise.


Example 1:


	Input: equations = ["a==b","b!=a"]
	Output: false
	Explanation: If we assign say, a = 1 and b = 1, then the first equation is satisfied, but not the second.
	There is no way to assign the variables to satisfy both equations.
	
Example 2:

	Input: equations = ["b==a","a==b"]
	Output: true
	Explanation: We could assign a = 1 and b = 1 to satisfy both equations.

 


Note:

	1 <= equations.length <= 500
	equations[i].length == 4
	equations[i][0] is a lowercase letter.
	equations[i][1] is either '=' or '!'.
	equations[i][2] is '='.
	equations[i][3] is a lowercase letter.


### 解析

根据题意，给定一个字符串数组 equations ，equations 表示变量之间的关系，其中每个字符串 equations[i] 的长度为 4，并采用以下两种不同形式之一： "x<sub>i</sub>==y<sub>i</sub>" 或 "x<sub>i</sub>!=y<sub>i</sub>"。这里 x<sub>i</sub> 和 y<sub>i</sub> 代表一个小写字母的变量名。 如果这些变量都满足 equations 表示的关系，则返回 true ，否则返回 false 。

其实这道题读完我们就能了解到，这是在考查并查集，因为在使用 == 来将两个不同的字母进行相连之后，由于 == 的传递性质，这些字母也都是互相相等的，用并查集的角度来进行解释，就是这些字母都能放到一个集合里，他们的公共最远祖先可以用同一个字母进行表示。

有了上面的基本思想，我们可以先遍历一次所有 equations ，当两个字母相等的时候，我们去分别将两个字母放入自己所属的集合当中，然后再遍历一次所有 equations ，当两个字母不相等的时候，如果他们能放入同一个集合说明会产生矛盾，直接返回 False 即可，否则正常遍历结束之后，直接返回 True 。

在实现的时候只需要使用一个列表 father 来表示每个字母对应的集合的最远公共祖先，开始的时候每个字母的祖先都是自己，然后在合并的时候，不断更新自己的最远祖先，在查找的时候只需要沿着父节点路线向上找到最远祖先即可。


时间复杂度为 O(N+AlogA) ，空间复杂度为 O(A) ，N 是 equations 的长度，A 是字母的总数。


### 解答
	
	class Solution:
	    class UnionFind:
	        def __init__(self):
	            self.father = list(range(26))
	        def find(self, index):
	            if index == self.father[index]:
	                return index
	            self.father[index] = self.find(self.father[index])
	            return self.father[index]
	        def union(self, index1, index2):
	            self.father[self.find(index1)] = self.find(index2)
	
	    def equationsPossible(self, equations: List[str]) -> bool:
	        unionFind = Solution.UnionFind()
	        for e in equations:
	            if e[1] == "=":
	                a = ord(e[0]) - ord("a")
	                b = ord(e[3]) - ord("a")
	                unionFind.union(a, b)
	        for e in equations:
	            if e[1] == "!":
	                a = ord(e[0]) - ord("a")
	                b = ord(e[3]) - ord("a")
	                if unionFind.find(a) == unionFind.find(b):
	                    return False
	        return True
	 
	 

### 运行结果
	
	Runtime: 47 ms, faster than 94.56% of Python3 online submissions for Satisfiability of Equality Equations.
	Memory Usage: 14.2 MB, less than 34.67% of Python3 online submissions for Satisfiability of Equality Equations.

### 原题链接

https://leetcode.com/problems/satisfiability-of-equality-equations/


您的支持是我最大的动力
