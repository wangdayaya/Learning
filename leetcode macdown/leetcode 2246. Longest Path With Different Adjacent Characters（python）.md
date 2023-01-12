leetcode  2246. Longest Path With Different Adjacent Characters（python）


这道题是第 289 场 leetcode 周赛的第四题，难度为 hard ，主要考察的是就是对字符串和列表的基本操作。

### 描述

You are given a tree (i.e. a connected, undirected graph that has no cycles) rooted at node 0 consisting of n nodes numbered from 0 to n - 1. The tree is represented by a 0-indexed array parent of size n, where parent[i] is the parent of node i. Since node 0 is the root, parent[0] == -1.

You are also given a string s of length n, where s[i] is the character assigned to node i.

Return the length of the longest path in the tree such that no pair of adjacent nodes on the path have the same character assigned to them.



Example 1:


![](https://assets.leetcode.com/uploads/2022/03/25/testingdrawio.png)

	Input: parent = [-1,0,0,1,1,2], s = "abacbe"
	Output: 3
	Explanation: The longest path where each two adjacent nodes have different characters in the tree is the path: 0 -> 1 -> 3. The length of this path is 3, so 3 is returned.
	It can be proven that there is no longer path that satisfies the conditions. 
	
Example 2:

![](https://assets.leetcode.com/uploads/2022/03/25/graph2drawio.png)

	Input: parent = [-1,0,0,0], s = "aabc"
	Output: 3
	Explanation: The longest path where each two adjacent nodes have different characters is the path: 2 -> 0 -> 3. The length of this path is 3, so 3 is returned.





Note:

	n == parent.length == s.length
	1 <= n <= 10^5
	0 <= parent[i] <= n - 1 for all i >= 1
	parent[0] == -1
	parent represents a valid tree.
	s consists of only lowercase English letters.


### 解析

根据题意，给定一棵树（即无环的连通无向图），其根节点为节点 0 ，由 n 个节点组成，编号从 0 到 n - 1 。该树由大小为 n 的 0 索引数组 parent 表示，其中 parent[i] 是节点 i 的父节点。 由于节点 0 是根节点，因此 parent[0] == -1 。还给出一个长度为 n 的字符串 s，其中 s[i] 是分配给节点 i 的字符。需要注意的是这里不仅仅是二叉树，可能是 n 叉树。 返回树中最长路径的长度，且路径上不会出现相邻节点具有相同的字符。

很明显这道题考察的就是 DFS ，所以整体思路就是沿着这个方向进行，要想路径长度最长，那么对于某个节点 node 来说，最长的路径就是某子树中的最长的合法路径（如果有的话）加上另一个某子树中的最长的合法路径（如果有的话）再加上一（该节点自身长度为 1 ）。

所以我们定义一个递归函数 dfs(i) ，表示从节点 i 开始向下的最长路径长度。那么从节点 i 开始向下的两条最长路径肯定来自于某两个子树，这样我们就能通过计算局部的最长路径更新全局变量 result 。

### 解答
				

	class Solution(object):
	    def __init__(self):
	        self.result = 0
	
	    def longestPath(self, parent, s):
	        children = [[] for _ in range(len(s))]
	        for i, j in enumerate(parent):
	            if j >= 0:
	                children[j].append(i)
	
	        def dfs(i):
	            tmp = [0]
	            for j in children[i]:
	                cur = dfs(j)
	                if s[i] != s[j]:
	                    tmp.append(cur)
	            tmp = heapq.nlargest(2, tmp)
	            self.result = max(self.result, sum(tmp) + 1)
	            return max(tmp) + 1
	        dfs(0)
	        return self.result
            	      
			
### 运行结果



	141 / 141 test cases passed.
	Status: Accepted
	Runtime: 3899 ms
	Memory Usage: 156 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-289/problems/longest-path-with-different-adjacent-characters/

您的支持是我最大的动力
