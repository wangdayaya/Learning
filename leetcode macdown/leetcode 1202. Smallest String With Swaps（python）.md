leetcode  1202. Smallest String With Swaps（python）




### 描述

You are given a string s, and an array of pairs of indices in the string pairs where pairs[i] = [a, b] indicates 2 indices(0-indexed) of the string.

You can swap the characters at any pair of indices in the given pairs any number of times.

Return the lexicographically smallest string that s can be changed to after using the swaps.



Example 1:

	Input: s = "dcab", pairs = [[0,3],[1,2]]
	Output: "bacd"
	Explaination: 
	Swap s[0] and s[3], s = "bcad"
	Swap s[1] and s[2], s = "bacd"

	
Example 2:

	Input: s = "dcab", pairs = [[0,3],[1,2],[0,2]]
	Output: "abcd"
	Explaination: 
	Swap s[0] and s[3], s = "bcad"
	Swap s[0] and s[2], s = "acbd"
	Swap s[1] and s[2], s = "abcd"


Example 3:

	Input: s = "cba", pairs = [[0,1],[1,2]]
	Output: "abc"
	Explaination: 
	Swap s[0] and s[1], s = "bca"
	Swap s[1] and s[2], s = "bac"
	Swap s[0] and s[1], s = "abc"

	


Note:

	1 <= s.length <= 10^5
	0 <= pairs.length <= 10^5
	0 <= pairs[i][0], pairs[i][1] < s.length
	s only contains lower case English letters.


### 解析

根据题意，给定一个字符串 s ，以及索引对数组，其中 pairs[i] = [a, b] 表示字符串的 2 个索引（0 索引）。我们可以多次交换给定 pair 中任何一对索引处的字符。返回使用交换后可以更改为 s 的字典最小字符串。


其实这就是一个考察并查集的题目，也是一个图相关的题目，我们根据题意可以知道相连的字符之间可以任意换位置，我们可以将其转化为一个图，节点就是字符，边就是 pairs 给出的相连索引，所以我们通过并查集的方法，找到每个位置的索引的根索引，然后通过根索引找出相连的所有字符位置，将这些字符排序然后再挨个放入对应索引，这样就是实现在合理操作的基础上的字典序最小的字符串。

时间复杂度为 O(NlogN) ，空间复杂度为 O(N) 。


### 解答
				
	class Solution(object):
	    def smallestStringWithSwaps(self, s, pairs):
	        N = len(s)
	        s = list(s)
	        father = [0] * N
	        for i in range(N):
	            father[i] = i
	
	        def findFather(x):
	            if father[x] != x:
	                father[x] = findFather(father[x])
	            return father[x]
	        def union(x, y):
	            x = father[x]
	            y = father[y]
	            if x<y:
	                father[y] = x
	            else:
	                father[x] = y
	        for x,y in pairs:
	            if findFather(x) != findFather(y):
	                union(x,y)
	        map = collections.defaultdict(list)
	        for i in range(N):
	            map[findFather(i)].append(i)
	
	        for k,v in map.items():
	            tmp = []
	            for idx in v:
	                tmp.append(s[idx])
	            tmp.sort()
	            k = 0
	            for idx in v:
	                s[idx] = tmp[k]
	                k += 1
	        return ''.join(s)

            	      
			
### 运行结果


	Runtime: 747 ms, faster than 75.00% of Python online submissions for Smallest String With Swaps.
	Memory Usage: 53.2 MB, less than 82.35% of Python online submissions for Smallest String With Swaps.

### 原题链接

https://leetcode.com/problems/smallest-string-with-swaps/


您的支持是我最大的动力
