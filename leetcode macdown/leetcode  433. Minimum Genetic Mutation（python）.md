leetcode 433. Minimum Genetic Mutation （python）




### 描述


A gene string can be represented by an 8-character long string, with choices from 'A', 'C', 'G', and 'T'. Suppose we need to investigate a mutation from a gene string start to a gene string end where one mutation is defined as one single character changed in the gene string.

* For example, "AACCGGTT" --> "AACCGGTA" is one mutation.

There is also a gene bank bank that records all the valid gene mutations. A gene must be in bank to make it a valid gene string. Given the two gene strings start and end and the gene bank bank, return the minimum number of mutations needed to mutate from start to end. If there is no such a mutation, return -1. Note that the starting point is assumed to be valid, so it might not be included in the bank.




Example 1:

	Input: start = "AACCGGTT", end = "AACCGGTA", bank = ["AACCGGTA"]
	Output: 1

	
Example 2:

	Input: start = "AACCGGTT", end = "AAACGGTA", bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]
	Output: 2


Example 3:

	Input: start = "AAAAACCC", end = "AACCCCCC", bank = ["AAAACCCC","AAACCCCC","AACCCCCC"]
	Output: 3



Note:

	start.length == 8
	end.length == 8
	0 <= bank.length <= 10
	bank[i].length == 8
	start, end, and bank[i] consist of only the characters ['A', 'C', 'G', 'T'].


### 解析

根据题意，基因串可以用一个 8 个字符的字符串表示，有“A”、“C”、“G”和“T”可供选择。假设我们需要研究从基因串 start 到 end 中的突变，其中一个突变被定义为基因链中改变的一个字符。

* *例如，“AACCGGTT”-->“AACCGGTA”是一个突变。

还有一个基因库 bank ，记录了所有有效的基因突变。基因必须在 bank 中才能使其成为有效的基因串。给定两个基因串的开始 start 和结束 end 以及基因库 bank ，返回从头到尾突变所需的最小突变数。如果没有这样的突变，则返回 -1 。其中 start 默认有效，但是不会出现在 bank 中。

通过题目中的限制条件我们可以发现，start 和 end 的长度为 8 ，bank 的长度最大为 10 ，所以我们可以使用暴力的办法进行解题。题目让我们找需要变化的最小次数，因为每个字母都可能发生突变，且突变的可能有四种 A、C、G、T ，所以我们可以使用 BFS 来进行罗列所有有效的可能的合法的 end ，并和 start 进行比对。

* 定义一个队列 Q ，里面的每一项内容是（当前变异的字符串，已经变化的次数），并定义一个 visited 存放已经出现过的字符串
* 循环弹出 Q 最前面的一项内容（s，steps），如果被弹出的字符串 s 和 end 相等就直接返回记录好的已经变化的次数 steps ；否则将 s 中每个字符都使用四种变异字符进行替换，将所有还没见过的字符串且在 bank 中的合法字符串都存放于 Q 和 visited 中，不断进行上述过程
* 如果 Q 循环结束，说明没有结果直接返回 -1 。

时间复杂度为 O(4 \* 8 \* M) ，空间复杂度为 O(8 \* M) 。8 是每个基因的长度，4 是可用的突变字符数， M 是 bank 的长度。


### 解答

	class Solution(object):
	    def minMutation(self, start, end, bank):
	        """
	        :type start: str
	        :type end: str
	        :type bank: List[str]
	        :rtype: int
	        """
	        Q = collections.deque([(start, 0)])
	        visited = {start}
	        while Q:
	            s, steps = Q.popleft()
	            if s == end:
	                return steps
	            for c in "ACGT":
	                for i in range(len(s)):
	                    tmp = s[:i] + c + s[i + 1:]
	                    if tmp not in visited and tmp in bank:
	                        Q.append((tmp, steps + 1))
	                        visited.add(tmp)
	        return -1

### 运行结果

	Runtime: 15 ms, faster than 95.56% of Python online submissions for Minimum Genetic Mutation.
	Memory Usage: 13.5 MB, less than 44.44% of Python online submissions for Minimum Genetic Mutation.


### 原题链接

https://leetcode.com/problems/minimum-genetic-mutation/


您的支持是我最大的动力
