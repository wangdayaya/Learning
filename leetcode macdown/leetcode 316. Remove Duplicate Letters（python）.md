leetcode  316. Remove Duplicate Letters（python）




### 描述

Given a string s, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.

* This question is the same as 1081:https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/

Example 1:


	Input: s = "bcabc"
	Output: "abc"
	


Note:


	1 <= s.length <= 10^4
	s consists of lowercase English letters.

### 解析

根据题意，给定一个字符串 s，删除字符串中重复的字母，使每个字母出现一次且只出现一次。 您必须确保您的结果是所有可能结果中按字典顺序排列的最小的，并且字符之间的相对位置保持不变。

这道题和 1081 其实是一样的，做完这道题可以直接跳过去用另外一道题巩固练习：[https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/]()


这周的题目都是考察栈的知识和实际的应用，这道题有一点不同之处，如果是刷题经验比较丰富的同学应该读题之后就会有思路，像这种字母按照前后顺序进行字典顺序的情况，最直接的方法就是使用单调栈。

* 首先我们要使用字典 d 来对 s 中出现的字符进行计数
* 定义空栈 stack ，遍历 s 中的每个字符 c ，每次遍历执行如下操作
* 如果 c 已经出现在 stack ，为了满足题中每个字符只能出现一次的情况下我们只对 d[c] 减一，然后直接进行后面字符的遍历即可。如果 c 没有出现在 stack ，那么它是一个新的字符，我们需要看栈顶和 c 的关系，是否需要对栈顶不符合题意的字符进行弹出，不符合题意的栈顶字符就是 stack[-1] 大于 c （不满足字典序），并且 d[stack[-1]] 大于 0 （后面还有有可用的相同字符）
* 然后将当前的 c 加入栈顶，并且 d[c] 减一
* 所有字符都经过上述的操作，最后将 stack 拼接成字符串返回即可

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				

	class Solution(object):
	    def removeDuplicateLetters(self, s):
	        stack = []
	        d = collections.defaultdict(int)
	        for c in s:
	            d[c] += 1
	        for c in s:
	            if c in stack:
	                d[c] -= 1
	                continue
	            while stack and ord(stack[-1]) > ord(c) and d[stack[-1]]>0:
	                stack.pop()
	            d[c] -= 1
	            stack.append(c)
	        return "".join(stack)
	        
            	      
			
### 运行结果

	Runtime: 45 ms, faster than 34.58% of Python online submissions for Remove Duplicate Letters.
	Memory Usage: 13.6 MB, less than 72.50% of Python online submissions for Remove Duplicate Letters.


### 原题链接


https://leetcode.com/problems/remove-duplicate-letters/

您的支持是我最大的动力
