leetcode  1209. Remove All Adjacent Duplicates in String II（python）




### 描述

You are given a string s and an integer k, a k duplicate removal consists of choosing k adjacent and equal letters from s and removing them, causing the left and the right side of the deleted substring to concatenate together.

We repeatedly make k duplicate removals on s until we no longer can.

Return the final string after all such duplicate removals have been made. It is guaranteed that the answer is unique.



Example 1:

	Input: s = "abcd", k = 2
	Output: "abcd"
	Explanation: There's nothing to delete.

	
Example 2:

	Input: s = "deeedbbcccbdaa", k = 3
	Output: "aa"
	Explanation: 
	First delete "eee" and "ccc", get "ddbbbdaa"
	Then delete "bbb", get "dddaa"
	Finally delete "ddd", get "aa"


Example 3:


	Input: s = "pbbcggttciiippooaais", k = 2
	Output: "ps"
	



Note:

	1 <= s.length <= 10^5
	2 <= k <= 10^4
	s only contains lower case English letters.


### 解析

根据题意，给定一个字符串 s 和一个整数 k ，当 s 中有 k 个重复相邻且一样的字母，可以将它们删除，然后将删除后的左子字符串和右子字符串连接在一起。我们反复对 s 进行上述的删除操作，直到我们不再可以操作为止。返回最终字符串。 题目已经保证答案是唯一的。

这道题很明显就是考察栈的相关的知识点，以为我们知道题意的删除操作有点类似出栈的操作，加入我们将字符进行入栈，当相邻且相同字符数量达到 k 的时候，将这 k 个相同的字符都要出栈，然后再对后面的字符进行重复的入栈和出栈操作。

此时我们已经定下了大体的解题方向，就是用栈的思路，结合题意，我们可以将当前的字符和计数器作为一个元组一块进行入栈，如果有相邻且相同的字符，只需要把栈顶的元组的计数器加一即可，如果计数器达到了 k 那么，我们就将整个元组都弹出，然后对后面的元素进行相同的操作，最后我们得了一个结果的栈，只需要将其恢复成字符串然后返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				
	class Solution(object):
	    def removeDuplicates(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: str
	        """
	        stack = []
	        for i,c in enumerate(s):
	            if not stack:
	                stack.append([c,1])
	            elif c == stack[-1][0]:
	                stack[-1][1] += 1
	                if stack[-1][1] == k:
	                    stack.pop()
	            elif c != stack[-1][0]:
	                stack.append([c,1])
	        result = ''
	        for c,n in stack:
	            result += c*n
	        return result

            	      
			
### 运行结果

	Runtime: 275 ms, faster than 20.29% of Python online submissions for Remove All Adjacent Duplicates in String II.
	Memory Usage: 19.3 MB, less than 62.57% of Python online submissions for Remove All Adjacent Duplicates in String II.


### 原题链接



https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string-ii/

您的支持是我最大的动力
