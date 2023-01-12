leetcode  402. Remove K Digits（python）

### 每日经典

《明日歌》 ——钱福（明）

明日复明日，明日何其多。

我生待明日，万事成蹉跎。

世人若被明日累，春去秋来老将至。

朝看水东流，暮看日西坠。

百年明日能几何？请君听我明日歌。

### 描述

Given string num representing a non-negative integer num, and an integer k, return the smallest possible integer after removing k digits from num.



Example 1:


	Input: num = "1432219", k = 3
	Output: "1219"
	Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
	
Example 2:

	Input: num = "10200", k = 1
	Output: "200"
	Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.


Example 3:

	Input: num = "10", k = 2
	Output: "0"
	Explanation: Remove all the digits from the number and it is left with nothing which is 0.

	


Note:

	1 <= k <= num.length <= 10^5
	num consists of only digits.
	num does not have any leading zeros except for the zero itself.


### 解析


根据题意，给定表示非负整数 num 的字符串 num 和整数 k，从 num 中删除 k 位后返回可能的最小整数。题意很简单，用到的思路也是单调栈解法，因为删除之后的数字想要保证最小整数，所以维护一个单调递增的栈 stack ，尽量将前面较大的数字删除。如果只删除了小于 k 的数字，说明 num 此时是一个递增的字符串，我们只需要将最后的几个数字删除即可。需要注意的是前置 0 要去掉，而且如果数字都被删除，结果应该是 "0" 。

### 解答
					
	class Solution(object):
	    def removeKdigits(self, num, k):
	        """
	        :type num: str
	        :type k: int
	        :rtype: str
	        """
	        stack = []
	        for c in num:
	            while stack and k>0 and stack[-1]>c:
	                stack.pop()
	                k-=1
	            stack.append(c)
	        if k>0:
	            stack = stack[:-k]
	        result = "".join(stack).lstrip("0")
	        if not result:
	            return "0"
	        return result
	        

            	      
			
### 运行结果


	Runtime: 28 ms, faster than 84.06% of Python online submissions for Remove K Digits.
	Memory Usage: 14.1 MB, less than 14.74% of Python online submissions for Remove K Digits.

原题链接：https://leetcode.com/problems/remove-k-digits/



您的支持是我最大的动力
