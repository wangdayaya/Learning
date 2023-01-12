leetcode  32. Longest Valid Parentheses（python）


### 描述

Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.



Example 1:


	Input: s = "(()"
	Output: 2
	Explanation: The longest valid parentheses substring is "()".
	
Example 2:


	Input: s = ")()())"
	Output: 4
	Explanation: The longest valid parentheses substring is "()()".

Example 3:


	Input: s = ""
	Output: 0
	



Note:

	0 <= s.length <= 3 * 10^4
	s[i] is '(', or ')'.


### 解析


根据题意，给定一个仅包含字符 '(' 和 ')' 的字符串，找出最长有效的括号子字符串的长度。

这道题的题目很简单，我们一看到这种括号类型的题目，基本上解决方法也就确定了，那就是用栈，因为左右括号的配对，我们是用栈顶元素和新来的元素就能进行很好的判断，决定括号该保留还是该出栈。

这里我们要找的是合法字符串的最长长度，按照常规思路，我们要计算长度，肯定要知道左边界和右边界，右边界肯定是新来的括号，左边界就需要我们保存在栈里面，左边界其实就是我们已经遍历过的左括号的索引，所以：

* 遇到左括号的时候，我们就将其索引入栈
* 遇到右括号的时候，有两种情况：第一种情况是如果栈不为空，说明当前右括号有可以配对的左括号，我们用此时的右括号索引减去栈顶的左括号索引，就是当前合法子字符串的长度，同时更新结果值；第二种情况就是栈为空，说明当前右括号没有可以配对的左括号，直接将其索引入栈，表示已经遍历过的字符在这个位置不合法，也就是当前最后一个没有成功匹配的右括号索引。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。



### 解答
				

	class Solution(object):
	    def longestValidParentheses(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        result = 0
	        stack = [-1]
	        for i, c in enumerate(s):
	            if c == '(':
	                stack.append(i)
	            elif c == ')':
	                stack.pop()
	                if not stack:
	                    stack.append(i)
	                else:
	                    result = max(result, i-stack[-1])
	        return result
			
### 运行结果


	Runtime: 49 ms, faster than 42.05% of Python online submissions for Longest Valid Parentheses.
	Memory Usage: 14.3 MB, less than 38.15% of Python online submissions for Longest Valid Parentheses.
	
### 解析


当然因为这道题是求“最值”类型的题目，我们也可以用动态规划来尝试解决，定义 dp[i] 表示索引为 i 的字符位结尾时候出现的最长合法括号子字符串的长度。

* 当 s[i] = '(' 的时候，我们知道肯定 dp[i] 为 0 ，因为当前位置还无法形成合法的括号组合
* 当 s[i] = ')' 的时候，我们有两种情况需要讨论：
* （一）当 s[i-1] 为 '(' 的时候，我们知道此时的  s[i-1] 和  s[i] 可以形成合法的括号此时 dp[i] 最少为 2 ，但是在 s[i-1] 之前也有可能有相邻的合法的子字符串，也可有可能不是，这都无所谓，我们只需要加上即可，此时的 
		
		dp[i] = dp[i-2] + 2 
* （二）当 s[i-1] 为 ')' 的时候，我们知道此时的 i-dp[i-1]-1 到 i-1 的范围内是合法的子字符串，同时  i-dp[i-1]-1  之前的以 i-dp[i-1]-2 位结尾的子字符串也可能有合法的子字符串，所以 

		dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]



时间复杂度为 O(N) ，空间复杂度为 O(N) 。



### 解答
				

	class Solution(object):
	    def longestValidParentheses(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        N = len(s)
	        dp = [0] * N
	        result = 0
	        for i in range(1, N):
	            c = s[i]
	            if c == ')':
	                if s[i - 1] == '(':
	                    dp[i] = 2
	                    if i >= 2:
	                        dp[i] = dp[i - 2] + dp[i]
	                elif dp[i - 1] > 0:
	                    if i - dp[i - 1] - 1 >= 0 and s[i - dp[i - 1] - 1] == '(':
	                        dp[i] = dp[i - 1] + 2
	                        if i - dp[i - 1] - 2 >= 0:
	                            dp[i] = dp[i] + dp[i - dp[i - 1] - 2]
	            result = max(result, dp[i])
	        return result
	

			
### 运行结果


	Runtime: 52 ms, faster than 36.85% of Python online submissions for Longest Valid Parentheses.
	Memory Usage: 14.2 MB, less than 38.15% of Python online submissions for Longest Valid Parentheses.

### 原题链接

https://leetcode.com/problems/longest-valid-parentheses/

您的支持是我最大的动力
